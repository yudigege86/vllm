# Self-SWA Profiling Report

This report summarizes profiling work for self-SWA speculative decoding on
`zai-org/glm-4-9b-chat-1m` with a 100k-token prompt, 8k self-SWA window, and
512 output tokens on ROCm with the AITER attention backend.

The detailed kernel analysis focuses on eager TP=1 rank-0 traces, because that
configuration removes tensor-parallel communication noise and makes the sink
path behavior easier to isolate.

## Run Configurations

Common settings:

```bash
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
VLLM_RPC_TIMEOUT=1800000
```

Common workload:

```bash
.venv/bin/python examples/features/speculative_decoding/self_swa_offline.py \
  --prompt-lens 100k \
  --self-swa-window-sizes 8192 \
  --output-len 512 \
  --num-prompts 1 \
  --attention-backend ROCM_AITER_FA \
  --enforce-eager \
  --tp 1
```

Trace directories:

```text
sink=4 eager TP=1:
/tmp/vllm-self-swa-profile-100k-w8k-o512-eager-tp1

sink=0 eager TP=1:
/tmp/vllm-self-swa-profile-100k-w8k-o512-eager-tp1-sink0

original TP=8 sink=4:
/tmp/vllm-self-swa-profile-100k-w8k-o512
```

## High-Level Results

| Run | Decode tokens | Decode time | Throughput | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline eager TP=1 | 511 | 6.80-6.84s | 74.66-75.18 tok/s | 1.00x | N/A |
| Self-SWA sink=4 eager TP=1 | 511 | 19.96s | 25.60 tok/s | 0.34x | 4.72 mean length |
| Self-SWA sink=0 eager TP=1 | 511 | 7.62s | 67.09 tok/s | 0.89x | 4.62 mean length |
| Original TP=8 sink=4 | 511 | 30.03s | 17.01 tok/s | 0.57x | 4.72 mean length |

Self-SWA sink=4 metrics:

```text
num_drafts: 109
num_draft_tokens: 436
num_accepted_tokens: 406
mean_acceptance_length: 4.72
acceptance_at_token_0: 0.94
acceptance_at_token_1: 0.93
acceptance_at_token_2: 0.93
acceptance_at_token_3: 0.93
```

Self-SWA sink=0 metrics:

```text
num_drafts: 111
num_draft_tokens: 444
num_accepted_tokens: 402
mean_acceptance_length: 4.62
acceptance_at_token_0: 0.93
acceptance_at_token_1: 0.91
acceptance_at_token_2: 0.89
acceptance_at_token_3: 0.89
```

## Trace Phase Isolation

The profiler captures the full `generate()` call, so prefill and decode must be
separated before interpreting kernel costs.

Prefill scopes:

```text
execute_context_1(16384)_generation_0(0)
execute_context_1(1696)_generation_0(0)
```

Decode scopes:

```text
execute_context_0(0)_generation_1(...)
gpu_model_runner: draft
```

For self-SWA, target verification and draft work are separate decode components.
Kernel attribution should use CPU launch scope where possible; raw timestamp
overlap can be misleading because HIP kernels execute asynchronously.

## Prefill Findings

The large FMHA kernels in the sink=0 trace are prefill/context kernels, not
decode kernels.

Representative prefill kernel pattern:

| Kernel | Calls | Total time | Avg/call | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `fmha_fwd_hd128_bf16_rtna_group` | 480 | 4404-4409 ms | ~9.18 ms | Prefill/context only |
| `Cijk GEMM 256x192x64` | 480 | 1869-1870 ms | ~3.89 ms | Prefill/context only |
| `fmha_fwd_hd128_bf16_causal_rtna_group` | 280 | 977-978 ms | ~3.49 ms | Prefill/context only |
| `merge_attn_states_kernel` | 480 | 139-140 ms | ~291 us | Prefill/context only |
| `cp_mha_gather_cache` | 480 | 44 ms | ~92 us | Prefill/context only |

In the sink=0 self-SWA trace, every `fmha_fwd_hd128_bf16_rtna_group` launch
mapped back to `aiter::fmha_v3_varlen_fwd` under prefill execute contexts:

```text
_ZN5aiter30fmha_fwd_hd128_bf16_rtna_groupE.kd
count: 480
kernel total: 4404.6 ms
launch scope: gpu_model_runner: forward
execute_context:
  360 under execute_context_1(16384)_generation_0(0)
  120 under execute_context_1(1696)_generation_0(0)
```

## Baseline Decode Kernel Breakdown

Configuration: eager TP=1, sink=4 baseline trace. Normalized by 511 decode
tokens.

```text
execute_context decode scopes: 513
raw trace wall: 19424.279 ms
kernel total: 4297.346 ms
kernel avg/token: 8.410 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/token |
| --- | ---: | ---: | ---: | ---: |
| `wvSplitK_hf_sml bf16 GEMM` | 82,271 | 2136.265 | 25.966 us | 4.181 |
| `paged_attention_ll4mi_QKV_mfma16` | 20,440 | 1224.962 | 59.930 us | 2.397 |
| `paged_attention_ll4mi_reduce` | 20,440 | 281.664 | 13.780 us | 0.551 |
| `RMSNorm CK kernel` | 41,391 | 193.680 | 4.679 us | 0.379 |
| `vectorized add bf16` | 40,880 | 188.807 | 4.619 us | 0.369 |
| `act_and_mul / SiLU` | 20,440 | 115.434 | 5.647 us | 0.226 |
| `rotary_embedding` | 20,440 | 67.335 | 3.294 us | 0.132 |
| `reshape_and_cache_flash` | 20,440 | 57.417 | 2.809 us | 0.112 |
| `__amd_rocclr_copyBuffer` | 5,110 | 14.155 | 2.770 us | 0.028 |

## Self-SWA Sink=4 Verification Decode

This is target verification decode, normalized by 109 spec iterations.

```text
execute_context decode scopes: 110
raw trace wall: 2577.699 ms
kernel total: 1458.072 ms
kernel avg/spec step: 13.377 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/spec step |
| --- | ---: | ---: | ---: | ---: |
| `kernel_unified_attention_3d` | 4,360 | 678.769 | 155.681 us | 6.227 |
| `Cijk GEMM 32x16x512` | 4,469 | 320.292 | 71.670 us | 2.938 |
| `Cijk GEMM 64x16x256` | 4,360 | 143.413 | 32.893 us | 1.316 |
| `Cijk GEMM 64x16x128` | 4,360 | 66.919 | 15.348 us | 0.614 |
| `Cijk GEMM 16x16x512` | 4,360 | 62.608 | 14.360 us | 0.574 |
| `RMSNorm CK kernel` | 8,829 | 43.705 | 4.950 us | 0.401 |
| `vectorized add bf16` | 8,720 | 40.774 | 4.676 us | 0.374 |
| `act_and_mul / SiLU` | 4,360 | 28.434 | 6.522 us | 0.261 |
| `reduce_segments` | 4,360 | 25.435 | 5.834 us | 0.233 |
| `reshape_and_cache_flash` | 4,360 | 17.916 | 4.109 us | 0.164 |
| `rotary_embedding` | 4,360 | 17.749 | 4.071 us | 0.163 |

## Self-SWA Sink=4 Draft Decode

This is the drafter path, normalized by 436 drafted tokens.

```text
draft scopes: 115
raw draft wall: 26013.668 ms
kernel total: 11004.109 ms
kernel avg/drafted token: 25.239 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/drafted token |
| --- | ---: | ---: | ---: | ---: |
| `fmha_fwd_hd128_bf16_rtna_group` | 18,240 | 5719.130 | 313.549 us | 13.117 |
| `wvSplitK_hf_sml bf16 GEMM` | 73,416 | 1956.548 | 26.650 us | 4.487 |
| `cp_mha_gather_cache` | 18,240 | 565.006 | 30.976 us | 1.296 |
| `__amd_rocclr_copyBuffer` | 74,105 | 334.088 | 4.508 us | 0.766 |
| `FillFunctor elementwise` | 91,200 | 279.994 | 3.070 us | 0.642 |
| `CUDAFunctor_add int` | 54,720 | 241.616 | 4.415 us | 0.554 |
| `RMSNorm CK kernel` | 36,936 | 163.460 | 4.425 us | 0.375 |
| `vectorized add bf16` | 36,480 | 162.134 | 4.444 us | 0.372 |
| `direct_copy elementwise` | 37,051 | 138.831 | 3.747 us | 0.318 |
| `rocprim scan/trampoline` | 36,480 | 132.916 | 3.644 us | 0.305 |
| `compute_cuda_kernel<long>` | 18,240 | 113.149 | 6.203 us | 0.260 |
| `act_and_mul / SiLU` | 18,240 | 105.542 | 5.786 us | 0.242 |
| `rotary_embedding` | 18,240 | 83.542 | 4.580 us | 0.192 |
| `reshape_and_cache_flash` | 18,240 | 82.981 | 4.549 us | 0.190 |

The explicit KV gather/copy rows are:

| Kernel | Calls | Total ms | Avg/call | ms/drafted token |
| --- | ---: | ---: | ---: | ---: |
| `cp_mha_gather_cache` | 18,240 | 565.006 | 30.976 us | 1.296 |
| `__amd_rocclr_copyBuffer` | 74,105 | 334.088 | 4.508 us | 0.766 |

Sink=4 draft is dominated by dense FMHA plus explicit KV gather/copy.

## Self-SWA Sink=0 Draft Decode

This analysis attributes kernels by CPU launch inside `gpu_model_runner: draft`,
not by raw timestamp overlap. Normalized by 444 drafted tokens.

```text
draft scope count: 118
steady draft wall after excluding first-use JIT scopes: 5402.587 ms
kernel total launched from draft: 3979.587 ms
avg kernel time/drafted token: 8.963 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/drafted token |
| --- | ---: | ---: | ---: | ---: |
| `wvSplitK_hf_sml bf16 GEMM` | 75,348 | 2017.684 | 26.778 us | 4.544 |
| `paged_attention_ll4mi_QKV_mfma16` | 18,720 | 1105.341 | 59.046 us | 2.490 |
| `paged_attention_ll4mi_reduce` | 18,080 | 249.928 | 13.823 us | 0.563 |
| `RMSNorm CK kernel` | 37,908 | 166.451 | 4.391 us | 0.375 |
| `vectorized add bf16` | 37,440 | 165.963 | 4.433 us | 0.374 |
| `act_and_mul / SiLU` | 18,720 | 106.806 | 5.705 us | 0.241 |
| `rotary_embedding` | 18,720 | 70.799 | 3.782 us | 0.159 |
| `reshape_and_cache_flash` | 18,720 | 67.782 | 3.621 us | 0.153 |
| `argmax reduce bf16` | 468 | 6.174 | 13.192 us | 0.014 |
| `__amd_rocclr_copyBuffer` | 1,175 | 4.329 | 3.684 us | 0.010 |

Grouped by CPU launch op:

| CPU op | Kernel calls | Total ms | Avg/kernel | ms/drafted token |
| --- | ---: | ---: | ---: | ---: |
| `_rocm_C::wvSplitK` | 75,348 | 2017.684 | 26.778 us | 4.544 |
| `aiter::paged_attention_v1` | 37,440 | 1361.526 | 36.366 us | 3.067 |
| `aiter::rms_norm` | 37,908 | 166.451 | 4.391 us | 0.375 |
| `aten::add` | 37,558 | 166.332 | 4.429 us | 0.375 |
| `_C::silu_and_mul` | 18,720 | 106.806 | 5.705 us | 0.241 |
| `_C::rotary_embedding` | 18,720 | 70.799 | 3.782 us | 0.159 |
| `_C_cache_ops::reshape_and_cache_flash` | 18,720 | 67.782 | 3.621 us | 0.153 |

## First-Use JIT and Draft Wall Time

The raw `gpu_model_runner: draft` CPU annotations in the sink=0 trace were
inflated by first-use AITER paged-attention JIT/build work.

```text
CPU draft scopes: 118
raw summed CPU draft wall: 32924.235 ms
merged CPU draft wall:     32924.235 ms
```

First draft scope durations:

```text
512.337 ms
13469.123 ms
2906.104 ms
3174.923 ms
3474.918 ms
3743.358 ms
240.884 ms
```

Those 7 scopes sum to:

```text
27521.648 ms
```

Steady-state draft scopes:

```text
111 scopes
5402.587 ms total
48.672 ms / draft scope
12.168 ms / drafted token
```

The gap between steady draft wall and launched kernel time is therefore:

```text
steady draft wall:           5402.587 ms
kernel time launched draft:  3979.587 ms
gap:                         1423.000 ms
```

This gap is likely CPU overhead, scheduling, synchronization, launch overhead,
and non-kernel work.

## CUDAGraph Implications

CUDAGraphs could reduce some of the steady draft wall/kernel gap, but not the
kernel time itself.

Approximate upper bound from sink=0:

```text
gap: ~1423 ms total
~3.2 ms / drafted token
~12.8 ms / spec iteration
```

Current blocker: `SelfSWAProposer.propose` explicitly sets:

```python
cudagraph_runtime_mode=CUDAGraphMode.NONE
```

So the self-SWA draft path does not benefit from CUDAGraph capture today.

## Main Findings

1. Baseline decode uses optimized paged attention:
   `paged_attention_ll4mi_QKV_mfma16` and
   `paged_attention_ll4mi_reduce`.

2. Sink=4 self-SWA draft decode uses the sink path:
   gather visible sink+recent KV, then run dense varlen FMHA.

3. The sink=4 bottleneck is real:
   `fmha_fwd_hd128_bf16_rtna_group` alone costs 5719 ms total, or
   13.117 ms per drafted token.

4. KV gather/copy is also significant in sink=4:
   `cp_mha_gather_cache` costs 1.296 ms per drafted token and
   `__amd_rocclr_copyBuffer` costs 0.766 ms per drafted token.

5. Sink=0 changes the draft attention path to paged attention and removes most
   of the sink gather + dense FMHA cost.

6. Sink=0 improved self-SWA eager TP=1 from 25.60 tok/s to 67.09 tok/s.

7. FMHA seen in sink=0 traces is prefill-only after phase isolation, not draft
   decode.

## Recommended Next Steps

1. Keep sink=0 as the performance baseline and test it across prompt lengths,
   output lengths, and `num_spec_tokens` values.

2. If sink tokens are required, prototype a sink-aware paged attention kernel
   that can attend to `[sink] + [recent window]` without materializing gathered
   K/V buffers.

3. Evaluate graph capture for self-SWA draft after making the draft path stable
   enough to replace `CUDAGraphMode.NONE`.

4. For future profiles, avoid including first-use JIT compilation in steady-state
   timing. Run a warmup pass before starting the profiler or discard the first
   few draft scopes during analysis.
