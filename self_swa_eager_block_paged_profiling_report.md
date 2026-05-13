# Self-SWA Eager Block-Paged Profiling Report

This report compares eager-mode self-SWA speculative decoding on
`zai-org/glm-4-9b-chat-1m` with a 100k-token prompt, 8k self-SWA window, and
512 output tokens on ROCm with the AITER attention backend.

The main comparison is between the default exact sink=4 path, which gathers
sink+recent KV and runs dense FMHA, and the opt-in block-aligned paged path
enabled with `VLLM_SELF_SWA_BLOCK_ALIGNED_PAGED_ATTN=1`.

## Run Configurations

Common environment:

```bash
HIP_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0
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
  --num-spec-tokens 4 \
  --attention-backend ROCM_AITER_FA \
  --enforce-eager \
  --tp 1
```

Warmup runs used the same configuration with `--output-len 16` before the
profiled runs. The warmups reduced some global first-use effects, but each
profiled run still used a fresh engine process, so per-process AITER module
loads/JIT-like setup remain visible in the first few draft scopes.

Trace directories:

```text
default exact sink=4 eager TP=1:
/tmp/vllm-self-swa-eager-default-o512

opt-in block-aligned paged sink=4 eager TP=1:
/tmp/vllm-self-swa-eager-block-paged-o512
```

Run logs:

```text
/tmp/self_swa_eager_profile_logs/default_exact_o512_profile.log
/tmp/self_swa_eager_profile_logs/block_paged_o512_profile.log
```

## High-Level Results

| Run | Decode tokens | Decode time | Throughput | Speedup vs same-run baseline | Exact match | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Baseline eager TP=1, default run | 511 | 6.82s | 74.91 tok/s | 1.00x | N/A | N/A |
| Default exact self-SWA sink=4 eager TP=1 | 511 | 20.72s | 24.66 tok/s | 0.33x | PASS | 4.72 mean length |
| Baseline eager TP=1, opt-in run | 511 | 6.56s | 77.88 tok/s | 1.00x | N/A | N/A |
| Block-aligned paged self-SWA sink=4 eager TP=1 | 511 | 14.66s | 34.86 tok/s | 0.45x | PASS | 4.86 mean length |

The block-aligned paged path improved eager self-SWA throughput from
24.66 tok/s to 34.86 tok/s, a 1.41x speedup over the default exact sink=4 path.
It still trails the eager baseline because the self-SWA path remains eager,
does extra draft/verification work, and has significant CPU/scheduling overhead.

Default exact self-SWA metrics:

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

Block-aligned paged self-SWA metrics:

```text
num_drafts: 106
num_draft_tokens: 424
num_accepted_tokens: 409
mean_acceptance_length: 4.86
acceptance_at_token_0: 0.99
acceptance_at_token_1: 0.98
acceptance_at_token_2: 0.94
acceptance_at_token_3: 0.94
```

## Trace Phase Isolation

Profiler traces capture the full `generate()` call. Kernel attribution here uses
the CPU launch scope associated with each kernel external id.

Prefill scopes:

```text
execute_context_1(16384)_generation_0(0)
execute_context_1(1696)_generation_0(0)
```

Decode/verification scopes:

```text
execute_context_0(0)_generation_1(...)
```

Draft scopes:

```text
gpu_model_runner: draft
```

## Baseline Decode Kernel Breakdown

Configuration: eager TP=1 baseline trace from the default run. Normalized by
511 decode tokens.

```text
execute_context decode scopes: 511
decode scope wall: 6830.852 ms
kernel total launched from decode: 4303.161 ms
kernel avg/decode token: 8.421 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/token |
| --- | ---: | ---: | ---: | ---: |
| `wvSplitK_hf_sml bf16 GEMM` | 82,271 | 2133.986 | 25.939 us | 4.176 |
| `paged_attention_ll4mi_QKV_mfma16` | 20,440 | 1225.951 | 59.978 us | 2.399 |
| `paged_attention_ll4mi_reduce` | 20,440 | 281.660 | 13.780 us | 0.551 |
| `RMSNorm CK kernel` | 41,391 | 189.611 | 4.581 us | 0.371 |
| `vectorized add bf16` | 40,880 | 188.720 | 4.616 us | 0.369 |
| `act_and_mul / SiLU` | 20,440 | 118.338 | 5.790 us | 0.232 |
| `rotary_embedding` | 20,440 | 72.677 | 3.556 us | 0.142 |
| `reshape_and_cache_flash` | 20,440 | 59.020 | 2.887 us | 0.115 |
| `__amd_rocclr_copyBuffer` | 5,110 | 14.003 | 2.740 us | 0.027 |

### Baseline Decode CPU Wall Gap

This uses the default-run baseline trace:

```text
/tmp/vllm-self-swa-eager-default-o512/baseline_prompt_len_100000_rank0.1778642720322306117.pt.trace.json.gz
```

Kernel attribution maps kernels back to decode-scope CPU events by
`External id`. That gives 4301.493 ms of decode-launched kernels, 1.668 ms
below the kernel table above. The 0.04% difference is a trace correlation
boundary/detail and does not affect the gap profile.

| Slice | CPU wall | Correlated kernels | Wall-kernel gap |
| --- | ---: | ---: | ---: |
| All baseline decode | 6830.852 ms | 4301.493 ms | 2529.359 ms |
| First decode scope | 266.802 ms | 8.453 ms | 258.349 ms |
| Steady scopes 2-511 | 6564.051 ms | 4293.041 ms | 2271.010 ms |

First 10 decode-scope CPU wall durations:

```text
266.802 ms
13.103 ms
12.900 ms
12.497 ms
12.837 ms
12.954 ms
12.687 ms
12.990 ms
12.496 ms
12.979 ms
```

First 10 decode-scope kernel totals:

```text
8.453 ms
8.384 ms
8.416 ms
8.397 ms
8.416 ms
8.406 ms
8.407 ms
8.403 ms
8.398 ms
8.416 ms
```

The first baseline decode scope is setup-heavy. Its exclusive CPU wall is
dominated by a 4-byte device-to-host `hipMemcpyWithStream` under
`gpu_model_runner: preprocess`, which accounts for 190.874 ms of the
266.802 ms scope. After that first scope, scalar sync/D2H drops to about
0.107 ms/token.

CPU exclusive wall by component:

| Component | All decode | First scope | Steady scopes 2-511 |
| --- | ---: | ---: | ---: |
| Scalar sync / D2H copies via `aten::item` / `_local_scalar_dense` / `hipMemcpyWithStream` | 245.842 ms | 191.435 ms | 54.407 ms |
| HIP launch runtime, mostly `hipLaunchKernel` | 1727.132 ms | 61.748 ms | 1665.384 ms |
| ATen metadata/index ops | 796.846 ms | 1.853 ms | 794.993 ms |
| Attention CPU wrappers | 989.491 ms | 3.014 ms | 986.477 ms |
| Uninstrumented Python/C++ gaps | 30.511 ms | 0.127 ms | 30.384 ms |
| Model forward CPU wrappers | 1642.170 ms | 5.137 ms | 1637.033 ms |
| Other CPU/HIP/runtime bookkeeping | 1398.861 ms | 3.488 ms | 1395.373 ms |

Largest exclusive CPU contributors:

| Event | Component | Exclusive wall |
| --- | --- | ---: |
| `hipLaunchKernel` | HIP launch runtime | 1722.970 ms |
| `gpu_model_runner: forward` | Model forward CPU wrappers | 1461.937 ms |
| `aiter::paged_attention_v1` | Attention CPU wrappers | 661.734 ms |
| `vllm::rocm_unquantized_gemm` | Other CPU/runtime | 552.598 ms |
| `vllm::unified_attention_with_output` | Attention CPU wrappers | 327.757 ms |
| `hipMemcpyWithStream` | Scalar sync / D2H | 203.218 ms |
| `_rocm_C::wvSplitK` | Other CPU/runtime | 208.174 ms |
| `aiter::rms_norm` | Other CPU/runtime | 192.669 ms |
| `gpu_model_runner: preprocess` | Model forward CPU wrappers | 167.749 ms |
| `vllm::unified_kv_cache_update` | Other CPU/runtime | 161.914 ms |

The opt-in run's separate baseline trace is consistent:

```text
/tmp/vllm-self-swa-eager-block-paged-o512/baseline_prompt_len_100000_rank0.1778643269083128339.pt.trace.json.gz
```

| Baseline trace | Decode scopes | CPU wall | Kernel total | Wall-kernel gap | Steady gap/token excluding first |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default-run baseline | 511 | 6830.852 ms | 4301.493 ms | 2529.359 ms | 4.453 ms |
| Opt-in-run baseline | 511 | 6572.002 ms | 4296.015 ms | 2275.987 ms | 3.957 ms |

Compared with block-aligned paged self-SWA draft, baseline decode is much more
GPU-kernel dominated. Baseline decode spends roughly 63% of wall time in
correlated kernels, or 65% after excluding the first setup-heavy scope. The
block-aligned paged draft path has 4278.203 ms of draft-correlated kernels
against 20939.670 ms of raw CPU draft wall, so most of its remaining cost is
host-side eager scheduling, launch/runtime overhead, and speculative decode
bookkeeping rather than the paged-attention kernels themselves. This is why
baseline decode reaches much higher throughput even though it uses the same
basic paged-attention kernel family: it executes one target-model decode path
per token with far less draft/verification orchestration and a much smaller
CPU wall/kernel gap.

The baseline path is the standard eager decode hot path. The block-aligned
self-SWA path instead runs inside `SelfSWAProposer.propose`, serially drafts
tokens, updates positions, slot mappings, sequence lengths, and per-layer
attention metadata, builds synthetic block tables with `arange`, `where`,
`gather`, `empty`, and window arithmetic, calls model forward with
`CUDAGraphMode.NONE`, and then performs greedy sampling plus speculative decode
bookkeeping. The attention change fixed the draft GPU attention bottleneck; the
remaining gap is mostly host-side eager orchestration such as `hipLaunchKernel`,
ATen metadata/index ops, attention wrappers, and model wrappers. The next likely
win is making the draft metadata/static buffers graph-capturable or otherwise
cheaper to update.

## Default Exact Self-SWA Verification Decode

This is target verification decode, normalized by 109 spec iterations.

```text
execute_context decode scopes: 109
decode scope wall: 2287.962 ms
kernel total launched from verification: 1452.218 ms
kernel avg/spec step: 13.323 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/spec step |
| --- | ---: | ---: | ---: | ---: |
| `kernel_unified_attention_3d` | 4,360 | 677.424 | 155.372 us | 6.215 |
| `Cijk GEMM 32x16x512` | 4,469 | 321.542 | 71.949 us | 2.950 |
| `Cijk GEMM 64x16x256` | 4,360 | 143.740 | 32.968 us | 1.319 |
| `Cijk GEMM 64x16x128` | 4,360 | 66.818 | 15.325 us | 0.613 |
| `Cijk GEMM 16x16x512` | 4,360 | 63.061 | 14.464 us | 0.579 |
| `RMSNorm CK kernel` | 8,829 | 41.980 | 4.755 us | 0.385 |
| `vectorized add bf16` | 8,720 | 40.852 | 4.685 us | 0.375 |
| `act_and_mul / SiLU` | 4,360 | 28.405 | 6.515 us | 0.261 |
| `reduce_segments` | 4,360 | 25.566 | 5.864 us | 0.235 |

## Default Exact Self-SWA Draft Decode

This is the default exact sink=4 drafter path, normalized by 436 drafted tokens.

```text
draft scopes: 115
raw CPU draft wall: 27025.693 ms
kernel total launched from draft: 10855.946 ms
kernel avg/drafted token: 24.899 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/drafted token |
| --- | ---: | ---: | ---: | ---: |
| `fmha_fwd_hd128_bf16_rtna_group` | 18,240 | 5712.845 | 313.204 us | 13.103 |
| `wvSplitK_hf_sml bf16 GEMM` | 73,416 | 1949.279 | 26.551 us | 4.471 |
| `cp_mha_gather_cache` | 18,240 | 567.453 | 31.110 us | 1.302 |
| `__amd_rocclr_copyBuffer` | 74,105 | 311.726 | 4.207 us | 0.715 |
| `RMSNorm CK kernel` | 36,936 | 162.879 | 4.410 us | 0.374 |
| `vectorized add bf16` | 36,480 | 162.819 | 4.463 us | 0.373 |
| `act_and_mul / SiLU` | 18,240 | 107.856 | 5.913 us | 0.247 |
| `rotary_embedding` | 18,240 | 82.602 | 4.529 us | 0.189 |
| `reshape_and_cache_flash` | 18,240 | 81.028 | 4.442 us | 0.186 |

The default draft bottleneck remains dense FMHA plus explicit KV materialization:

```text
fmha_fwd_hd128_bf16_rtna_group: 5712.845 ms
cp_mha_gather_cache:             567.453 ms
__amd_rocclr_copyBuffer:         311.726 ms
```

## Block-Aligned Paged Self-SWA Verification Decode

This is target verification decode for the opt-in run, normalized by 106 spec
iterations. Verification is similar to the default path; the main change is in
the drafter.

```text
execute_context decode scopes: 106
decode scope wall: 2032.974 ms
kernel total launched from verification: 1420.663 ms
kernel avg/spec step: 13.403 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/spec step |
| --- | ---: | ---: | ---: | ---: |
| `kernel_unified_attention_3d` | 4,240 | 661.006 | 155.898 us | 6.236 |
| `Cijk GEMM 32x16x512` | 4,346 | 312.256 | 71.849 us | 2.946 |
| `Cijk GEMM 64x16x256` | 4,240 | 139.564 | 32.916 us | 1.317 |
| `Cijk GEMM 64x16x128` | 4,240 | 65.015 | 15.334 us | 0.613 |
| `Cijk GEMM 16x16x512` | 4,240 | 61.247 | 14.445 us | 0.578 |
| `RMSNorm CK kernel` | 8,586 | 42.856 | 4.991 us | 0.404 |
| `vectorized add bf16` | 8,480 | 39.907 | 4.706 us | 0.376 |
| `act_and_mul / SiLU` | 4,240 | 27.873 | 6.574 us | 0.263 |
| `reduce_segments` | 4,240 | 24.815 | 5.853 us | 0.234 |

## Block-Aligned Paged Self-SWA Draft Decode

This is the opt-in block-aligned paged drafter path, normalized by 424 drafted
tokens.

```text
draft scopes: 112
raw CPU draft wall: 20939.670 ms
kernel total launched from draft: 4281.337 ms
kernel avg/drafted token: 10.097 ms
```

| Kernel | Calls | Total ms | Avg/call | ms/drafted token |
| --- | ---: | ---: | ---: | ---: |
| `wvSplitK_hf_sml bf16 GEMM` | 71,484 | 1920.787 | 26.870 us | 4.530 |
| `paged_attention_ll4mi_QKV_mfma16` | 17,760 | 212.936 | 11.990 us | 0.502 |
| `__amd_rocclr_copyBuffer` | 36,635 | 177.483 | 4.845 us | 0.419 |
| `RMSNorm CK kernel` | 35,964 | 159.005 | 4.421 us | 0.375 |
| `vectorized add bf16` | 35,520 | 124.475 | 3.504 us | 0.294 |
| `act_and_mul / SiLU` | 17,760 | 105.149 | 5.921 us | 0.248 |
| `paged_attention_ll4mi_reduce` | 17,760 | 76.248 | 4.293 us | 0.180 |
| `reshape_and_cache_flash` | 17,760 | 68.762 | 3.872 us | 0.162 |
| `rotary_embedding` | 17,760 | 59.682 | 3.360 us | 0.141 |

Draft attention changed from dense FMHA plus gather to paged attention:

```text
default draft fmha_fwd_hd128_bf16_rtna_group: 5712.845 ms
default draft cp_mha_gather_cache:             567.453 ms

block-paged draft paged_attention_ll4mi_QKV:    212.936 ms
block-paged draft paged_attention_ll4mi_reduce:  76.248 ms
block-paged draft cp_mha_gather_cache:            0.000 ms
```

## First-Use Setup and CPU Wall Gap

The first draft scopes still include per-process setup cost despite running a
short warmup in earlier processes.

Default exact first draft scope durations:

```text
510.373 ms
1097.039 ms
1258.714 ms
1544.151 ms
1835.577 ms
2117.974 ms
362.804 ms
```

These first 7 scopes sum to 8726.632 ms.

Block-aligned paged first draft scope durations:

```text
511.503 ms
1029.235 ms
1211.895 ms
1494.718 ms
1782.523 ms
2065.401 ms
312.588 ms
```

These first 7 scopes sum to 8407.863 ms.

Raw CPU draft wall/kernel gaps from the kernel breakdown tables:

| Run | Raw draft wall | Draft kernel total | Raw gap |
| --- | ---: | ---: | ---: |
| Default exact sink=4 | 27025.693 ms | 10855.946 ms | 16169.747 ms |
| Block-aligned paged sink=4 | 20939.670 ms | 4281.337 ms | 16658.333 ms |

The opt-in path removes most of the draft kernel bottleneck, but eager-mode
CPU/scheduling overhead remains large. A second pass over the block-aligned
trace maps kernels back to draft-scope CPU events by `External id`:

```text
/tmp/vllm-self-swa-eager-block-paged-o512/self-SWA_prompt_len_100000_window_size_8192_rank0.1778643437876587306.pt.trace.json.gz
```

That correlation attributes 4278.203 ms of kernels to draft scopes, 3.134 ms
below the kernel-breakdown total above. The 0.07% difference is a trace
correlation boundary/detail and does not change the gap profile.

| Slice | Draft scopes | CPU wall | Correlated kernels | Wall-kernel gap | Wall/scope | Kernel/scope | Gap/scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All block-aligned draft | 112 | 20939.670 ms | 4278.203 ms | 16661.468 ms | 186.961 ms | 38.198 ms | 148.763 ms |
| First 7 setup-heavy scopes | 7 | 8407.862 ms | 227.591 ms | 8180.271 ms | 1201.123 ms | 32.513 ms | 1168.610 ms |
| Steady scopes 8-112 | 105 | 12531.808 ms | 4050.611 ms | 8481.196 ms | 119.351 ms | 38.577 ms | 80.773 ms |

CPU exclusive wall attribution for block-aligned draft:

| Component | All draft | First 7 | Steady scopes 8-112 |
| --- | ---: | ---: | ---: |
| Scalar sync / D2H copies via `aten::item` / `_local_scalar_dense` | 8490.357 ms | 7597.631 ms | 892.727 ms |
| HIP launch runtime, mostly `hipLaunchKernel` | 4811.556 ms | 381.566 ms | 4429.990 ms |
| Aten metadata/index ops (`arange`, `where`, `gather`, `empty`, window arithmetic) | 2321.651 ms | 126.618 ms | 2195.033 ms |
| Attention CPU wrappers | 2137.691 ms | 116.483 ms | 2021.207 ms |
| Uninstrumented Python/C++ gaps | 1535.590 ms | 87.853 ms | 1447.736 ms |
| Model forward CPU wrappers | 1352.557 ms | 72.995 ms | 1279.562 ms |
| Other CPU/HIP bookkeeping | 290.269 ms | 24.716 ms | 265.562 ms |

The first 7 block-aligned draft scopes are dominated by setup-time scalar
extraction syncs and one-byte device-to-host copies, not draft kernels. The
first draft scope alone takes 511.503 ms of CPU wall while only 0.068 ms of
draft-correlated kernels run; 480.131 ms is one blocking one-byte D2H copy.

In this table, "first 7" means the first seven block-aligned draft scopes in
the profiled process. They are cold/setup-heavy outliers, dominated by blocking
scalar syncs and device-to-host copies rather than steady draft compute. They
matter for cold first-request latency and short benchmark wall time, but they
should be reported separately or discarded for kernel and serving steady-state
comparisons. In a long-lived serving process, in-process warmup should amortize
or remove most of this cost before measuring steady decode.

After setup, the largest residual is eager launch overhead. Steady scopes
8-112 contain 692,055 `hipLaunchKernel` calls taking 4425.788 ms, about
6.4 us per launch. Metadata/index construction and attention wrapper CPU time
are the next largest buckets. There are no NCCL/all-reduce events inside the
draft scopes. This attribution is limited by the trace lacking Python stacks,
operator shapes, and memory metadata.

## Main Findings

1. Eager baseline decode uses optimized paged attention:
   `paged_attention_ll4mi_QKV_mfma16` and
   `paged_attention_ll4mi_reduce`.

2. Default exact sink=4 self-SWA draft decode still uses the sink gather path:
   materialize visible sink+recent KV, then run dense varlen FMHA.

3. The default exact draft bottleneck is dominated by
   `fmha_fwd_hd128_bf16_rtna_group`, which costs 5712.845 ms total, or
   13.103 ms per drafted token.

4. The opt-in block-aligned paged path removes draft dense FMHA and
   `cp_mha_gather_cache`; draft attention becomes
   `paged_attention_ll4mi_QKV_mfma16` plus `paged_attention_ll4mi_reduce`.

5. Draft kernel time drops from 10855.946 ms to 4281.337 ms, a 2.54x reduction.

6. End-to-end eager decode throughput improves from 24.66 tok/s to 34.86 tok/s,
   a 1.41x speedup for self-SWA sink=4.

7. Both self-SWA paths exact-match the eager greedy baseline for this workload.

8. Eager CPU/scheduling overhead is now a major residual cost. Even after the
   attention kernel improvement, the block-aligned paged path spends far more
   CPU draft wall time than GPU kernel time.

## Validation

Focused unit tests:

```bash
.venv/bin/python -m pytest tests/v1/spec_decode/test_self_swa.py -v
```

Result:

```text
7 passed
```

The pytest cache could not be written because the workspace cache directory is
not writable from this session, but the tests themselves passed.
