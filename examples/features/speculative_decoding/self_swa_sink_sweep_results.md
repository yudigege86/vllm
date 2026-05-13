# Self-SWA Sink Size Sweep Results

Model: `zai-org/glm-4-9b-chat-1m`

Original sink sweep settings:

| Setting | Value |
|---|---:|
| Prompt lengths | `100k`, `1M`, `4M` |
| Window sizes | `8192`, `32768`, `131072` |
| Output length | `512` |
| Num prompts | `1` |
| Num speculative tokens | `4` |
| Tensor parallel size | `8` |
| Max model length | `4000517` |
| HF overrides | `seq_length=4194304`, `max_position_embeddings=4194304` |
| Attention backend | `ROCM_AITER_FA` |

Notes:

| Term | Meaning |
|---|---|
| Mean accept | `1 + num_accepted_tokens / num_drafts`; max is `5.0` for `num_spec_tokens=4`. |
| Accept rate | `num_accepted_tokens / num_draft_tokens`. |
| Exact match | Whether self-SWA output token IDs exactly matched the greedy baseline. |
| Skipped case | `prompt_len=100k`, `window_size=131072` was skipped because the window is larger than the prompt. |
| Block-aligned paged path | Opt-in TP=4 path using paged attention for self-SWA draft decode. `self_swa_sink_size=16` is the KV block size, and sink/recent regions are rounded to block boundaries, so this is not token-exact equivalent to the earlier `sink_size=4` sweep. |

## Sink Size 0

### Decode Speed

| Prompt length | Window size | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
|---:|---:|:---:|---:|---:|---:|---:|
| 100,000 | 8,192 | true | 121.26 | 64.07 | 0.53x | 4.61 |
| 1,000,000 | 8,192 | true | 57.91 | 31.03 | 0.54x | 2.80 |
| 4,000,000 | 8,192 | true | 21.51 | 7.50 | 0.35x | 2.17 |
| 100,000 | 32,768 | true | 121.26 | 80.67 | 0.67x | 4.62 |
| 1,000,000 | 32,768 | true | 57.91 | 36.43 | 0.63x | 2.90 |
| 4,000,000 | 32,768 | true | 21.51 | 7.62 | 0.35x | 2.20 |
| 100,000 | 131,072 | skipped | - | - | - | - |
| 1,000,000 | 131,072 | true | 57.91 | 36.72 | 0.63x | 2.95 |
| 4,000,000 | 131,072 | true | 21.51 | 8.17 | 0.38x | 2.38 |

### Acceptance

| Prompt length | Window size | Drafts | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates |
|---:|---:|---:|---:|---:|---:|---|
| 100,000 | 8,192 | 111 | 444 | 401 | 90.3% | 0.92, 0.91, 0.89, 0.89 |
| 1,000,000 | 8,192 | 184 | 736 | 331 | 45.0% | 0.47, 0.45, 0.45, 0.44 |
| 4,000,000 | 8,192 | 236 | 944 | 275 | 29.1% | 0.31, 0.29, 0.28, 0.28 |
| 100,000 | 32,768 | 111 | 444 | 402 | 90.5% | 0.93, 0.91, 0.89, 0.89 |
| 1,000,000 | 32,768 | 176 | 704 | 335 | 47.6% | 0.49, 0.47, 0.47, 0.47 |
| 4,000,000 | 32,768 | 232 | 928 | 279 | 30.1% | 0.32, 0.30, 0.29, 0.29 |
| 100,000 | 131,072 | skipped | - | - | - | - |
| 1,000,000 | 131,072 | 174 | 696 | 339 | 48.7% | 0.51, 0.48, 0.48, 0.48 |
| 4,000,000 | 131,072 | 216 | 864 | 298 | 34.5% | 0.37, 0.34, 0.34, 0.33 |

## Sink Size 4

### Decode Speed

| Prompt length | Window size | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
|---:|---:|:---:|---:|---:|---:|---:|
| 100,000 | 8,192 | true | 120.97 | 39.19 | 0.32x | 4.72 |
| 1,000,000 | 8,192 | true | 57.70 | 33.22 | 0.58x | 4.74 |
| 4,000,000 | 8,192 | true | 20.88 | 21.16 | 1.01x | 4.56 |
| 100,000 | 32,768 | true | 120.97 | 17.97 | 0.15x | 4.77 |
| 1,000,000 | 32,768 | true | 57.70 | 16.70 | 0.29x | 4.79 |
| 4,000,000 | 32,768 | true | 20.88 | 13.02 | 0.62x | 4.68 |
| 100,000 | 131,072 | skipped | - | - | - | - |
| 1,000,000 | 131,072 | true | 57.70 | 5.50 | 0.10x | 4.86 |
| 4,000,000 | 131,072 | true | 20.88 | 4.92 | 0.24x | 4.68 |

### Acceptance

| Prompt length | Window size | Drafts | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates |
|---:|---:|---:|---:|---:|---:|---|
| 100,000 | 8,192 | 109 | 436 | 406 | 93.1% | 0.94, 0.93, 0.93, 0.93 |
| 1,000,000 | 8,192 | 108 | 432 | 404 | 93.5% | 0.95, 0.94, 0.94, 0.92 |
| 4,000,000 | 8,192 | 112 | 448 | 399 | 89.1% | 0.92, 0.90, 0.88, 0.87 |
| 100,000 | 32,768 | 108 | 432 | 407 | 94.2% | 0.95, 0.94, 0.94, 0.94 |
| 1,000,000 | 32,768 | 107 | 428 | 406 | 94.9% | 0.96, 0.95, 0.94, 0.93 |
| 4,000,000 | 32,768 | 110 | 440 | 405 | 92.0% | 0.95, 0.93, 0.91, 0.90 |
| 100,000 | 131,072 | skipped | - | - | - | - |
| 1,000,000 | 131,072 | 106 | 424 | 409 | 96.5% | 0.98, 0.96, 0.96, 0.95 |
| 4,000,000 | 131,072 | 110 | 440 | 405 | 92.0% | 0.95, 0.93, 0.91, 0.90 |

## Block-Aligned Paged Self-SWA

This run exercised the opt-in block-aligned paged draft path, not the token-exact `sink_size=4` path above. The effective sink size is one KV block (`self_swa_sink_size=16`), and both sink and recent regions are rounded to block boundaries before dispatching paged attention. All completed cases exact-matched the greedy baseline.

Run settings:

| Setting | Value |
|---|---:|
| Prompt lengths | `100k`, `1M`, `4M` |
| Window sizes | `8192`, `32768`, `131072` |
| Output length | `512` |
| Num prompts | `1` |
| Num speculative tokens | `4` |
| Tensor parallel size | `4` |
| Max model length | `4000517` |
| HF overrides | `seq_length=4194304`, `max_position_embeddings=4194304` |
| Attention backend | `ROCM_AITER_FA` |
| Enforce eager | `true` |
| Self-SWA sink size | `16` |
| Env | `VLLM_SELF_SWA_BLOCK_ALIGNED_PAGED_ATTN=1`, `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False` |

### Decode Speed

| Prompt length | Window size | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
|---:|---:|:---:|---:|---:|---:|---:|
| 100,000 | 8,192 | true | 91.55 | 49.47 | 0.54x | 4.86 |
| 1,000,000 | 8,192 | true | 26.56 | 43.35 | 1.63x | 4.62 |
| 4,000,000 | 8,192 | true | 15.51 | 25.97 | 1.67x | 4.81 |
| 100,000 | 32,768 | true | 91.55 | 53.05 | 0.58x | 4.86 |
| 1,000,000 | 32,768 | true | 26.56 | 44.98 | 1.69x | 4.79 |
| 4,000,000 | 32,768 | true | 15.51 | 26.33 | 1.70x | 4.86 |
| 100,000 | 131,072 | skipped | - | - | - | - |
| 1,000,000 | 131,072 | true | 26.56 | 44.94 | 1.69x | 4.84 |
| 4,000,000 | 131,072 | true | 15.51 | 26.56 | 1.71x | 4.86 |

### Acceptance

| Prompt length | Window size | Drafts | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates |
|---:|---:|---:|---:|---:|---:|---|
| 100,000 | 8,192 | 106 | 424 | 409 | 96.5% | 0.99, 0.98, 0.94, 0.94 |
| 1,000,000 | 8,192 | 111 | 444 | 402 | 90.5% | 0.94, 0.90, 0.89, 0.89 |
| 4,000,000 | 8,192 | 107 | 428 | 408 | 95.3% | 0.98, 0.96, 0.94, 0.93 |
| 100,000 | 32,768 | 106 | 424 | 409 | 96.5% | 0.99, 0.98, 0.94, 0.94 |
| 1,000,000 | 32,768 | 107 | 428 | 406 | 94.9% | 0.98, 0.94, 0.93, 0.93 |
| 4,000,000 | 32,768 | 106 | 424 | 409 | 96.5% | 0.99, 0.98, 0.95, 0.93 |
| 100,000 | 131,072 | skipped | - | - | - | - |
| 1,000,000 | 131,072 | 106 | 424 | 407 | 96.0% | 0.99, 0.95, 0.95, 0.94 |
| 4,000,000 | 131,072 | 106 | 424 | 409 | 96.5% | 0.98, 0.97, 0.96, 0.94 |

## Block-Aligned Paged Num Spec Tokens Sweep

This TP=4 eager run used the same opt-in block-aligned paged self-SWA path at `prompt_len=4M` and `window_size=8192`, sweeping larger speculative batch sizes. All completed cases exact-matched the greedy baseline.

Run settings:

| Setting | Value |
|---|---:|
| Prompt length | `4M` |
| Window size | `8192` |
| Output length | `512` |
| Num prompts | `1` |
| Num speculative tokens | `8`, `16`, `32` |
| Tensor parallel size | `4` |
| Max model length | `4000545` |
| HF overrides | `seq_length=4194304`, `max_position_embeddings=4194304` |
| Attention backend | `ROCM_AITER_FA` |
| Enforce eager | `true` |
| Self-SWA sink size | `16` |
| Env | `HIP_VISIBLE_DEVICES=1,2,3,4`, `VLLM_SELF_SWA_BLOCK_ALIGNED_PAGED_ATTN=1`, `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False` |

### Decode Speed

| Num speculative tokens | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
|---:|:---:|---:|---:|---:|---:|
| 8 | true | 21.08 | 30.20 | 1.43x | 8.16 |
| 16 | true | 21.08 | 28.43 | 1.35x | 13.18 |
| 32 | true | 21.08 | 25.22 | 1.20x | 21.46 |

### Timing

| Run | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
|---|---:|---:|---:|
| Baseline | 2447.58 | 24.24 | 2423.34 |
| Self-SWA, 8 spec tokens | 2479.91 | 16.92 | 2462.99 |
| Self-SWA, 16 spec tokens | 2512.58 | 17.97 | 2494.61 |
| Self-SWA, 32 spec tokens | 2591.09 | 20.26 | 2570.83 |

### Acceptance

| Num speculative tokens | Drafts | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates |
|---:|---:|---:|---:|---:|---|
| 8 | 63 | 504 | 451 | 89.5% | 0.97, 0.95, 0.92, 0.90, 0.87, 0.86, 0.84, 0.84 |
| 16 | 39 | 624 | 475 | 76.1% | 0.90, 0.87, 0.82, 0.79, 0.77, 0.74, 0.74, 0.74, 0.74, 0.74, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72 |
| 32 | 24 | 768 | 491 | 63.9% | 0.83, 0.79, 0.75, 0.75, 0.67 x4, 0.62 x18, 0.58 x4, 0.54 x2 |

## Sink Size Comparison

### Self-SWA Decode Throughput

| Prompt length | Window size | Sink 0 tok/s | Sink 4 tok/s | Sink 4 vs sink 0 |
|---:|---:|---:|---:|---:|
| 100,000 | 8,192 | 64.07 | 39.19 | 0.61x |
| 1,000,000 | 8,192 | 31.03 | 33.22 | 1.07x |
| 4,000,000 | 8,192 | 7.50 | 21.16 | 2.82x |
| 100,000 | 32,768 | 80.67 | 17.97 | 0.22x |
| 1,000,000 | 32,768 | 36.43 | 16.70 | 0.46x |
| 4,000,000 | 32,768 | 7.62 | 13.02 | 1.71x |
| 1,000,000 | 131,072 | 36.72 | 5.50 | 0.15x |
| 4,000,000 | 131,072 | 8.17 | 4.92 | 0.60x |

### Acceptance Rate

| Prompt length | Window size | Sink 0 accept rate | Sink 4 accept rate | Delta |
|---:|---:|---:|---:|---:|
| 100,000 | 8,192 | 90.3% | 93.1% | +2.8 pp |
| 1,000,000 | 8,192 | 45.0% | 93.5% | +48.5 pp |
| 4,000,000 | 8,192 | 29.1% | 89.1% | +60.0 pp |
| 100,000 | 32,768 | 90.5% | 94.2% | +3.7 pp |
| 1,000,000 | 32,768 | 47.6% | 94.9% | +47.3 pp |
| 4,000,000 | 32,768 | 30.1% | 92.0% | +61.9 pp |
| 1,000,000 | 131,072 | 48.7% | 96.5% | +47.8 pp |
| 4,000,000 | 131,072 | 34.5% | 92.0% | +57.5 pp |

### Block-Aligned Paged vs Sink Size 4 Decode Throughput

This is a cross-run comparison: the sink sweep used TP=8, while the block-aligned paged run used TP=4 and different draft attention semantics.

| Prompt length | Window size | Sink 4 TP=8 tok/s | Block-aligned TP=4 tok/s | Block-aligned vs sink 4 |
|---:|---:|---:|---:|---:|
| 100,000 | 8,192 | 39.19 | 49.47 | 1.26x |
| 1,000,000 | 8,192 | 33.22 | 43.35 | 1.30x |
| 4,000,000 | 8,192 | 21.16 | 25.97 | 1.23x |
| 100,000 | 32,768 | 17.97 | 53.05 | 2.95x |
| 1,000,000 | 32,768 | 16.70 | 44.98 | 2.69x |
| 4,000,000 | 32,768 | 13.02 | 26.33 | 2.02x |
| 1,000,000 | 131,072 | 5.50 | 44.94 | 8.17x |
| 4,000,000 | 131,072 | 4.92 | 26.56 | 5.40x |

## Estimated TTFT

Estimated TTFT is computed as:

```text
elapsed_s_including_prefill - decode_elapsed_s
```

These estimates include frontend/engine overhead. For first self-SWA cases in a window, they can also include one-time AITER kernel compilation or cache-loading overhead, so treat them as upper-bound wall-clock TTFT estimates rather than isolated prefill kernel time.

The `sink=0`, `window=8192` TTFT estimates are compile-contaminated outliers from the first self-SWA window run and should not be compared directly with warm runs.

### Baseline Estimated TTFT

| Sink sweep | Prompt length | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
|---|---:|---:|---:|---:|
| sink=0 sweep | 100,000 | 6.36 | 4.21 | 2.15 |
| sink=0 sweep | 1,000,000 | 90.36 | 8.82 | 81.54 |
| sink=0 sweep | 4,000,000 | 1172.66 | 23.75 | 1148.91 |
| sink=4 sweep | 100,000 | 6.34 | 4.22 | 2.12 |
| sink=4 sweep | 1,000,000 | 90.40 | 8.86 | 81.54 |
| sink=4 sweep | 4,000,000 | 1185.66 | 24.48 | 1161.18 |

### Sink Size 0 Estimated TTFT

| Prompt length | Window size | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
|---:|---:|---:|---:|---:|
| 100,000 | 8,192 | 63.10 | 7.98 | 55.12* |
| 1,000,000 | 8,192 | 242.03 | 16.47 | 225.56* |
| 4,000,000 | 8,192 | 1884.07 | 68.13 | 1815.94* |
| 100,000 | 32,768 | 8.76 | 6.33 | 2.43 |
| 1,000,000 | 32,768 | 98.54 | 14.03 | 84.51 |
| 4,000,000 | 32,768 | 1238.16 | 67.09 | 1171.07 |
| 100,000 | 131,072 | skipped | skipped | skipped |
| 1,000,000 | 131,072 | 101.88 | 13.92 | 87.96 |
| 4,000,000 | 131,072 | 1233.82 | 62.54 | 1171.28 |

\* Compile-contaminated estimate from the first sink=0 self-SWA window run.

### Sink Size 4 Estimated TTFT

| Prompt length | Window size | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
|---:|---:|---:|---:|---:|
| 100,000 | 8,192 | 16.02 | 13.04 | 2.98 |
| 1,000,000 | 8,192 | 110.80 | 15.38 | 95.42 |
| 4,000,000 | 8,192 | 1193.46 | 24.15 | 1169.31 |
| 100,000 | 32,768 | 31.97 | 28.43 | 3.54 |
| 1,000,000 | 32,768 | 126.09 | 30.60 | 95.49 |
| 4,000,000 | 32,768 | 1233.38 | 39.25 | 1194.13 |
| 100,000 | 131,072 | skipped | skipped | skipped |
| 1,000,000 | 131,072 | 223.44 | 92.99 | 130.45 |
| 4,000,000 | 131,072 | 1402.03 | 103.85 | 1298.18 |

### Block-Aligned Paged Estimated TTFT

These estimates come from the TP=4 block-aligned paged run, so they are not directly comparable with the TP=8 sink sweep estimates above.

| Prompt length | Window size | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
|---:|---:|---:|---:|---:|
| 100,000 | 8,192 | 14.17 | 10.33 | 3.84 |
| 1,000,000 | 8,192 | 176.23 | 11.79 | 164.44 |
| 4,000,000 | 8,192 | 2296.88 | 19.67 | 2277.21 |
| 100,000 | 32,768 | 13.28 | 9.63 | 3.65 |
| 1,000,000 | 32,768 | 175.87 | 11.36 | 164.51 |
| 4,000,000 | 32,768 | 2295.07 | 19.41 | 2275.66 |
| 100,000 | 131,072 | skipped | skipped | skipped |
| 1,000,000 | 131,072 | 192.80 | 11.37 | 181.43 |
| 4,000,000 | 131,072 | 2292.46 | 19.24 | 2273.22 |

## Observations

| Observation | Details |
|---|---|
| Sink size 4 greatly improves acceptance. | Long-context accept rates rose from roughly `29-49%` with sink `0` to roughly `89-97%` with sink `4`. |
| Higher acceptance did not always improve speed. | Sink `4` was fastest relative to sink `0` for the `4M, 8192` and `4M, 32768` cases, but was slower for short contexts and the `131072` window. |
| Block-aligned paged self-SWA improved long-context decode speed. | In the TP=4 `num_spec_tokens=4` run it reached `1.63-1.71x` over the TP=4 baseline for completed `1M` and `4M` cases. |
| `num_spec_tokens=8` was fastest in the 4M/8k larger-batch sweep. | At TP=4 eager with block-aligned paged self-SWA, `num_spec_tokens=8` reached `30.20` tok/s (`1.43x`) versus `28.43` tok/s for `16` and `25.22` tok/s for `32`. Larger batches raised mean accept length from `8.16` to `21.46`, but lowered throughput because draft work increased and later-position acceptance fell. |
| Block-aligned paged results are a separate path. | The run used `self_swa_sink_size=16` with block-boundary rounding, so compare it as the paged path rather than as token-exact `sink_size=4`. |
| First-window TTFT estimates can be misleading. | The `sink=0`, `window=8192` estimated TTFT rows include cold AITER/JIT overhead and are outliers. |
| Best 4M result in the original sink sweep | `sink_size=4`, `window_size=8192`: `21.16` self-SWA decode tok/s, `1.01x` vs baseline, `89.1%` accept rate. |
| Best 4M block-aligned throughput | `num_spec_tokens=8`, `window_size=8192`: `30.20` self-SWA decode tok/s, `1.43x` vs same-run TP=4 baseline, `89.5%` accept rate. |
| All completed cases matched baseline exactly. | Every non-skipped self-SWA case reported exact token match `PASS`. |
