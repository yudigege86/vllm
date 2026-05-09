# Self-SWA Sink Size Sweep Results

Model: `zai-org/glm-4-9b-chat-1m`

Run settings:

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

## Observations

| Observation | Details |
|---|---|
| Sink size 4 greatly improves acceptance. | Long-context accept rates rose from roughly `29-49%` with sink `0` to roughly `89-97%` with sink `4`. |
| Higher acceptance did not always improve speed. | Sink `4` was fastest relative to sink `0` for the `4M, 8192` and `4M, 32768` cases, but was slower for short contexts and the `131072` window. |
| Best 4M result among these runs | `sink_size=4`, `window_size=8192`: `21.16` self-SWA decode tok/s, `1.01x` vs baseline, `89.1%` accept rate. |
| All completed cases matched baseline exactly. | Every non-skipped self-SWA case reported exact token match `PASS`. |
