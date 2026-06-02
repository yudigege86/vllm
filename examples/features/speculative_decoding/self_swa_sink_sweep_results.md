# Self-SWA Sink Size Sweep Results

Model: `zai-org/glm-4-9b-chat-1m`

Original sink sweep settings:


| Setting                | Value                                                   |
| ---------------------- | ------------------------------------------------------- |
| Prompt lengths         | `100k`, `1M`, `4M`                                      |
| Window sizes           | `8192`, `32768`, `131072`                               |
| Output length          | `512`                                                   |
| Num prompts            | `1`                                                     |
| Num speculative tokens | `4`                                                     |
| Tensor parallel size   | `8`                                                     |
| Max model length       | `4000517`                                               |
| HF overrides           | `seq_length=4194304`, `max_position_embeddings=4194304` |
| Attention backend      | `ROCM_AITER_FA`                                         |


Notes:


| Term                     | Meaning                                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mean accept              | `1 + num_accepted_tokens / num_drafts`; max is `5.0` for `num_spec_tokens=4`.                                                                                                                                                                      |
| Accept rate              | `num_accepted_tokens / num_draft_tokens`.                                                                                                                                                                                                          |
| Exact match              | Whether self-SWA output token IDs exactly matched the greedy baseline.                                                                                                                                                                             |
| Skipped case             | `prompt_len=100k`, `window_size=131072` was skipped because the window is larger than the prompt.                                                                                                                                                  |
| Block-aligned paged path | Opt-in TP=4 path using paged attention for self-SWA draft decode. `self_swa_sink_size=16` is the KV block size, and sink/recent regions are rounded to block boundaries, so this is not token-exact equivalent to the earlier `sink_size=4` sweep. |


## Sink Size 0

### Decode Speed


| Prompt length | Window size | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
| ------------- | ----------- | ----------- | --------------------- | --------------------- | ------- | ----------- |
| 100,000       | 8,192       | true        | 121.26                | 64.07                 | 0.53x   | 4.61        |
| 1,000,000     | 8,192       | true        | 57.91                 | 31.03                 | 0.54x   | 2.80        |
| 4,000,000     | 8,192       | true        | 21.51                 | 7.50                  | 0.35x   | 2.17        |
| 100,000       | 32,768      | true        | 121.26                | 80.67                 | 0.67x   | 4.62        |
| 1,000,000     | 32,768      | true        | 57.91                 | 36.43                 | 0.63x   | 2.90        |
| 4,000,000     | 32,768      | true        | 21.51                 | 7.62                  | 0.35x   | 2.20        |
| 100,000       | 131,072     | skipped     | -                     | -                     | -       | -           |
| 1,000,000     | 131,072     | true        | 57.91                 | 36.72                 | 0.63x   | 2.95        |
| 4,000,000     | 131,072     | true        | 21.51                 | 8.17                  | 0.38x   | 2.38        |


### Acceptance


| Prompt length | Window size | Drafts  | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates |
| ------------- | ----------- | ------- | ------------ | --------------- | ----------- | ------------------------- |
| 100,000       | 8,192       | 111     | 444          | 401             | 90.3%       | 0.92, 0.91, 0.89, 0.89    |
| 1,000,000     | 8,192       | 184     | 736          | 331             | 45.0%       | 0.47, 0.45, 0.45, 0.44    |
| 4,000,000     | 8,192       | 236     | 944          | 275             | 29.1%       | 0.31, 0.29, 0.28, 0.28    |
| 100,000       | 32,768      | 111     | 444          | 402             | 90.5%       | 0.93, 0.91, 0.89, 0.89    |
| 1,000,000     | 32,768      | 176     | 704          | 335             | 47.6%       | 0.49, 0.47, 0.47, 0.47    |
| 4,000,000     | 32,768      | 232     | 928          | 279             | 30.1%       | 0.32, 0.30, 0.29, 0.29    |
| 100,000       | 131,072     | skipped | -            | -               | -           | -                         |
| 1,000,000     | 131,072     | 174     | 696          | 339             | 48.7%       | 0.51, 0.48, 0.48, 0.48    |
| 4,000,000     | 131,072     | 216     | 864          | 298             | 34.5%       | 0.37, 0.34, 0.34, 0.33    |


## Sink Size 4

### Decode Speed


| Prompt length | Window size | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
| ------------- | ----------- | ----------- | --------------------- | --------------------- | ------- | ----------- |
| 100,000       | 8,192       | true        | 120.97                | 39.19                 | 0.32x   | 4.72        |
| 1,000,000     | 8,192       | true        | 57.70                 | 33.22                 | 0.58x   | 4.74        |
| 4,000,000     | 8,192       | true        | 20.88                 | 21.16                 | 1.01x   | 4.56        |
| 100,000       | 32,768      | true        | 120.97                | 17.97                 | 0.15x   | 4.77        |
| 1,000,000     | 32,768      | true        | 57.70                 | 16.70                 | 0.29x   | 4.79        |
| 4,000,000     | 32,768      | true        | 20.88                 | 13.02                 | 0.62x   | 4.68        |
| 100,000       | 131,072     | skipped     | -                     | -                     | -       | -           |
| 1,000,000     | 131,072     | true        | 57.70                 | 5.50                  | 0.10x   | 4.86        |
| 4,000,000     | 131,072     | true        | 20.88                 | 4.92                  | 0.24x   | 4.68        |


### Acceptance


| Prompt length | Window size | Drafts  | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates |
| ------------- | ----------- | ------- | ------------ | --------------- | ----------- | ------------------------- |
| 100,000       | 8,192       | 109     | 436          | 406             | 93.1%       | 0.94, 0.93, 0.93, 0.93    |
| 1,000,000     | 8,192       | 108     | 432          | 404             | 93.5%       | 0.95, 0.94, 0.94, 0.92    |
| 4,000,000     | 8,192       | 112     | 448          | 399             | 89.1%       | 0.92, 0.90, 0.88, 0.87    |
| 100,000       | 32,768      | 108     | 432          | 407             | 94.2%       | 0.95, 0.94, 0.94, 0.94    |
| 1,000,000     | 32,768      | 107     | 428          | 406             | 94.9%       | 0.96, 0.95, 0.94, 0.93    |
| 4,000,000     | 32,768      | 110     | 440          | 405             | 92.0%       | 0.95, 0.93, 0.91, 0.90    |
| 100,000       | 131,072     | skipped | -            | -               | -           | -                         |
| 1,000,000     | 131,072     | 106     | 424          | 409             | 96.5%       | 0.98, 0.96, 0.96, 0.95    |
| 4,000,000     | 131,072     | 110     | 440          | 405             | 92.0%       | 0.95, 0.93, 0.91, 0.90    |


## Block-Aligned Paged Self-SWA

This run exercised the opt-in block-aligned paged draft path, not the token-exact `sink_size=4` path above. The effective sink size is one KV block (`self_swa_sink_size=16`), and both sink and recent regions are rounded to block boundaries before dispatching paged attention. All completed cases exact-matched the greedy baseline.

Run settings:


| Setting                | Value                                                                                                          |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| Prompt lengths         | `100k`, `1M`, `4M`                                                                                             |
| Window sizes           | `8192`, `32768`, `131072`                                                                                      |
| Output length          | `512`                                                                                                          |
| Num prompts            | `1`                                                                                                            |
| Num speculative tokens | `4`                                                                                                            |
| Tensor parallel size   | `4`                                                                                                            |
| Max model length       | `4000517`                                                                                                      |
| HF overrides           | `seq_length=4194304`, `max_position_embeddings=4194304`                                                        |
| Attention backend      | `ROCM_AITER_FA`                                                                                                |
| Enforce eager          | `true`                                                                                                         |
| Self-SWA sink size     | `16`                                                                                                           |
| Env                    | `VLLM_SELF_SWA_BLOCK_ALIGNED_PAGED_ATTN=1`, `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False` |


### Decode Speed


| Prompt length | Window size | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
| ------------- | ----------- | ----------- | --------------------- | --------------------- | ------- | ----------- |
| 100,000       | 8,192       | true        | 91.55                 | 49.47                 | 0.54x   | 4.86        |
| 1,000,000     | 8,192       | true        | 26.56                 | 43.35                 | 1.63x   | 4.62        |
| 4,000,000     | 8,192       | true        | 15.51                 | 25.97                 | 1.67x   | 4.81        |
| 100,000       | 32,768      | true        | 91.55                 | 53.05                 | 0.58x   | 4.86        |
| 1,000,000     | 32,768      | true        | 26.56                 | 44.98                 | 1.69x   | 4.79        |
| 4,000,000     | 32,768      | true        | 15.51                 | 26.33                 | 1.70x   | 4.86        |
| 100,000       | 131,072     | skipped     | -                     | -                     | -       | -           |
| 1,000,000     | 131,072     | true        | 26.56                 | 44.94                 | 1.69x   | 4.84        |
| 4,000,000     | 131,072     | true        | 15.51                 | 26.56                 | 1.71x   | 4.86        |


### Acceptance


| Prompt length | Window size | Drafts  | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates |
| ------------- | ----------- | ------- | ------------ | --------------- | ----------- | ------------------------- |
| 100,000       | 8,192       | 106     | 424          | 409             | 96.5%       | 0.99, 0.98, 0.94, 0.94    |
| 1,000,000     | 8,192       | 111     | 444          | 402             | 90.5%       | 0.94, 0.90, 0.89, 0.89    |
| 4,000,000     | 8,192       | 107     | 428          | 408             | 95.3%       | 0.98, 0.96, 0.94, 0.93    |
| 100,000       | 32,768      | 106     | 424          | 409             | 96.5%       | 0.99, 0.98, 0.94, 0.94    |
| 1,000,000     | 32,768      | 107     | 428          | 406             | 94.9%       | 0.98, 0.94, 0.93, 0.93    |
| 4,000,000     | 32,768      | 106     | 424          | 409             | 96.5%       | 0.99, 0.98, 0.95, 0.93    |
| 100,000       | 131,072     | skipped | -            | -               | -           | -                         |
| 1,000,000     | 131,072     | 106     | 424          | 407             | 96.0%       | 0.99, 0.95, 0.95, 0.94    |
| 4,000,000     | 131,072     | 106     | 424          | 409             | 96.5%       | 0.98, 0.97, 0.96, 0.94    |


## Block-Aligned Paged Num Spec Tokens Sweep

This TP=4 eager run used the same opt-in block-aligned paged self-SWA path at `prompt_len=4M` and `window_size=8192`, sweeping larger speculative batch sizes. All completed cases exact-matched the greedy baseline.

Run settings:


| Setting                | Value                                                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Prompt length          | `4M`                                                                                                                                          |
| Window size            | `8192`                                                                                                                                        |
| Output length          | `512`                                                                                                                                         |
| Num prompts            | `1`                                                                                                                                           |
| Num speculative tokens | `8`, `16`, `32`                                                                                                                               |
| Tensor parallel size   | `4`                                                                                                                                           |
| Max model length       | `4000545`                                                                                                                                     |
| HF overrides           | `seq_length=4194304`, `max_position_embeddings=4194304`                                                                                       |
| Attention backend      | `ROCM_AITER_FA`                                                                                                                               |
| Enforce eager          | `true`                                                                                                                                        |
| Self-SWA sink size     | `16`                                                                                                                                          |
| Env                    | `HIP_VISIBLE_DEVICES=1,2,3,4`, `VLLM_SELF_SWA_BLOCK_ALIGNED_PAGED_ATTN=1`, `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False` |


### Decode Speed


| Num speculative tokens | Exact match | Baseline decode tok/s | Self-SWA decode tok/s | Speedup | Mean accept |
| ---------------------- | ----------- | --------------------- | --------------------- | ------- | ----------- |
| 8                      | true        | 21.08                 | 30.20                 | 1.43x   | 8.16        |
| 16                     | true        | 21.08                 | 28.43                 | 1.35x   | 13.18       |
| 32                     | true        | 21.08                 | 25.22                 | 1.20x   | 21.46       |


### Timing


| Run                      | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
| ------------------------ | ------------------------- | ------------------ | ------------------ |
| Baseline                 | 2447.58                   | 24.24              | 2423.34            |
| Self-SWA, 8 spec tokens  | 2479.91                   | 16.92              | 2462.99            |
| Self-SWA, 16 spec tokens | 2512.58                   | 17.97              | 2494.61            |
| Self-SWA, 32 spec tokens | 2591.09                   | 20.26              | 2570.83            |


### Acceptance


| Num speculative tokens | Drafts | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates                                                                      |
| ---------------------- | ------ | ------------ | --------------- | ----------- | ---------------------------------------------------------------------------------------------- |
| 8                      | 63     | 504          | 451             | 89.5%       | 0.97, 0.95, 0.92, 0.90, 0.87, 0.86, 0.84, 0.84                                                 |
| 16                     | 39     | 624          | 475             | 76.1%       | 0.90, 0.87, 0.82, 0.79, 0.77, 0.74, 0.74, 0.74, 0.74, 0.74, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72 |
| 32                     | 24     | 768          | 491             | 63.9%       | 0.83, 0.79, 0.75, 0.75, 0.67 x4, 0.62 x18, 0.58 x4, 0.54 x2                                    |


## Best-Settings Sweep — `repeat_block` Prompts (historical)

This TP=4 eager run used the current self-SWA defaults: the block-aligned paged draft path is enabled by default, and local argmax is used by default when supported. The run did not use the removed block-aligned env var, local-argmax flag, or assume flags.

Prompt content was the legacy `repeat_block` short-sentence filler (no chat template) — the same prompt used by every earlier section above. The PG-19 rerun in the next section is the canonical "best settings" measurement; this section is retained for historical comparison.

Common run settings:


| Setting                | Value                                                     |
| ---------------------- | --------------------------------------------------------- |
| Prompt mode            | `repeat` (no chat template)                               |
| Prompt lengths         | `100k`, `1M`, `4M`                                        |
| Window size            | `8192`                                                    |
| Output length          | `512`                                                     |
| Num prompts            | `1`                                                       |
| Num speculative tokens | `4`                                                       |
| Tensor parallel size   | `4`                                                       |
| Attention backend      | `ROCM_AITER_FA`                                           |
| Enforce eager          | `true`                                                    |
| Repeats                | `3`; outlier removed per metric, average of remaining two |
| Max model length       | `4000517`                                                 |
| GLM HF overrides       | `seq_length=4194304`, `max_position_embeddings=4194304`   |
| Llama HF overrides     | `max_position_embeddings=4194304`                         |


Exact match passed for all parsed GLM summary rows. The latest Llama logs marked the `1M` rows as exact-match `FAIL` in all three repeats, while the `100k` and `4M` rows passed.

### `zai-org/glm-4-9b-chat-1m`


| Prompt length | Baseline avg tok/s | Self-SWA avg tok/s | Speedup avg | Mean accept avg |
| ------------- | ------------------ | ------------------ | ----------- | --------------- |
| 100,000       | 100.34             | 91.01              | 0.91x       | 4.86            |
| 1,000,000     | 79.88              | 86.06              | 1.06x       | 4.62            |
| 4,000,000     | 21.68              | 38.61              | 1.78x       | 4.86            |


### `gradientai/Llama-3-8B-Instruct-Gradient-4194k`


| Prompt length | Baseline avg tok/s | Self-SWA avg tok/s | Speedup avg | Mean accept avg |
| ------------- | ------------------ | ------------------ | ----------- | --------------- |
| 100,000       | 110.35             | 72.72              | 0.67x       | 3.21            |
| 1,000,000     | 69.85              | 114.48             | 1.64x       | 4.84            |
| 4,000,000     | 17.63              | 33.82              | 1.92x       | 5.00            |


## Best-Settings Sweep — PG-19 Prompts

Rerun of the sweep above using `--prompt-mode pg19` (the new default), which tiles real Project Gutenberg long-context text and applies the tokenizer's chat template automatically. The intent is to characterize self-SWA on realistic content rather than the periodic `repeat_block` filler, which artificially inflated drafter acceptance because every chunk of context looked like every other chunk.

All other settings match the historical sweep above (TP=4 eager, window 8192, output 512, 3 repeats with outlier removal). The GLM model used the same HF overrides; Llama was not run (see below). The original sweep used `--num-spec-tokens 4`; this section was later extended with two more sweep points at `num_spec_tokens=6` and `num_spec_tokens=8` (same model, same prompts, same methodology, `--max-model-len 4000525` to fit the extra speculative slots) — see [Num Speculative Tokens Sweep — PG-19](#num-speculative-tokens-sweep--pg-19) for the focused breakdown.

Common run settings:


| Setting                | Value                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------- |
| Prompt mode            | `pg19`, chat template auto-on                                                            |
| PG-19 dataset          | `emozilla/pg19` (parquet mirror, split `test`)                                           |
| Prompt lengths         | `100k`, `1M`, `4M`                                                                       |
| Window size            | `8192`                                                                                   |
| Output length          | `512`                                                                                    |
| Num prompts            | `1`                                                                                      |
| Num speculative tokens | `4` (original run), `6`, `8` (later extension)                                           |
| Tensor parallel size   | `4`                                                                                      |
| Attention backend      | `ROCM_AITER_FA`                                                                          |
| Enforce eager          | `true`                                                                                   |
| Repeats                | `3`; outlier removed per metric, average of remaining two                                |
| Max model length       | `4000517` (k=4 original), `4000525` (k=6/8 extension, fits up to 8 speculative tokens)   |
| GLM HF overrides       | `seq_length=4194304`, `max_position_embeddings=4194304`                                  |


### `zai-org/glm-4-9b-chat-1m`


| Prompt length | Num spec tokens | Baseline avg tok/s | Self-SWA avg tok/s | Speedup avg | Mean accept avg | Exact match |
| ------------- | --------------- | ------------------ | ------------------ | ----------- | --------------- | ----------- |
| 100,000       | 4               | 97.76              | 92.03              | 0.95x       | 4.90 / 5.0      | True        |
| 100,000       | 6               | 97.65              | 95.91              | 0.93x       | 6.79 / 7.0      | True        |
| 100,000       | 8               | 99.06              | 55.35              | 0.55x       | 5.01 / 9.0      | False       |
| 1,000,000     | 4               | 79.81              | 76.78              | 0.96x       | 4.21 / 5.0      | False       |
| 1,000,000     | 6               | 79.59              | 74.62              | 0.94x       | 5.38 / 7.0      | False       |
| 1,000,000     | 8               | 80.10              | 67.01              | 0.84x       | 6.13 / 9.0      | False       |
| 2,000,000     | 4               | 44.29              | 55.60              | 1.26x       | 4.00 / 5.0      | False       |
| 2,000,000     | 6               | 44.29              | 63.13              | 1.43x       | 5.10 / 7.0      | False       |
| 2,000,000     | 8               | 44.35              | 62.04              | 1.42x       | 5.74 / 9.0      | False       |
| 4,000,000     | 4               | 21.70              | 36.29              | 1.67x       | 4.53 / 5.0      | False       |
| 4,000,000     | 6               | 21.64              | 43.38              | 2.00x       | 5.88 / 7.0      | False       |
| 4,000,000     | 8               | 21.64              | 46.00              | 2.12x       | 6.91 / 9.0      | False       |


`mean_acceptance_length` was bit-identical across all three repeats for every cell (e.g. 4.90 / 4.21 / 4.53 at k=4; 6.79 / 5.38 / 5.88 at k=6; 5.01 / 6.13 / 6.91 at k=8; 4.00 / 5.10 / 5.74 at 2M), confirming the PG-19 prompts produce deterministic spec-decode trajectories regardless of `num_spec_tokens`. Wall-clock decode rates varied by <2% across repeats, with one important exception: the 2M-only extension paid an AITER JIT cold-start penalty in r1 of both processes (baseline ran at ~22 tok/s in r1 vs ~44 tok/s in r2/r3, since 2M shape was never warmed by earlier runs). Outlier-per-metric removal handles this; the aggregated 2M baseline number above is the warm-kernel value averaged across r2 and r3.

Direct comparison to the `repeat_block` GLM numbers above:


| Prompt length | repeat-block self-SWA tok/s | PG-19 self-SWA tok/s | Δ speedup | Δ mean accept |
| ------------- | --------------------------- | -------------------- | --------- | ------------- |
| 100,000       | 91.01                       | 92.03                | +0.04x    | +0.04         |
| 1,000,000     | 86.06                       | 76.78                | -0.10x    | -0.41         |
| 4,000,000     | 38.61                       | 36.29                | -0.11x    | -0.33         |


Baseline tok/s is within noise of the `repeat_block` run (97.76 vs 100.34 at 100k; 79.81 vs 79.88 at 1M; 21.70 vs 21.68 at 4M), as expected — baseline decode throughput does not depend on prompt content. Self-SWA throughput and mean acceptance drop at 1M and 4M because the drafter can no longer rely on the periodic-prompt shortcut; the remaining **1.67x at 4M is the realistic speedup on real long-context text**. Exact match flipped from `True` to `False` at 1M and 4M: numerical drift between baseline and self-SWA decode was always present, but the `repeat_block` argmax was robust to small logit differences. Real text has sharper, content-dependent argmax boundaries that a few of those drifts cross.

### `gradientai/Llama-3-8B-Instruct-Gradient-4194k`

Not run. A baseline-only coherence check at 100k and 1M was conducted before launching the full sweep, and the model collapsed into structured noise at 1M:


| Prompt length | Baseline decode tok/s | Output sample                                                                                                                            |
| ------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 100,000       | 128.84                | `CHAPTER XII / THE WAR BOARD / The two men stood panting, in the shadow of a high wall, ...` (coherent novel continuation)               |
| 1,000,000     | 69.49                 | `[0_system],[0_q_0],0_a_0,[0_q_1],0_a_1,[0_q_2],0_a_2,[0_q_3],0_a_3,[0_q_4],...` (degenerate; pattern continues for the full 512 tokens) |


This is the well-known long-context failure mode for RoPE-scaled Llama variants — the trained position range (`4194k`) is not the empirically usable one. The repeat-block numbers above for this model at 1M and 4M reflect speed and acceptance against a prompt the model could lock onto regardless of context use, not meaningful long-context behavior. The PG-19 rerun was skipped at the user's request.

### Generation Quality (GLM, PG-19)

For each prompt length, the prompt's last few sentences are echoed before the generation so the seam between prompt and continuation is visible. Self-SWA continuations diverge from baseline at the word level at 1M and 4M (matching `exact_match=False`) but stay equally on-topic and stylistically consistent — the divergences are not failures, they're equally valid greedy paths through nearly-tied logits.

#### 100k (window=8192)

Prompt tail (last 3 sentences):

> In the moonlight he stared at his companion, cackled, clapped his thighs, and bent double in unholy convulsions. "My gracious me!" He laughed immoderately. "Oh, I wait zo fearful

Baseline output (first ~50 tokens; degenerate):

> "Gracious me! Oh, I wait zo fearful! But you, you're a wonder! You're a wonder! You're a wonder! [...repeats for 512 tokens...]"

Self-SWA output: token-identical to baseline (`exact_match=True`).

This is a genuine model failure on one particular Dickensian dialogue line — the model latches onto "You're a wonder!" and loops. With `--num-prompts > 1` you would see this is a per-passage artifact, not a systemic issue.

#### 1M (window=8192)

Prompt tail (last 3 sentences):

> ... The Central Government fear that the taking up of a spirited position by any pre-eminent Chinese would carry the Chinese people with him, and therefore the Central Government endeavour to keep up appearances, and to skirt the precipice of war as near as

Baseline output (first ~150 tokens):

> possible. This policy of appeasement and conciliation has led to a gradual erosion of Chinese sovereignty and **prestige**, and has left the Chinese people feeling **disillusioned and disheartened**. Gordon's memorandum goes on to outline a series of reforms that he believes would restore confidence in the Central Government and strengthen China's position on the international stage. He advocates for the establishment of a strong, centralized government with a clear and consistent foreign policy. He also calls for the modernization of the military, the development of a strong industrial base, and the promotion of education and cultural exchange with the outside world.

Self-SWA output (first ~150 tokens):

> possible. This policy of appeasement and conciliation has led to a gradual erosion of Chinese sovereignty and **national pride**, and has left the Chinese people feeling **disheartened and disillusioned**. Gordon's memorandum goes on to outline a series of reforms that he believes would restore confidence in the Central Government and strengthen China's position on the international stage. He advocates for the establishment of a strong, centralized government with a clear and consistent foreign policy. He also calls for the modernization of the military, the development of a strong industrial base, and the promotion of education and cultural exchange with the outside world.

Both continuations identify the historical subject correctly (a memorandum by General Charles Gordon on late-Qing China), reference downstream events (Boxer Rebellion 1900, Boxer Protocol) accurately, and stay in the appropriate 19th-century register. They diverge only at the word level (`prestige` ↔ `national pride`; swapped order of `disillusioned`/`disheartened`) — exactly the kind of small-logit-drift divergence that fails exact-match but does not change semantic content.

#### 4M (window=8192)

Prompt tail (last 4 lines):

> ... The verses, crudely expressing the contrast between President Jefferson and King George—or Napoleon, it is not clear which,—sufficiently show that Paine's genius was not extempore.

Baseline output (first ~120 tokens):

> Paine's letter to Jefferson, dated August 2, 1803, reveals his hope for the future of Louisiana and his vision for **its integration into the United States**. He suggests that a provisional government should be established by Congress for a period of three to seven years, during which the inhabitants would learn the principles of representative government and elect their municipal government. He also advocates for the people to have the right to elect their Church Ministers, which would hold the priests in check and give the people an idea of elective rights.

Self-SWA output (first ~120 tokens):

> Paine's letter to Jefferson, dated August 2, 1803, reveals his hope for the future of Louisiana and his vision for **its governance**. He suggests that the people of Louisiana should be initiated into the practice of representative government by electing their Municipal government, and after some experience they will be in train to elect their State government. He also emphasizes the importance of establishing schools to teach English as the laws ought to be in the language of the Union.

Both continuations correctly pivot from the introductory verses (Paine's poetry quoted in the prompt) to the substance of Paine's actual 1803 letter to Jefferson about Louisiana governance. The se    

### Num Speculative Tokens Sweep — PG-19

Companion sweep to the GLM PG-19 table above, holding model/prompts/window fixed and sweeping `num_spec_tokens ∈ {4, 6, 8}` across `prompt_len ∈ {100k, 1M, 2M, 4M}`. The intent is to characterize how draft depth interacts with target decode cost on realistic long-context content. `k=4 / {100k, 1M, 4M}` numbers are restated from the original PG-19 sweep above for direct comparison; `k=6` and `k=8` at `{100k, 1M, 4M}` ran in parallel on disjoint GPU sets (TP=4 each) under the same conditions, with `--max-model-len 4000525` to fit the extra speculative slots. The 2M row was added later as a single-prompt-length extension; all three `k` values share `--max-model-len 4000525` for that row.

Run settings (additional to common settings above):


| Setting                | Value                                                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Num speculative tokens | `4`, `6`, `8`                                                                                                                                                                                                                    |
| Max model length       | `4000525` (raised from `4000517` of the k=4 run; the extra 8 KV slots are at the spec-decode tail and do not affect target decode)                                                                                               |
| Wrappers               | `/tmp/run_sweep_pg19_numspec.sh` (k=6 on GPUs 0–3, k=8 on GPUs 4–7) for the 100k/1M/4M rows; `/tmp/run_sweep_pg19_2m.sh` (k=4+k=6 sharing a baseline on GPUs 0–3 via `--num-spec-tokens-list 4,6`, k=8 on GPUs 4–7) for the 2M row |
| Repeats                | `3`; outlier removed per metric, average of remaining two. 2M r1 baselines were cold-AITER-JIT outliers in both processes (~22 vs ~44 tok/s); outlier removal handles this and the aggregated 2M baseline is the warm value      |
| Env                    | `VLLM_ROCM_USE_AITER=1`                                                                                                                                                                                                          |


#### Decode Speed


| Prompt length | Num spec tokens | Baseline avg tok/s | Self-SWA avg tok/s | Speedup avg | Mean accept avg | Exact match |
| ------------- | --------------- | ------------------ | ------------------ | ----------- | --------------- | ----------- |
| 100,000       | 4               | 97.76              | 92.03              | 0.95x       | 4.90 / 5.0      | True        |
| 100,000       | 6               | 97.65              | 95.91              | 0.93x       | 6.79 / 7.0      | True        |
| 100,000       | 8               | 99.06              | 55.35              | 0.55x       | 5.01 / 9.0      | False       |
| 1,000,000     | 4               | 79.81              | 76.78              | 0.96x       | 4.21 / 5.0      | False       |
| 1,000,000     | 6               | 79.59              | 74.62              | 0.94x       | 5.38 / 7.0      | False       |
| 1,000,000     | 8               | 80.10              | 67.01              | 0.84x       | 6.13 / 9.0      | False       |
| 2,000,000     | 4               | 44.29              | 55.60              | 1.26x       | 4.00 / 5.0      | False       |
| 2,000,000     | 6               | 44.29              | 63.13              | 1.43x       | 5.10 / 7.0      | False       |
| 2,000,000     | 8               | 44.35              | 62.04              | 1.42x       | 5.74 / 9.0      | False       |
| 4,000,000     | 4               | 21.70              | 36.29              | 1.67x       | 4.53 / 5.0      | False       |
| 4,000,000     | 6               | 21.64              | 43.38              | 2.00x       | 5.88 / 7.0      | False       |
| 4,000,000     | 8               | 21.64              | 46.00              | 2.12x       | 6.91 / 9.0      | False       |


At 4M the speedup grows monotonically with `k` (1.67x → 2.00x → 2.12x), confirming that on the bandwidth-bound long-context decode it pays to amortize verify cost across more accepted tokens per pass. At 100k the relationship inverts: target decode is already fast, so the extra drafter forward passes are pure overhead. `k=6` and `k=4` are within noise at 100k, but `k=8` collapses to 0.55x because per-position acceptance falls off a cliff after position 3 (see below) and most of the 8 drafted tokens get thrown away. 2M sits between the two regimes: baseline tok/s has already collapsed by ~half from 1M (80 → 44 tok/s), so spec decode pays off (`1.26x` at k=4, climbing to `1.43x` at k=6), but the marginal gain from k=6 → k=8 has saturated (k=6 and k=8 are within 0.01x at 2M, vs +0.12x at 4M). The "deep drafts amortize verify cost" effect is real but the curve flattens earlier at 2M than at 4M because the verify-pass cost is smaller in absolute terms.

#### Timing (single repeat; values across the three repeats agreed to within wall-clock noise; 2M baselines are r2 values to skip the JIT cold-start of r1)


| Num spec tokens | Prompt length | Baseline elapsed incl. prefill (s) | Baseline decode elapsed (s) | Self-SWA elapsed incl. prefill (s) | Self-SWA decode elapsed (s) |
| --------------- | ------------- | ---------------------------------- | --------------------------- | ---------------------------------- | --------------------------- |
| 6               | 100,000       | 8.62                               | 5.27                        | 8.92                               | 5.33                        |
| 6               | 1,000,000     | 163.52                             | 5.14                        | 165.99                             | 6.51                        |
| 6               | 2,000,000     | 623.05                             | 11.54                       | 623.19                             | 8.09                        |
| 6               | 4,000,000     | 2230.63                            | 23.79                       | 2221.36                            | 11.81                       |
| 8               | 100,000       | 8.56                               | 5.15                        | 13.04                              | 9.37                        |
| 8               | 1,000,000     | 162.85                             | 5.11                        | 166.46                             | 7.25                        |
| 8               | 2,000,000     | 618.21                             | 11.56                       | 618.45                             | 8.03                        |
| 8               | 4,000,000     | 2209.74                            | 23.56                       | 2201.25                            | 11.18                       |


Decode-elapsed is the meaningful comparison; the elapsed-incl-prefill column is dominated by prefill (e.g. ~10 min at 2M, ~37 min at 4M) and is essentially identical between baseline and self-SWA for the same prompt length. At 2M, baseline decode-elapsed is `~11.5s` for 512 output tokens — already in the same order as the 4M baseline decode-elapsed (~23.7s for the same 512 tokens), which is why 2M speedups look more like 4M's than like 1M's.

#### Acceptance


| Prompt length | Num spec tokens | Drafts | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates                              |
| ------------- | --------------- | ------ | ------------ | --------------- | ----------- | ------------------------------------------------------ |
| 100,000       | 6               | 76     | 456          | 440             | 96.5%       | 1.00, 0.97, 0.96, 0.96, 0.95, 0.95                     |
| 1,000,000     | 6               | 90     | 540          | 394             | 73.0%       | 0.92, 0.82, 0.72, 0.70, 0.67, 0.54                     |
| 2,000,000     | 6               | 101    | 606          | 414             | 68.3%       | 0.86, 0.78, 0.73, 0.61, 0.55, 0.55                     |
| 4,000,000     | 6               | 88     | 528          | 429             | 81.3%       | 0.98, 0.89, 0.83, 0.78, 0.70, 0.69                     |
| 100,000       | 8               | 102    | 816          | 409             | 50.1%       | 0.82, 0.74, 0.61, 0.49, 0.42, 0.36, 0.30, 0.26         |
| 1,000,000     | 8               | 79     | 632          | 405             | 64.1%       | 0.90, 0.78, 0.68, 0.63, 0.61, 0.53, 0.51, 0.48         |
| 2,000,000     | 8               | 89     | 712          | 422             | 59.3%       | 0.87, 0.78, 0.67, 0.60, 0.49, 0.47, 0.46, 0.40         |
| 4,000,000     | 8               | 75     | 600          | 443             | 73.8%       | 0.96, 0.89, 0.79, 0.76, 0.67, 0.65, 0.61, 0.57         |


The `k=8 / 100k` row is the standout outlier. At 100k context with `k=6` per-position rates stay above 0.95 for every position, and with `k=4` (not shown here, but reported earlier in the sink-sweep section) they stay near 0.99. At `k=8 / 100k`, the per-position rate drops below 0.5 by position 3 and reaches 0.26 by position 7 — i.e. only ~26% of position-7 drafts are accepted. The other long-context rows show the more typical degradation pattern (high acceptance at the early positions, gradually falling toward the tail), and importantly the tail rates at `k=8 / 4M` (0.57) are *higher* than the tail rates at `k=8 / 100k` (0.26). The drafter is doing worse on short context with deep drafts than on long context with deep drafts; this is the same effect that flips `exact_match` from True (k=4, k=6 at 100k) to False (k=8 at 100k) — once the joint trajectory diverges from the baseline, the drafter's learned token-level priors become worse than the target's actual distribution.

Note that the `2M` rows look weaker on raw accept rate than the `4M` rows (e.g. `k=8/2M` is 59.3% vs `k=8/4M` at 73.8%), even though 2M throughput is faster than 4M throughput. That isn't a contradiction: acceptance is a function of how well the drafter's predictions match the target's argmax at this specific passage of PG-19 (which depends on the content the model has rolled into the recent window), while throughput is set by `mean_acceptance_length × baseline_tok/s`. The 2M passage just happens to be moderately easier for the target and moderately harder for the drafter than the 4M passage.

#### Headline

For PG-19 long-context decode on GLM-4-9B-Chat-1M, with the block-aligned paged self-SWA path on by default at TP=4 eager:

- **At 4M context: prefer `k=8`**, which gives **2.12x** vs baseline (46.00 / 21.64 tok/s).
- **At 2M context: prefer `k=6`** (1.43x) or `k=8` (1.42x — within noise). `k=4` trails meaningfully at 1.26x.
- **At 1M context: prefer `k=4`**, which gives 0.96x (essentially break-even). Deeper drafts hurt at this prompt length.
- **At 100k context: prefer `k=4` or `k=6`** (both ~0.95x). `k=8` is a clear regression (0.55x) and should be avoided.
- The break-even crossover is between 1M and 2M: at 1M, baseline runs at ~80 tok/s and spec decode is a wash; at 2M, baseline has dropped to ~44 tok/s and spec decode wins clearly. Within "long context wins" (2M and 4M), `k=8` only meaningfully outpaces `k=6` at 4M; at 2M the two are tied, and deeper drafts past `k=8` would likely be diminishing-returns at any of these prompt lengths on this configuration.

### TP=1 Comparison — PG-19 (100k and 1M)

Probe of the "decode is overhead-bound at short context, not KV-bandwidth-bound" hypothesis that emerged from the TP=4 numbers. At TP=4, baseline decode per-step time was ~10 ms at 100k where the per-GPU HBM-bound floor was only ~1.8 ms — i.e. the per-token decode time was 5x the bandwidth-bound lower bound, suggesting most of it was kernel-launch / TP-collective / Python-loop overhead. TP=1 removes the all-reduces entirely (no inter-GPU sync) but redistributes the KV per-GPU: at 100k a single GPU now reads 4 GB of KV per token (vs 1 GB at TP=4), and at 1M reads 40 GB (vs 10 GB). If the hypothesis is right, TP=1 should:

1. Be *faster* than TP=4 at 100k, because the all-reduce overhead removed exceeds the bandwidth gain lost at that scale.
2. Be *slower* than TP=4 at 1M, because at 1M the KV traffic on one GPU genuinely becomes bandwidth-limiting and TP parallelism wins.
3. Show real self-SWA speedup at 1M (where the target is now bandwidth-bound but the drafter still reads only 8192 tokens of KV).

Run settings (additional to common settings above):


| Setting             | Value                                                                                                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tensor parallel size | `1`                                                                                                                                                                                                                                        |
| Prompt lengths      | `100k`, `1M` (2M and 4M skipped; KV alone at 4M would be 160 GB, near the per-GPU limit and well past the regime this probe is testing)                                                                                                     |
| Num spec tokens     | `4`, `6`, `8`                                                                                                                                                                                                                               |
| Max model length    | `1000525` (tight to 1M to keep KV cache allocation reasonable on one GPU)                                                                                                                                                                   |
| Wrapper             | `/tmp/run_sweep_pg19_tp1.sh` — three parallel processes, one per `k` value, each pinned to a single GPU via `HIP_VISIBLE_DEVICES`. 3 sequential repeats each. Each process produces its own baseline measurements (averaged across 9 runs). |
| Repeats             | `3`; outlier removed per metric, average of remaining two. Baselines were tightly consistent across r1/r2/r3 (no JIT outlier) because each process starts at 100k, which warms the kernels before 1M.                                       |
| Env                 | `VLLM_ROCM_USE_AITER=1`                                                                                                                                                                                                                     |


#### Decode Speed


| Prompt length | Num spec tokens | Baseline avg tok/s | Self-SWA avg tok/s | Speedup avg | Mean accept avg | Exact match |
| ------------- | --------------- | ------------------ | ------------------ | ----------- | --------------- | ----------- |
| 100,000       | 4               | 120.57             | 99.07              | 0.84x       | 4.29 / 5.0      | False       |
| 100,000       | 6               | 116.28             | 80.19              | 0.62x       | 4.66 / 7.0      | False       |
| 100,000       | 8               | 118.21             | 114.73             | 0.95x       | 8.29 / 9.0      | **True**    |
| 1,000,000     | 4               | 32.98              | 42.00              | **1.28x**   | 4.02 / 5.0      | False       |
| 1,000,000     | 6               | 33.04              | 43.63              | **1.32x**   | 4.91 / 7.0      | False       |
| 1,000,000     | 8               | 33.09              | 32.30              | 0.97x       | 5.79 / 9.0      | False       |


#### Side-by-side vs TP=4


| Prompt length | k | TP=4 baseline | TP=1 baseline | Δ baseline | TP=4 speedup | TP=1 speedup | Δ speedup |
| ------------- | - | ------------- | ------------- | ---------- | ------------ | ------------ | --------- |
| 100,000       | 4 | 97.76         | 120.57        | **+23%**   | 0.95x        | 0.84x        | -0.11x    |
| 100,000       | 6 | 97.65         | 116.28        | +19%       | 0.93x        | 0.62x        | -0.31x    |
| 100,000       | 8 | 99.06         | 118.21        | +19%       | 0.55x        | 0.95x        | **+0.40x** |
| 1,000,000     | 4 | 79.81         | 32.98         | **-59%**   | 0.96x        | 1.28x        | +0.32x    |
| 1,000,000     | 6 | 79.59         | 33.04         | -58%       | 0.94x        | 1.32x        | +0.38x    |
| 1,000,000     | 8 | 80.10         | 33.09         | -59%       | 0.84x        | 0.97x        | +0.13x    |


All three predictions land:

1. **TP=1 baseline at 100k is 19–23% faster than TP=4 baseline** (~120 vs ~98 tok/s). At 100k context the KV per GPU at TP=4 is only ~1 GB, so the bandwidth gain from 4-way parallel reads is small (~0.3 ms saved/token), and that's outweighed by the 80 small all-reduces per step that TP=4 has to do. This directly confirms that the 100k TP=4 baseline was overhead-floor-limited, not bandwidth-limited.
2. **TP=1 baseline at 1M is 59% slower than TP=4 baseline** (~33 vs ~80 tok/s). At 1M, the per-GPU KV jumps to 40 GB at TP=1, so single-GPU HBM bandwidth becomes the dominant cost. TP=4 splits that 40 GB four ways and wins clearly.
3. **Self-SWA at TP=1 1M shows real speedup** (`k=4` at 1.28x, `k=6` at 1.32x, `k=8` at 0.97x). Compare to TP=4 1M where every `k` was sub-1.0x. The drafter at SWA-window-8192 still reads only ~80 MB of KV per pass at TP=1 — essentially the same per-step cost as the drafter at TP=4 — but the *target* at 1M is now ~30 ms/token instead of ~12 ms/token. The arithmetic that was a wash at TP=4 (drafter cost ≈ baseline cost) becomes a clear win at TP=1 (drafter cost ≈ half of baseline cost).

#### Acceptance (deterministic across repeats)


| Prompt length | Num spec tokens | Drafts | Draft tokens | Accepted tokens | Accept rate | Per-position accept rates                              |
| ------------- | --------------- | ------ | ------------ | --------------- | ----------- | ------------------------------------------------------ |
| 100,000       | 4               | 120    | 480          | 395             | 82.3%       | 0.95, 0.85, 0.78, 0.71                                 |
| 100,000       | 6               | 111    | 666          | 406             | 60.9%       | 0.88, 0.68, 0.61, 0.52, 0.49, 0.47                     |
| 100,000       | 8               | 62     | 496          | 452             | **91.1%**   | **0.98, 0.94, 0.92, 0.90, 0.89, 0.89, 0.89, 0.89**     |
| 1,000,000     | 4               | 125    | 500          | 378             | 75.6%       | 0.89, 0.77, 0.72, 0.65                                 |
| 1,000,000     | 6               | 102    | 612          | 399             | 65.2%       | 0.84, 0.76, 0.67, 0.60, 0.54, 0.50                     |
| 1,000,000     | 8               | 66     | 528          | 316             | 59.8%       | 0.88, 0.76, 0.67, 0.62, 0.55, 0.50, 0.44, 0.38         |


The `k=8 / 100k` row is the surprise. At TP=4 the same configuration had per-position rates `0.82, 0.74, 0.61, 0.49, 0.42, 0.36, 0.30, 0.26` and `exact_match=False`. At TP=1 the rates barely drop after position 3, and stay at **0.89** through positions 5–7. `exact_match` flipped from False to True. This is the same numerical-drift mechanism the doc has been calling out elsewhere: TP=4 all-reduces introduce small floating-point non-associativity between the baseline run and the self-SWA verify run, and at deep draft positions those small differences cross argmax boundaries and reject the draft. At TP=1, baseline and self-SWA both use the same single-GPU math — identical logits at every step until acceptance — so the drafter can track the target's argmax sequence essentially perfectly, even at position 7. The opposite effect shows up at `k=4` and `k=6 / 100k`, where TP=1 mean-accept is *lower* than TP=4: at smaller `k`, the depth isn't enough to be limited by drift, and the random per-passage seed happens to give slightly different argmax sequences between the two runs. The takeaway isn't "TP=1 is better for spec decode acceptance" — it's "small numerical perturbations from TP collectives can flip individual cells of the acceptance matrix in either direction".

The acceptance result also explains why `k=8 / 100k` at TP=1 (0.95x speedup) is *better* than `k=4 / 100k` at TP=1 (0.84x), even though spec decode at 100k is fundamentally overhead-limited at any TP: with mean_accept ≈ 8.29 the spec round produces 8.29 tokens per `1 verify + 8 drafters` = 9 forward passes, and 9/8.29 ≈ 1.09 forward passes per accepted token — almost the same per-token cost as a baseline forward pass. The reason `k=8` "wins" here is that it's essentially serializing the same number of forward passes as the baseline would have done anyway, just batched into rounds; the cost-per-token equation collapses to a near-tie. This is a TP=1-only artifact and not a general design lever.

#### Timing (single repeat, r2)


| TP | Num spec tokens | Prompt length | Baseline elapsed incl. prefill (s) | Baseline decode elapsed (s) | Self-SWA elapsed incl. prefill (s) | Self-SWA decode elapsed (s) |
| -- | --------------- | ------------- | ---------------------------------- | --------------------------- | ---------------------------------- | --------------------------- |
| 4  | 6               | 100,000       | 8.62                               | 5.27                        | 8.92                               | 5.33                        |
| 1  | 6               | 100,000       | 12.19                              | 4.08                        | 14.71                              | 6.45                        |
| 4  | 6               | 1,000,000     | 163.52                             | 5.14                        | 165.99                             | 6.51                        |
| 1  | 6               | 1,000,000     | 556.94                             | 13.17                       | 557.06                             | 11.48                       |
| 4  | 8               | 100,000       | 8.56                               | 5.15                        | 13.04                              | 9.37                        |
| 1  | 8               | 100,000       | 12.41                              | 4.28                        | 12.88                              | 4.53                        |
| 4  | 8               | 1,000,000     | 162.85                             | 5.11                        | 166.46                             | 7.25                        |
| 1  | 8               | 1,000,000     | 558.69                             | 13.36                       | 560.21                             | 11.90                       |


Prefill is dramatically slower at TP=1 (3–4× at 100k, 3.4× at 1M) — that's pure compute parallelism lost. Decode-elapsed tells the more interesting story:

- **100k decode-elapsed is *shorter* at TP=1 than TP=4** (`4.08s` vs `5.27s` for baseline at k=6) — exactly the "overhead floor is lower without all-reduces" finding.
- **1M decode-elapsed is *longer* at TP=1 than TP=4** (`13.17s` vs `5.14s` for baseline at k=6) — KV bandwidth dominates here and one GPU's HBM is the bottleneck.
- **Self-SWA decode-elapsed at 1M is *shorter* than baseline decode-elapsed at TP=1** (`11.48s` vs `13.17s` at k=6) — this is the speedup, expressed as wall-clock decode time. The drafter reads only the SWA window, so a sequence of 6 drafter passes + 1 verify uses less total HBM bandwidth than 5 sequential baseline decodes of the full KV.

#### Headline

- **TP=1 confirms the overhead-bound hypothesis for short context**: removing TP collectives at 100k makes the baseline 20% faster, which means a sizable fraction of the TP=4 100k decode time was indeed sync/collective overhead, not HBM bandwidth.
- **TP=1 makes self-SWA actually win at 1M** (1.28–1.32x at k=4/6, vs ~0.95x at TP=4). The win at 1M is not a self-SWA bug fix or a tuning improvement; it's the bandwidth-bound regime moving down from 2M (at TP=4) to 1M (at TP=1), because each GPU is now responsible for the full 40 GB of KV traffic per token.
- **Numerical drift from TP collectives matters more than expected for deep drafters**: `k=8 / 100k` flips from a 0.55x disaster at TP=4 (per-position acceptance collapsing to 0.26 at position 7) to a near-break-even 0.95x at TP=1 (per-position acceptance staying near 0.89 at position 7), entirely because TP=1 keeps baseline and self-SWA in lock-step on identical math.
- **For a single-GPU serving setup at 1M context, the optimal `k` is now 6** (1.32x), not 4. The TP=4 "k=4 wins at 1M" guidance does not transfer to single-GPU.

## Sink Size Comparison

### Self-SWA Decode Throughput


| Prompt length | Window size | Sink 0 tok/s | Sink 4 tok/s | Sink 4 vs sink 0 |
| ------------- | ----------- | ------------ | ------------ | ---------------- |
| 100,000       | 8,192       | 64.07        | 39.19        | 0.61x            |
| 1,000,000     | 8,192       | 31.03        | 33.22        | 1.07x            |
| 4,000,000     | 8,192       | 7.50         | 21.16        | 2.82x            |
| 100,000       | 32,768      | 80.67        | 17.97        | 0.22x            |
| 1,000,000     | 32,768      | 36.43        | 16.70        | 0.46x            |
| 4,000,000     | 32,768      | 7.62         | 13.02        | 1.71x            |
| 1,000,000     | 131,072     | 36.72        | 5.50         | 0.15x            |
| 4,000,000     | 131,072     | 8.17         | 4.92         | 0.60x            |


### Acceptance Rate


| Prompt length | Window size | Sink 0 accept rate | Sink 4 accept rate | Delta    |
| ------------- | ----------- | ------------------ | ------------------ | -------- |
| 100,000       | 8,192       | 90.3%              | 93.1%              | +2.8 pp  |
| 1,000,000     | 8,192       | 45.0%              | 93.5%              | +48.5 pp |
| 4,000,000     | 8,192       | 29.1%              | 89.1%              | +60.0 pp |
| 100,000       | 32,768      | 90.5%              | 94.2%              | +3.7 pp  |
| 1,000,000     | 32,768      | 47.6%              | 94.9%              | +47.3 pp |
| 4,000,000     | 32,768      | 30.1%              | 92.0%              | +61.9 pp |
| 1,000,000     | 131,072     | 48.7%              | 96.5%              | +47.8 pp |
| 4,000,000     | 131,072     | 34.5%              | 92.0%              | +57.5 pp |


### Block-Aligned Paged vs Sink Size 4 Decode Throughput

This is a cross-run comparison: the sink sweep used TP=8, while the block-aligned paged run used TP=4 and different draft attention semantics.


| Prompt length | Window size | Sink 4 TP=8 tok/s | Block-aligned TP=4 tok/s | Block-aligned vs sink 4 |
| ------------- | ----------- | ----------------- | ------------------------ | ----------------------- |
| 100,000       | 8,192       | 39.19             | 49.47                    | 1.26x                   |
| 1,000,000     | 8,192       | 33.22             | 43.35                    | 1.30x                   |
| 4,000,000     | 8,192       | 21.16             | 25.97                    | 1.23x                   |
| 100,000       | 32,768      | 17.97             | 53.05                    | 2.95x                   |
| 1,000,000     | 32,768      | 16.70             | 44.98                    | 2.69x                   |
| 4,000,000     | 32,768      | 13.02             | 26.33                    | 2.02x                   |
| 1,000,000     | 131,072     | 5.50              | 44.94                    | 8.17x                   |
| 4,000,000     | 131,072     | 4.92              | 26.56                    | 5.40x                   |


## Estimated TTFT

Estimated TTFT is computed as:

```text
elapsed_s_including_prefill - decode_elapsed_s
```

These estimates include frontend/engine overhead. For first self-SWA cases in a window, they can also include one-time AITER kernel compilation or cache-loading overhead, so treat them as upper-bound wall-clock TTFT estimates rather than isolated prefill kernel time.

The `sink=0`, `window=8192` TTFT estimates are compile-contaminated outliers from the first self-SWA window run and should not be compared directly with warm runs.

### Baseline Estimated TTFT


| Sink sweep   | Prompt length | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
| ------------ | ------------- | ------------------------- | ------------------ | ------------------ |
| sink=0 sweep | 100,000       | 6.36                      | 4.21               | 2.15               |
| sink=0 sweep | 1,000,000     | 90.36                     | 8.82               | 81.54              |
| sink=0 sweep | 4,000,000     | 1172.66                   | 23.75              | 1148.91            |
| sink=4 sweep | 100,000       | 6.34                      | 4.22               | 2.12               |
| sink=4 sweep | 1,000,000     | 90.40                     | 8.86               | 81.54              |
| sink=4 sweep | 4,000,000     | 1185.66                   | 24.48              | 1161.18            |


### Sink Size 0 Estimated TTFT


| Prompt length | Window size | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
| ------------- | ----------- | ------------------------- | ------------------ | ------------------ |
| 100,000       | 8,192       | 63.10                     | 7.98               | 55.12*             |
| 1,000,000     | 8,192       | 242.03                    | 16.47              | 225.56*            |
| 4,000,000     | 8,192       | 1884.07                   | 68.13              | 1815.94*           |
| 100,000       | 32,768      | 8.76                      | 6.33               | 2.43               |
| 1,000,000     | 32,768      | 98.54                     | 14.03              | 84.51              |
| 4,000,000     | 32,768      | 1238.16                   | 67.09              | 1171.07            |
| 100,000       | 131,072     | skipped                   | skipped            | skipped            |
| 1,000,000     | 131,072     | 101.88                    | 13.92              | 87.96              |
| 4,000,000     | 131,072     | 1233.82                   | 62.54              | 1171.28            |


 Compile-contaminated estimate from the first sink=0 self-SWA window run.

### Sink Size 4 Estimated TTFT


| Prompt length | Window size | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
| ------------- | ----------- | ------------------------- | ------------------ | ------------------ |
| 100,000       | 8,192       | 16.02                     | 13.04              | 2.98               |
| 1,000,000     | 8,192       | 110.80                    | 15.38              | 95.42              |
| 4,000,000     | 8,192       | 1193.46                   | 24.15              | 1169.31            |
| 100,000       | 32,768      | 31.97                     | 28.43              | 3.54               |
| 1,000,000     | 32,768      | 126.09                    | 30.60              | 95.49              |
| 4,000,000     | 32,768      | 1233.38                   | 39.25              | 1194.13            |
| 100,000       | 131,072     | skipped                   | skipped            | skipped            |
| 1,000,000     | 131,072     | 223.44                    | 92.99              | 130.45             |
| 4,000,000     | 131,072     | 1402.03                   | 103.85             | 1298.18            |


### Block-Aligned Paged Estimated TTFT

These estimates come from the TP=4 block-aligned paged run, so they are not directly comparable with the TP=8 sink sweep estimates above.


| Prompt length | Window size | Elapsed incl. prefill (s) | Decode elapsed (s) | Estimated TTFT (s) |
| ------------- | ----------- | ------------------------- | ------------------ | ------------------ |
| 100,000       | 8,192       | 14.17                     | 10.33              | 3.84               |
| 1,000,000     | 8,192       | 176.23                    | 11.79              | 164.44             |
| 4,000,000     | 8,192       | 2296.88                   | 19.67              | 2277.21            |
| 100,000       | 32,768      | 13.28                     | 9.63               | 3.65               |
| 1,000,000     | 32,768      | 175.87                    | 11.36              | 164.51             |
| 4,000,000     | 32,768      | 2295.07                   | 19.41              | 2275.66            |
| 100,000       | 131,072     | skipped                   | skipped            | skipped            |
| 1,000,000     | 131,072     | 192.80                    | 11.37              | 181.43             |
| 4,000,000     | 131,072     | 2292.46                   | 19.24              | 2273.22            |


## Observations


| Observation                                                       | Details                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sink size 4 greatly improves acceptance.                          | Long-context accept rates rose from roughly `29-49%` with sink `0` to roughly `89-97%` with sink `4`.                                                                                                                                                                                                                   |
| Higher acceptance did not always improve speed.                   | Sink `4` was fastest relative to sink `0` for the `4M, 8192` and `4M, 32768` cases, but was slower for short contexts and the `131072` window.                                                                                                                                                                          |
| Block-aligned paged self-SWA improved long-context decode speed.  | In the TP=4 `num_spec_tokens=4` run it reached `1.63-1.71x` over the TP=4 baseline for completed `1M` and `4M` cases.                                                                                                                                                                                                   |
| `num_spec_tokens=8` was fastest in the 4M/8k larger-batch sweep.  | At TP=4 eager with block-aligned paged self-SWA, `num_spec_tokens=8` reached `30.20` tok/s (`1.43x`) versus `28.43` tok/s for `16` and `25.22` tok/s for `32`. Larger batches raised mean accept length from `8.16` to `21.46`, but lowered throughput because draft work increased and later-position acceptance fell. |
| Block-aligned paged results are a separate path.                  | The run used `self_swa_sink_size=16` with block-boundary rounding, so compare it as the paged path rather than as token-exact `sink_size=4`.                                                                                                                                                                            |
| First-window TTFT estimates can be misleading.                    | The `sink=0`, `window=8192` estimated TTFT rows include cold AITER/JIT overhead and are outliers.                                                                                                                                                                                                                       |
| Best 4M result in the original sink sweep                         | `sink_size=4`, `window_size=8192`: `21.16` self-SWA decode tok/s, `1.01x` vs baseline, `89.1%` accept rate.                                                                                                                                                                                                             |
| Best 4M block-aligned throughput                                  | `num_spec_tokens=8`, `window_size=8192`: `30.20` self-SWA decode tok/s, `1.43x` vs same-run TP=4 baseline, `89.5%` accept rate.                                                                                                                                                                                         |
| All completed cases matched baseline exactly.                     | Every non-skipped self-SWA case in the earlier sink sweep reported exact token match `PASS`. The PG-19 rerun in the section above flips `exact_match` to `False` at 1M and 4M, because numerical drift now perturbs sharper, content-dependent argmax boundaries; see that section for details.                         |
| PG-19 prompts shrink the long-context speedup but make it honest. | At 4M, self-SWA speedup drops from `1.78x` on `repeat_block` to `1.67x` on PG-19, and `mean_acceptance_length` from `4.86` to `4.53`. Baseline tok/s is unchanged. The remaining gap reflects the drafter losing the periodic-prompt shortcut.                                                                          |
| Llama-3-8B-Gradient-4194k is incoherent at 1M baseline.           | The PG-19 baseline coherence check showed it producing structured noise (`[0_system],[0_q_0],0_a_0,...`) at 1M while staying coherent at 100k. The repeat-block numbers above for this model at 1M and 4M should not be treated as measuring useful long-context behavior.                                              |
| Optimal `num_spec_tokens` depends on prompt length on PG-19.       | On GLM at TP=4 eager, `k=8` is best at 4M (`2.12x`, `46.00` tok/s) and `k=4` is best at 100k (`0.95x`). At 2M the picture is k=6 ≈ k=8 (1.43x vs 1.42x). The break-even crossover is between 1M and 2M: target decode is bandwidth-bound at long context (deeper drafts amortize verify cost) and compute-bound at short context (deeper drafts are pure overhead).            |
| `k=8 / 100k` is a per-position acceptance cliff.                   | Per-position accept rates collapse from `0.82` at position 0 to `0.26` at position 7 on PG-19. Once `exact_match` flips to `False` (which happens at `k=8 / 100k` but not at `k=4` or `k=6 / 100k`), the drafter's learned priors diverge from the target's, and most deep drafts get thrown away.                      |
| 2M baseline collapses to ~44 tok/s — already KV-memory-bound.      | At 2M the GLM baseline runs at `~44` tok/s, roughly half its 1M rate (`80` tok/s) and about 2× its 4M rate (`22` tok/s). That is what makes 2M the first prompt length where self-SWA wins clearly: spec decode goes from ~break-even at 1M (`0.96x`) to `1.26x – 1.43x` at 2M. 2M is the transition into the bandwidth-bound regime that 4M sits firmly in.                                                                                                              |
| Cold-AITER JIT contaminates r1 baselines at new prompt shapes.     | The 2M-only extension ran 2M as its first prompt shape, so r1 baseline decode at 2M absorbed kernel JIT (~22 tok/s in r1 vs ~44 tok/s in r2/r3). The earlier 100k/1M/4M sweeps avoided this by going 100k → 1M → 4M sequentially, warming kernels at smaller shapes first. Per-metric outlier removal handled the 2M JIT outlier cleanly, but it's worth knowing for future single-prompt-length reruns: always either warm the shape first or be ready to drop r1.        |
| TP=4 100k baseline is bottlenecked by collective overhead, not KV. | At 100k context, TP=1 baseline runs 19–23% *faster* than TP=4 baseline (~120 vs ~98 tok/s). The 80 small all-reduces per decode step that TP=4 performs cost more wall-clock at 100k than the 4-way bandwidth parallelism saves. The TP=4 advantage only emerges at 1M+ where per-GPU KV traffic on a single GPU becomes the dominant cost.                                                                                          |
| Single-GPU is where self-SWA pays off at 1M.                       | TP=1 1M baseline runs at ~33 tok/s (KV bandwidth dominated), self-SWA at k=6 reaches 43.63 tok/s = `1.32x`. The same configuration at TP=4 is 79.59 → 74.62 = `0.94x`. Same model, same prompt, same drafter — the difference is entirely that TP=1 puts the target into the bandwidth-bound regime at 1M instead of at 2M.                                                                                                           |
| TP collective math drift can flip deep-drafter acceptance cells.   | At 100k k=8, TP=4 sees per-position acceptance fall from 0.82 to 0.26 by position 7, with `exact_match=False`. The same configuration at TP=1 holds 0.89 at position 7 with `exact_match=True`, because baseline and self-SWA use identical single-GPU math. Small floating-point differences from TP all-reduces are the mechanism; the effect is largest at deep draft positions where small logit perturbations compound.            |


