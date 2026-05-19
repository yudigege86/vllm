# Self-SWA Speculative Decoding

Self-SWA speculative decoding is an experimental greedy-only method that uses
the target model itself as the drafter. The draft pass temporarily keeps the
initial attention-sink tokens plus a fixed recent sliding window during decode,
writes provisional KV into the target request's lookahead slots, and lets the
normal full-attention target pass verify the draft tokens. Use greedy sampling
(`temperature=0.0`) when validating self-SWA output exactness.

This method is intended for standard decoder-only, full-attention models with
very long contexts, where the sliding-window draft can be cheaper than full
attention after the prompt exceeds the configured window size.

## Attention Sink Path

Self-SWA uses a block-aligned paged draft attention path. The draft attention
rounds the configured sink and recent regions outward to KV block boundaries,
builds synthetic block tables, and runs paged attention over those blocks.
For example, with a 16-token KV block size, `self_swa_sink_size=4` and
`self_swa_sink_size=16` both keep one full sink block.

The block-aligned paged path is ROCm AITER focused and still experimental. It
requires shuffle KV cache layout to be disabled:

```bash
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False
```

## Manual ROCm Validation

One validation target is `Qwen/Qwen2.5-7B-Instruct-1M` on ROCm AITER. Before
testing, keep shuffle KV cache disabled:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False
```

Run a greedy baseline and a `self_swa` run with the same prompts, then compare
the generated token IDs:

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
prompts = ["<long prompt exceeding the self-SWA window>"]

baseline = LLM(model="Qwen/Qwen2.5-7B-Instruct-1M")
self_swa = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-1M",
    speculative_config={
        "method": "self_swa",
        "num_speculative_tokens": 4,
        "self_swa_sink_size": 4,
    },
)

baseline_outputs = baseline.generate(prompts, sampling_params)
self_swa_outputs = self_swa.generate(prompts, sampling_params)

assert baseline_outputs[0].outputs[0].token_ids == (
    self_swa_outputs[0].outputs[0].token_ids
)
```

After exactness is confirmed, sweep `num_speculative_tokens` values and record
acceptance length, tokens/sec, and draft overhead. You can also compare
different `self_swa_sink_size` values to measure the effect of keeping initial
sink blocks plus recent blocks. See
[`self_swa_sink_sweep_results.md`](../../../examples/features/speculative_decoding/self_swa_sink_sweep_results.md)
for benchmark notes, including TP=4 eager block-aligned runs that exact-matched
completed greedy baselines and improved long-context decode throughput in those
tests.
