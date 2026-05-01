# Self-SWA Speculative Decoding

Self-SWA speculative decoding is an experimental greedy-only method that uses
the target model itself as the drafter. The draft pass temporarily applies a
fixed sliding-window attention mask during decode, writes provisional KV into
the target request's lookahead slots, and lets the normal full-attention target
pass verify the draft tokens.

This method is intended for standard decoder-only, full-attention models with
very long contexts, where the sliding-window draft can be cheaper than full
attention after the prompt exceeds the configured window size.

## Manual ROCm Validation

The first validation target is `Qwen/Qwen2.5-7B-Instruct-1M` on ROCm AITER.
Before testing, keep shuffle KV cache disabled:

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
    },
)

baseline_outputs = baseline.generate(prompts, sampling_params)
self_swa_outputs = self_swa.generate(prompts, sampling_params)

assert baseline_outputs[0].outputs[0].token_ids == (
    self_swa_outputs[0].outputs[0].token_ids
)
```

After exactness is confirmed, sweep `num_speculative_tokens` values such as
`2`, `4`, and `8`, and record acceptance length, tokens/sec, and draft overhead.
