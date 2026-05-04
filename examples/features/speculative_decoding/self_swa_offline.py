# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate self-SWA speculative decoding on GLM-4-9B-Chat-1M.

This script runs a greedy baseline and a greedy self-SWA run with the same
long prompt, compares generated token IDs, and prints self-SWA acceptance
metrics. It is intended for ROCm AITER with shuffled KV cache disabled.
"""

import gc
import os
import time
from argparse import BooleanOptionalAction, Namespace
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

# These environment variables must be set before importing vLLM.
os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm-cache")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "False")

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Metric, Vector

MODEL_NAME = "zai-org/glm-4-9b-chat-1m"
DEFAULT_PROMPT_LEN = 8192
DEFAULT_SELF_SWA_WINDOW_SIZE = 4096


@dataclass
class RunResult:
    elapsed_s: float
    output_tokens: int
    wall_output_tokens_per_s: float
    decode_tokens: int
    decode_elapsed_s: float
    decode_tokens_per_s: float
    output_token_ids: list[list[int]]
    output_texts: list[str]
    metrics: list[Metric] | None


def _env_is_true(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def validate_env() -> None:
    if not _env_is_true(os.environ["VLLM_ROCM_USE_AITER"]):
        raise RuntimeError("self-SWA validation requires VLLM_ROCM_USE_AITER=1.")
    if _env_is_true(os.environ["VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT"]):
        raise RuntimeError(
            "self-SWA validation requires "
            "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=False."
        )


def parse_args() -> Namespace:
    parser = FlexibleArgumentParser(
        description="Run greedy baseline vs self-SWA on GLM-4-9B-Chat-1M."
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    parser.add_argument(
        "--self-swa-window-size",
        type=int,
        default=DEFAULT_SELF_SWA_WINDOW_SIZE,
        help="Sliding-window size used by the self-SWA drafter.",
    )
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attention-backend", default="ROCM_AITER_FA")
    parser.add_argument(
        "--trust-remote-code", action=BooleanOptionalAction, default=True
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


def build_prompt_token_ids(
    tokenizer: AutoTokenizer,
    min_prompt_len: int,
    max_prompt_len: int,
    prompt_index: int,
    self_swa_window_size: int,
) -> list[int]:
    if min_prompt_len <= self_swa_window_size:
        raise ValueError(
            f"--prompt-len should exceed the self-SWA window "
            f"({self_swa_window_size}); got {min_prompt_len}."
        )
    if min_prompt_len > max_prompt_len:
        raise ValueError(
            f"--prompt-len={min_prompt_len} leaves no room for generation "
            f"under --max-model-len; max allowed prompt length is {max_prompt_len}."
        )

    paragraph = (
        f"Self-SWA GLM validation passage {prompt_index}. "
        "This text is intentionally repetitive so the prompt can exceed the "
        "sliding-window draft size while remaining deterministic. "
        "The generated answer should be concise and stable under greedy decoding.\n"
    )

    repeats = 1
    while True:
        content = paragraph * repeats
        messages = [
            {
                "role": "user",
                "content": (
                    content
                    + "\nSummarize the validation passage in one short sentence."
                ),
            }
        ]
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        if len(token_ids) >= min_prompt_len:
            break
        repeats *= 2

    if len(token_ids) > max_prompt_len:
        # Preserve the assistant-generation suffix from the chat template.
        token_ids = token_ids[-max_prompt_len:]
    return token_ids


def build_prompts(args: Namespace) -> list[dict[str, list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    max_prompt_len = args.max_model_len - args.output_len - args.num_spec_tokens - 1
    return [
        {
            "prompt_token_ids": build_prompt_token_ids(
                tokenizer,
                args.prompt_len,
                max_prompt_len,
                prompt_index=i,
                self_swa_window_size=args.self_swa_window_size,
            )
        }
        for i in range(args.num_prompts)
    ]


def make_llm_kwargs(args: Namespace, speculative_config: dict[str, Any] | None) -> dict:
    kwargs = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "enable_chunked_prefill": not args.disable_chunked_prefill,
        "attention_backend": args.attention_backend,
        "disable_log_stats": False,
    }
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    return kwargs


def cleanup_llm(llm: LLM | None) -> None:
    if llm is not None:
        with suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
        del llm
    gc.collect()
    if hasattr(torch, "accelerator"):
        torch.accelerator.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()


def get_decode_timing(outputs, token_ids: list[list[int]]) -> tuple[int, float, float]:
    decode_tokens = sum(max(len(ids) - 1, 0) for ids in token_ids)
    if decode_tokens == 0:
        return 0, 0.0, 0.0

    first_token_ts = []
    last_token_ts = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.first_token_ts <= 0 or metrics.last_token_ts <= 0:
            continue
        first_token_ts.append(metrics.first_token_ts)
        last_token_ts.append(metrics.last_token_ts)

    if not first_token_ts or not last_token_ts:
        return decode_tokens, 0.0, 0.0

    decode_elapsed_s = max(last_token_ts) - min(first_token_ts)
    if decode_elapsed_s <= 0:
        return decode_tokens, decode_elapsed_s, 0.0
    return decode_tokens, decode_elapsed_s, decode_tokens / decode_elapsed_s


def run_llm(
    name: str,
    args: Namespace,
    prompts: list[dict[str, list[int]]],
    speculative_config: dict[str, Any] | None,
) -> RunResult:
    print(f"\n=== {name} ===")
    llm: LLM | None = None
    try:
        llm = LLM(**make_llm_kwargs(args, speculative_config))
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.output_len,
        )
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        elapsed_s = time.perf_counter() - start
        metrics = llm.get_metrics()
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        texts = [output.outputs[0].text for output in outputs]
        total_output_tokens = sum(len(ids) for ids in token_ids)
        wall_toks_per_s = total_output_tokens / elapsed_s if elapsed_s > 0 else 0.0
        decode_tokens, decode_elapsed_s, decode_toks_per_s = get_decode_timing(
            outputs, token_ids
        )
        print(f"elapsed_s_including_prefill: {elapsed_s:.2f}")
        print(f"output_tokens: {total_output_tokens}")
        print(f"output_tokens_per_s_including_prefill: {wall_toks_per_s:.2f}")
        print(f"decode_tokens_excluding_first: {decode_tokens}")
        print(f"decode_elapsed_s: {decode_elapsed_s:.2f}")
        print(f"decode_tokens_per_s: {decode_toks_per_s:.2f}")
        return RunResult(
            elapsed_s,
            total_output_tokens,
            wall_toks_per_s,
            decode_tokens,
            decode_elapsed_s,
            decode_toks_per_s,
            token_ids,
            texts,
            metrics,
        )
    finally:
        cleanup_llm(llm)


def print_spec_metrics(metrics: list[Metric] | None, num_spec_tokens: int) -> None:
    if metrics is None:
        return

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos, count in enumerate(metric.values[:num_spec_tokens]):
                acceptance_counts[pos] += count

    acceptance_length = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    )
    print("\n=== self-SWA metrics ===")
    print(f"num_drafts: {num_drafts}")
    print(f"num_draft_tokens: {num_draft_tokens}")
    print(f"num_accepted_tokens: {num_accepted_tokens}")
    print(f"mean_acceptance_length: {acceptance_length:.2f}")
    for pos, count in enumerate(acceptance_counts):
        rate = count / num_drafts if num_drafts > 0 else 0.0
        print(f"acceptance_at_token_{pos}: {rate:.2f}")


def print_speed_comparison(baseline: RunResult, self_swa: RunResult) -> None:
    baseline_speed = baseline.decode_tokens_per_s
    self_swa_speed = self_swa.decode_tokens_per_s
    speedup = self_swa_speed / baseline_speed if baseline_speed > 0 else 0.0

    print("\n=== decode speed comparison ===")
    print(f"baseline_decode_tokens: {baseline.decode_tokens}")
    print(f"baseline_decode_elapsed_s: {baseline.decode_elapsed_s:.2f}")
    print(f"baseline_decode_tokens_per_s: {baseline_speed:.2f}")
    print(f"self_swa_decode_tokens: {self_swa.decode_tokens}")
    print(f"self_swa_decode_elapsed_s: {self_swa.decode_elapsed_s:.2f}")
    print(f"self_swa_decode_tokens_per_s: {self_swa_speed:.2f}")
    print(f"self_swa_vs_baseline_decode_speedup: {speedup:.2f}x")
    if baseline_speed == 0 or self_swa_speed == 0:
        print(
            "decode speed unavailable; use --output-len > 1 and keep "
            "disable_log_stats=False"
        )


def compare_outputs(baseline: RunResult, self_swa: RunResult) -> None:
    mismatches = []
    for i, (baseline_ids, self_swa_ids) in enumerate(
        zip(baseline.output_token_ids, self_swa.output_token_ids)
    ):
        if baseline_ids != self_swa_ids:
            mismatches.append(i)

    if not mismatches:
        print("\nExact token match: PASS")
        return

    print("\nExact token match: FAIL")
    for i in mismatches[:3]:
        print(f"prompt_index: {i}")
        print(f"baseline_token_ids: {baseline.output_token_ids[i]}")
        print(f"self_swa_token_ids: {self_swa.output_token_ids[i]}")
    raise AssertionError(f"{len(mismatches)} prompt(s) differed.")


def main() -> None:
    args = parse_args()
    validate_env()
    prompts = build_prompts(args)
    prompt_lens = [len(prompt["prompt_token_ids"]) for prompt in prompts]
    print(f"model: {args.model}")
    print(f"prompt_lens: {prompt_lens}")
    print(f"VLLM_ROCM_USE_AITER={os.environ['VLLM_ROCM_USE_AITER']}")
    print(
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT="
        f"{os.environ['VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT']}"
    )

    baseline = run_llm("baseline", args, prompts, speculative_config=None)

    self_swa_config = {
        "method": "self_swa",
        "num_speculative_tokens": args.num_spec_tokens,
        "self_swa_window_size": args.self_swa_window_size,
    }
    self_swa = run_llm("self-SWA", args, prompts, self_swa_config)

    print_speed_comparison(baseline, self_swa)
    compare_outputs(baseline, self_swa)
    print_spec_metrics(self_swa.metrics, args.num_spec_tokens)

    if args.print_output:
        print("\n=== generated text ===")
        for i, text in enumerate(self_swa.output_texts):
            print(f"[{i}] {text}")


if __name__ == "__main__":
    main()
