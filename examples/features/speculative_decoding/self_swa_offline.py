# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep self-SWA speculative decoding on GLM-4-9B-Chat-1M.

This script creates max-context engines for the sweep: one greedy baseline
engine, then one greedy self-SWA engine per sliding-window size. Each engine runs
the requested prompt lengths, comparing generated token IDs and printing speed
and per-case self-SWA acceptance metrics. It is intended for ROCm AITER with
shuffled KV cache disabled.
"""

import gc
import os
import time
from argparse import BooleanOptionalAction, Namespace
from contextlib import suppress
from copy import copy
from dataclasses import dataclass
from typing import Any

# These environment variables must be set before importing vLLM.
os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm-cache")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT", "False")

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.metrics.reader import Counter, Metric, Vector

MODEL_NAME = "zai-org/glm-4-9b-chat-1m"
DEFAULT_PROMPT_LENS = [100_000, 1_000_000, 2_000_000, 4_000_000]
DEFAULT_SELF_SWA_WINDOW_SIZES = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
]
DEFAULT_SELF_SWA_SINK_SIZE = 4
DEFAULT_TENSOR_PARALLEL_SIZE = 8


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


@dataclass
class SpecMetrics:
    num_drafts: int
    num_draft_tokens: int
    num_accepted_tokens: int
    acceptance_counts: list[int]
    mean_acceptance_length: float
    acceptance_rates: list[float]


@dataclass
class SweepResult:
    prompt_len: int
    actual_prompt_lens: list[int]
    max_model_len: int
    self_swa_window_size: int
    exact_match: bool
    baseline_decode_tokens_per_s: float
    self_swa_decode_tokens_per_s: float
    speedup: float
    spec_metrics: SpecMetrics


@dataclass
class PromptCase:
    prompt_len: int
    actual_prompt_lens: list[int]
    prompts: list[dict[str, list[int]]]


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


def _format_int_list(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _parse_human_int(value: str) -> int:
    normalized = value.strip().replace("_", "")
    if not normalized:
        raise ValueError("empty value")

    suffix = normalized[-1].lower()
    multiplier = 1
    if suffix in ("k", "m", "g"):
        multiplier = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}[suffix]
        normalized = normalized[:-1]

    parsed = int(float(normalized) * multiplier)
    if parsed <= 0:
        raise ValueError(f"expected a positive integer, got {value!r}")
    return parsed


def _parse_int_list(value: str, option_name: str) -> list[int]:
    try:
        parsed = [_parse_human_int(item) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"invalid {option_name}={value!r}: {exc}") from exc

    if not parsed:
        raise ValueError(f"{option_name} must contain at least one value")
    return parsed


def parse_args() -> Namespace:
    parser = FlexibleArgumentParser(
        description="Run a greedy baseline vs self-SWA sweep on GLM-4-9B-Chat-1M."
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--prompt-lens",
        default=_format_int_list(DEFAULT_PROMPT_LENS),
        help=(
            "Comma-separated prompt lengths to sweep. Suffixes k/m/g use "
            "decimal units, e.g. 100k,1m,2m,4m."
        ),
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=None,
        help="Optional single prompt length override for the old one-shot workflow.",
    )
    parser.add_argument(
        "--self-swa-window-sizes",
        default=_format_int_list(DEFAULT_SELF_SWA_WINDOW_SIZES),
        help="Comma-separated self-SWA window sizes to sweep.",
    )
    parser.add_argument(
        "--self-swa-window-size",
        type=int,
        default=None,
        help="Optional single self-SWA window size override.",
    )
    parser.add_argument(
        "--self-swa-sink-size",
        type=int,
        default=DEFAULT_SELF_SWA_SINK_SIZE,
        help="Initial attention-sink tokens kept by the self-SWA drafter.",
    )
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--num-spec-tokens", type=int, default=4)
    parser.add_argument("--tp", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Override max_model_len for every sweep case. By default each "
            "context uses prompt_len + output_len + num_spec_tokens + 1."
        ),
    )
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
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


def resolve_prompt_lens(args: Namespace) -> list[int]:
    if args.prompt_len is not None:
        if args.prompt_len <= 0:
            raise ValueError("--prompt-len must be positive")
        return [args.prompt_len]
    return _parse_int_list(args.prompt_lens, "--prompt-lens")


def resolve_window_sizes(args: Namespace) -> list[int]:
    if args.self_swa_window_size is not None:
        if args.self_swa_window_size <= 0:
            raise ValueError("--self-swa-window-size must be positive")
        return [args.self_swa_window_size]
    return _parse_int_list(args.self_swa_window_sizes, "--self-swa-window-sizes")


def check_window_eligible(
    prompt_len: int, window_size: int, self_swa_sink_size: int
) -> tuple[bool, str | None]:
    if window_size > prompt_len:
        return False, "window is larger than the prompt"
    if prompt_len <= window_size + self_swa_sink_size:
        return False, "self-SWA visible context would cover the whole prompt"
    return True, None


def auto_max_model_len(args: Namespace, prompt_len: int) -> int:
    return prompt_len + args.output_len + args.num_spec_tokens + 1


def make_sweep_args(args: Namespace, prompt_lens: list[int]) -> Namespace:
    sweep_args = copy(args)
    sweep_args.max_model_len = args.max_model_len or max(
        auto_max_model_len(args, prompt_len) for prompt_len in prompt_lens
    )
    return sweep_args


def build_prompt_token_ids(
    tokenizer: AutoTokenizer,
    prompt_len: int,
    max_prompt_len: int,
    prompt_index: int,
) -> list[int]:
    if prompt_len > max_prompt_len:
        raise ValueError(
            f"--prompt-len={prompt_len} leaves no room for generation under "
            f"--max-model-len; max allowed prompt length is {max_prompt_len}."
        )

    prefix = f"Self-SWA GLM validation passage {prompt_index}.\n"
    repeat_block = (
        "This deterministic validation text is repeated to create a long "
        "prompt while keeping greedy decoding stable. "
    )
    suffix = "\nSummarize the validation passage in one short sentence."

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    repeat_ids = tokenizer.encode(repeat_block, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if not repeat_ids:
        raise ValueError("repeat block produced no tokens")

    fixed_len = len(prefix_ids) + len(suffix_ids)
    if prompt_len <= fixed_len:
        raise ValueError(
            f"--prompt-len={prompt_len} is too short for the fixed prompt "
            f"prefix/suffix ({fixed_len} tokens)."
        )

    repeated_len = prompt_len - fixed_len
    full_repeats, remainder = divmod(repeated_len, len(repeat_ids))
    return (
        prefix_ids
        + repeat_ids * full_repeats
        + repeat_ids[:remainder]
        + suffix_ids
    )


def build_prompts(
    tokenizer: AutoTokenizer, args: Namespace, prompt_len: int
) -> list[dict[str, list[int]]]:
    max_prompt_len = args.max_model_len - args.output_len - args.num_spec_tokens - 1
    return [
        {
            "prompt_token_ids": build_prompt_token_ids(
                tokenizer,
                prompt_len,
                max_prompt_len,
                prompt_index=i,
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


def create_llm(
    args: Namespace, speculative_config: dict[str, Any] | None
) -> LLM:
    return LLM(**make_llm_kwargs(args, speculative_config))


def run_generation(
    name: str,
    llm: LLM,
    args: Namespace,
    prompts: list[dict[str, list[int]]],
) -> RunResult:
    print(f"\n=== {name} ===")
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


def collect_spec_metrics(
    metrics: list[Metric] | None, num_spec_tokens: int
) -> SpecMetrics:
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * num_spec_tokens
    for metric in metrics or []:
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
    acceptance_rates = [
        count / num_drafts if num_drafts > 0 else 0.0
        for count in acceptance_counts
    ]
    return SpecMetrics(
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        acceptance_counts=acceptance_counts,
        mean_acceptance_length=acceptance_length,
        acceptance_rates=acceptance_rates,
    )


def diff_spec_metrics(before: SpecMetrics, after: SpecMetrics) -> SpecMetrics:
    num_drafts = after.num_drafts - before.num_drafts
    num_draft_tokens = after.num_draft_tokens - before.num_draft_tokens
    num_accepted_tokens = after.num_accepted_tokens - before.num_accepted_tokens
    acceptance_counts = [
        after_count - before_count
        for before_count, after_count in zip(
            before.acceptance_counts, after.acceptance_counts
        )
    ]
    acceptance_length = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    )
    acceptance_rates = [
        count / num_drafts if num_drafts > 0 else 0.0
        for count in acceptance_counts
    ]
    return SpecMetrics(
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        acceptance_counts=acceptance_counts,
        mean_acceptance_length=acceptance_length,
        acceptance_rates=acceptance_rates,
    )


def print_spec_metrics(spec_metrics: SpecMetrics) -> None:
    print("\n=== self-SWA metrics ===")
    print(f"num_drafts: {spec_metrics.num_drafts}")
    print(f"num_draft_tokens: {spec_metrics.num_draft_tokens}")
    print(f"num_accepted_tokens: {spec_metrics.num_accepted_tokens}")
    print(f"mean_acceptance_length: {spec_metrics.mean_acceptance_length:.2f}")
    for pos, rate in enumerate(spec_metrics.acceptance_rates):
        print(f"acceptance_at_token_{pos}: {rate:.2f}")


def compute_speedup(baseline: RunResult, self_swa: RunResult) -> float:
    baseline_speed = baseline.decode_tokens_per_s
    self_swa_speed = self_swa.decode_tokens_per_s
    return self_swa_speed / baseline_speed if baseline_speed > 0 else 0.0


def print_speed_comparison(baseline: RunResult, self_swa: RunResult) -> float:
    baseline_speed = baseline.decode_tokens_per_s
    self_swa_speed = self_swa.decode_tokens_per_s
    speedup = compute_speedup(baseline, self_swa)

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
    return speedup


def compare_outputs(
    baseline: RunResult, self_swa: RunResult, raise_on_mismatch: bool = True
) -> bool:
    mismatches = []
    for i, (baseline_ids, self_swa_ids) in enumerate(
        zip(baseline.output_token_ids, self_swa.output_token_ids)
    ):
        if baseline_ids != self_swa_ids:
            mismatches.append(i)

    if not mismatches:
        print("\nExact token match: PASS")
        return True

    print("\nExact token match: FAIL")
    for i in mismatches[:3]:
        print(f"prompt_index: {i}")
        print(f"baseline_token_ids: {baseline.output_token_ids[i]}")
        print(f"self_swa_token_ids: {self_swa.output_token_ids[i]}")
    if raise_on_mismatch:
        raise AssertionError(f"{len(mismatches)} prompt(s) differed.")
    return False


def build_prompt_cases(
    args: Namespace, tokenizer: AutoTokenizer, prompt_lens: list[int]
) -> list[PromptCase]:
    prompt_cases = []
    for prompt_len in prompt_lens:
        prompts = build_prompts(tokenizer, args, prompt_len)
        actual_prompt_lens = [len(prompt["prompt_token_ids"]) for prompt in prompts]
        prompt_cases.append(
            PromptCase(
                prompt_len=prompt_len,
                actual_prompt_lens=actual_prompt_lens,
                prompts=prompts,
            )
        )
        print("\n" + "=" * 80)
        print(f"prompt_len: {prompt_len}")
        print(f"actual_prompt_lens: {actual_prompt_lens}")
        print(f"max_model_len: {args.max_model_len}")
        print("=" * 80)
    return prompt_cases


def run_baseline_sweep(
    args: Namespace, prompt_cases: list[PromptCase]
) -> dict[int, RunResult]:
    baseline_results = {}
    llm: LLM | None = None
    try:
        llm = create_llm(args, speculative_config=None)
        for case in prompt_cases:
            baseline_results[case.prompt_len] = run_generation(
                f"baseline prompt_len={case.prompt_len}",
                llm,
                args,
                case.prompts,
            )
    finally:
        cleanup_llm(llm)
    return baseline_results


def run_window_sweep(
    args: Namespace,
    prompt_cases: list[PromptCase],
    baseline_results: dict[int, RunResult],
    window_size: int,
) -> list[SweepResult]:
    results = []
    self_swa_config = {
        "method": "self_swa",
        "num_speculative_tokens": args.num_spec_tokens,
        "self_swa_window_size": window_size,
        "self_swa_sink_size": args.self_swa_sink_size,
    }
    llm: LLM | None = None
    try:
        llm = create_llm(args, self_swa_config)
        for case in prompt_cases:
            is_eligible, reason = check_window_eligible(
                case.prompt_len, window_size, args.self_swa_sink_size
            )
            if not is_eligible:
                print(
                    f"Skipping window_size={window_size} for "
                    f"prompt_len={case.prompt_len}: {reason}."
                )
                continue

            try:
                before_metrics = collect_spec_metrics(
                    llm.get_metrics(), args.num_spec_tokens
                )
                self_swa = run_generation(
                    f"self-SWA prompt_len={case.prompt_len} "
                    f"window_size={window_size}",
                    llm,
                    args,
                    case.prompts,
                )
                after_metrics = collect_spec_metrics(
                    self_swa.metrics, args.num_spec_tokens
                )
                spec_metrics = diff_spec_metrics(before_metrics, after_metrics)
                baseline = baseline_results[case.prompt_len]
                speedup = print_speed_comparison(baseline, self_swa)
                exact_match = compare_outputs(
                    baseline,
                    self_swa,
                    raise_on_mismatch=not args.continue_on_error,
                )
                print_spec_metrics(spec_metrics)
                results.append(
                    SweepResult(
                        prompt_len=case.prompt_len,
                        actual_prompt_lens=case.actual_prompt_lens,
                        max_model_len=args.max_model_len,
                        self_swa_window_size=window_size,
                        exact_match=exact_match,
                        baseline_decode_tokens_per_s=baseline.decode_tokens_per_s,
                        self_swa_decode_tokens_per_s=self_swa.decode_tokens_per_s,
                        speedup=speedup,
                        spec_metrics=spec_metrics,
                    )
                )
                if args.print_output:
                    print("\n=== generated text ===")
                    for i, text in enumerate(self_swa.output_texts):
                        print(f"[{i}] {text}")
            except Exception as exc:
                print(
                    f"self-SWA case failed for prompt_len={case.prompt_len}, "
                    f"window_size={window_size}: {exc!r}"
                )
                if not args.continue_on_error:
                    raise
    finally:
        cleanup_llm(llm)
    return results


def run_sweep(
    args: Namespace,
    tokenizer: AutoTokenizer,
    prompt_lens: list[int],
    window_sizes: list[int],
) -> list[SweepResult]:
    sweep_args = make_sweep_args(args, prompt_lens)
    print(f"max_model_len: {sweep_args.max_model_len}")
    prompt_cases = build_prompt_cases(sweep_args, tokenizer, prompt_lens)
    baseline_results = run_baseline_sweep(sweep_args, prompt_cases)

    results = []
    for window_size in window_sizes:
        print("\n" + "#" * 80)
        print(f"self_swa_window_size: {window_size}")
        print("#" * 80)
        results.extend(
            run_window_sweep(
                sweep_args,
                prompt_cases,
                baseline_results,
                window_size,
            )
        )
    return results


def print_sweep_summary(results: list[SweepResult]) -> None:
    if not results:
        print("\nNo self-SWA sweep cases completed.")
        return

    print("\n=== sweep summary ===")
    print(
        "prompt_len\tmax_model_len\twindow_size\texact_match\t"
        "baseline_decode_tps\tself_swa_decode_tps\tspeedup\t"
        "num_drafts\tmean_acceptance_length"
    )
    for result in results:
        print(
            f"{result.prompt_len}\t"
            f"{result.max_model_len}\t"
            f"{result.self_swa_window_size}\t"
            f"{result.exact_match}\t"
            f"{result.baseline_decode_tokens_per_s:.2f}\t"
            f"{result.self_swa_decode_tokens_per_s:.2f}\t"
            f"{result.speedup:.2f}\t"
            f"{result.spec_metrics.num_drafts}\t"
            f"{result.spec_metrics.mean_acceptance_length:.2f}"
        )


def main() -> None:
    args = parse_args()
    validate_env()
    prompt_lens = resolve_prompt_lens(args)
    window_sizes = resolve_window_sizes(args)
    print(f"model: {args.model}")
    print(f"prompt_lens: {prompt_lens}")
    print(f"self_swa_window_sizes: {window_sizes}")
    print(f"tensor_parallel_size: {args.tp}")
    print(f"output_len: {args.output_len}")
    print(f"num_prompts: {args.num_prompts}")
    print(f"num_spec_tokens: {args.num_spec_tokens}")
    print(
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN="
        f"{os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']}"
    )
    print(f"VLLM_ROCM_USE_AITER={os.environ['VLLM_ROCM_USE_AITER']}")
    print(
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT="
        f"{os.environ['VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    results = run_sweep(args, tokenizer, prompt_lens, window_sizes)
    print_sweep_summary(results)


if __name__ == "__main__":
    main()
