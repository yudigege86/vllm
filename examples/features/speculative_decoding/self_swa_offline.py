# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep self-SWA speculative decoding on GLM-4-9B-Chat-1M.

This script creates max-context engines for the sweep: one greedy baseline
engine, then one greedy self-SWA engine per sliding-window size. Each engine runs
the requested prompt lengths, comparing generated token IDs and printing speed
and per-case self-SWA acceptance metrics. The self-SWA path currently requires
the ROCm AITER attention backend; baseline-only runs can use other backends.

Prompt modes (``--prompt-mode``):

* ``pg19`` (default): tile real long-context PG-19 book text up to the
  requested prompt length and instruct the chat-tuned model to *continue*
  the passage. The last few sentences of the prompt body are echoed
  alongside the generation so seam quality can be inspected by eye. The
  corpus is loaded from ``--pg19-dataset`` (default ``emozilla/pg19``,
  a parquet mirror) since modern ``datasets`` no longer runs the
  ``deepmind/pg19`` script loader.
* ``niah``: build a haystack from PG-19, splice in a deterministic
  passcode needle at ``--niah-depth-frac`` (default ``0.5``), and ask the
  model to recall the passcode. Pass/fail per prompt is computed by
  substring match against the planted code; sweep ``--niah-depth-fracs``
  for a depth profile.
* ``repeat``: the legacy short-block repetition prompt, kept for
  reproducing pre-overhaul numbers.

When the tokenizer exposes a chat template (e.g. GLM-4-9B-Chat-1M) the
script wraps the prompt with it automatically; use ``--no-chat-template``
to feed raw token IDs and reproduce the pre-overhaul behaviour
(``--prompt-mode repeat --no-chat-template``).
"""

import gc
import json
import os
import random
import re
import time
from argparse import BooleanOptionalAction, Namespace
from contextlib import suppress
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These environment variables must be set before importing vLLM.
os.environ.setdefault("VLLM_CACHE_ROOT", "/tmp/vllm-cache")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
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
DEFAULT_PROMPT_MODE = "pg19"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_NIAH_DEPTH_FRAC = 0.5
DEFAULT_PG19_SPLIT = "test"
DEFAULT_PG19_DATASET = "emozilla/pg19"
DEFAULT_PROMPT_TAIL_SENTENCES = 3
DEFAULT_PG19_CACHE_DIR = os.path.join(
    os.environ.get("VLLM_CACHE_ROOT", "/tmp/vllm-cache"), "pg19"
)
NIAH_RNG_SEED = 0xC0FFEE
CHAT_BODY_PLACEHOLDER = "[VLLM_BODY_PLACEHOLDER]"


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
    niah_passes: list[bool | None] | None = None


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
    num_spec_tokens: int
    self_swa_window_size: int
    exact_match: bool | None
    baseline_decode_tokens_per_s: float | None
    self_swa_decode_tokens_per_s: float
    speedup: float | None
    spec_metrics: SpecMetrics
    depth_frac: float | None = None
    baseline_niah_pass_rate: float | None = None
    self_swa_niah_pass_rate: float | None = None


@dataclass
class PromptCase:
    prompt_len: int
    actual_prompt_lens: list[int]
    prompts: list[dict[str, list[int]]]
    expected_answers: list[str | None]
    prompt_tail_texts: list[str | None]
    depth_frac: float | None = None


@dataclass
class BuiltPrompt:
    """A single tokenized prompt plus optional metadata used downstream.

    Attributes:
        token_ids: The full prompt token ID sequence handed to the engine.
        expected_answer: Substring the model must emit to count as a NIAH pass.
            ``None`` for prompt modes without an automated retrieval check.
        prompt_tail_text: Decoded tail of the prompt body (last few sentences),
            printed alongside the generation for manual inspection. ``None``
            when not requested.
        depth_frac: NIAH needle depth recorded so it can be carried through to
            sweep results.
    """

    token_ids: list[int]
    expected_answer: str | None = None
    prompt_tail_text: str | None = None
    depth_frac: float | None = None


def _env_is_true(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def validate_env(args: Namespace) -> None:
    uses_aiter_fa = args.attention_backend == "ROCM_AITER_FA"
    baseline_only = getattr(args, "baseline_only", False)
    self_swa_only = getattr(args, "self_swa_only", False)
    if baseline_only and self_swa_only:
        raise RuntimeError(
            "--baseline-only and --self-swa-only are mutually exclusive."
        )
    if uses_aiter_fa and not _env_is_true(os.environ.get("VLLM_ROCM_USE_AITER", "0")):
        raise RuntimeError(
            "ROCM_AITER_FA requires VLLM_ROCM_USE_AITER=1. Use "
            "--attention-backend ROCM_ATTN for a non-AITER baseline run."
        )
    if not baseline_only and not uses_aiter_fa:
        raise RuntimeError(
            "self-SWA currently requires --attention-backend ROCM_AITER_FA. "
            "Use --baseline-only when testing a non-AITER backend."
        )
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
    parser.add_argument(
        "--num-spec-tokens-list",
        default=None,
        help=(
            "Optional comma-separated num_spec_tokens values to sweep while "
            "reusing the same baseline, e.g. 4,8,16,32,64."
        ),
    )
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
        "--hf-overrides",
        type=json.loads,
        default={},
        help="JSON or dotted Hugging Face config overrides passed to LLM.",
    )
    parser.add_argument(
        "--trust-remote-code", action=BooleanOptionalAction, default=True
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Directory for torch profiler traces. Profiles each generate() call.",
    )
    parser.add_argument("--profile-delay-iterations", type=int, default=0)
    parser.add_argument("--profile-max-iterations", type=int, default=0)
    parser.add_argument(
        "--profile-with-stack", action=BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--profile-record-shapes", action=BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--profile-explicit",
        action="store_true",
        help=(
            "Use explicit llm.start_profile()/stop_profile() around selected "
            "generate() calls. This enables warmup before profiling."
        ),
    )
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=0,
        help="Number of unprofiled warmup generate() calls before profiled calls.",
    )
    parser.add_argument(
        "--profile-separate-dirs",
        action="store_true",
        help=(
            "When profiling, write each baseline/self-SWA case to its own "
            "subdirectory under --profile-dir."
        ),
    )
    parser.add_argument(
        "--profile-split-prefill-decode",
        action="store_true",
        help=(
            "For each profiled case, first profile a max_tokens=1 prefill-heavy "
            "generate(), then profile the normal output length for decode analysis. "
            "The decode profile reuses the same prompt so prefix caching can reuse "
            "the prompt KV populated by the prefill probe."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--prompt-mode",
        choices=["repeat", "pg19", "niah"],
        default=DEFAULT_PROMPT_MODE,
        help=(
            "Prompt source: 'repeat' is the legacy short-block repetition, "
            "'pg19' tiles real long-context PG-19 text and asks the model "
            "to continue it, 'niah' inserts a deterministic passcode needle "
            "into a PG-19 haystack and asks the model to retrieve it."
        ),
    )
    parser.add_argument(
        "--chat-template",
        action=BooleanOptionalAction,
        default=None,
        help=(
            "Whether to wrap the prompt in the tokenizer's chat template. "
            "Default (unset) auto-enables it when the tokenizer has one. "
            "Pass --no-chat-template to feed raw token IDs."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System message used when the chat template is active.",
    )
    parser.add_argument(
        "--niah-depth-frac",
        type=float,
        default=DEFAULT_NIAH_DEPTH_FRAC,
        help=(
            "Fractional depth in (0, 1) at which to place the NIAH needle "
            "within the haystack body. Used only when --prompt-mode=niah and "
            "--niah-depth-fracs is unset."
        ),
    )
    parser.add_argument(
        "--niah-depth-fracs",
        default=None,
        help=(
            "Optional comma-separated list of NIAH depth fractions to sweep, "
            "e.g. '0.05,0.5,0.95'. When set, each prompt length is expanded "
            "into one prompt case per depth (multiplies total case count)."
        ),
    )
    parser.add_argument(
        "--pg19-dataset",
        default=DEFAULT_PG19_DATASET,
        help=(
            "HuggingFace dataset repo to load for the PG-19 corpus. Default "
            "is the parquet mirror `emozilla/pg19`; the script loader for "
            "`deepmind/pg19` is incompatible with modern `datasets`."
        ),
    )
    parser.add_argument(
        "--pg19-split",
        choices=["train", "validation", "test"],
        default=DEFAULT_PG19_SPLIT,
        help="PG-19 dataset split used as the prompt corpus.",
    )
    parser.add_argument(
        "--pg19-cache-dir",
        default=DEFAULT_PG19_CACHE_DIR,
        help=(
            "Directory used to cache the tokenized PG-19 blob. Defaults to "
            "$VLLM_CACHE_ROOT/pg19."
        ),
    )
    parser.add_argument(
        "--prompt-tail-sentences",
        type=int,
        default=DEFAULT_PROMPT_TAIL_SENTENCES,
        help=(
            "Number of sentences from the end of the prompt body to echo "
            "before the generated text in --prompt-mode=pg19. Lets you eyeball "
            "the seam between prompt and continuation."
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the greedy baseline, useful for backend crash isolation.",
    )
    parser.add_argument(
        "--self-swa-only",
        action="store_true",
        help=(
            "Run only self-SWA cases, useful for profiling self-SWA in a "
            "separate process without producing an extra baseline trace."
        ),
    )
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


def resolve_num_spec_tokens(args: Namespace) -> list[int]:
    if args.num_spec_tokens_list is None:
        if args.num_spec_tokens <= 0:
            raise ValueError("--num-spec-tokens must be positive")
        return [args.num_spec_tokens]
    return _parse_int_list(args.num_spec_tokens_list, "--num-spec-tokens-list")


def resolve_depth_fracs(args: Namespace) -> list[float | None]:
    """Return the list of depth fractions to iterate.

    For non-NIAH modes this is a single-element list of ``None`` so each
    prompt length produces exactly one prompt case. NIAH mode either uses the
    --niah-depth-fracs sweep list or falls back to --niah-depth-frac.
    """
    if args.prompt_mode != "niah":
        return [None]

    if args.niah_depth_fracs is not None:
        depths: list[float | None] = []
        for raw in args.niah_depth_fracs.split(","):
            value = float(raw.strip())
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"--niah-depth-fracs values must be in (0, 1); got {value}"
                )
            depths.append(value)
        if not depths:
            raise ValueError("--niah-depth-fracs must contain at least one value")
        return depths

    if not 0.0 < args.niah_depth_frac < 1.0:
        raise ValueError(
            f"--niah-depth-frac must be in (0, 1); got {args.niah_depth_frac}"
        )
    return [args.niah_depth_frac]


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


def make_sweep_args(
    args: Namespace, prompt_lens: list[int], num_spec_tokens_values: list[int]
) -> Namespace:
    sweep_args = copy(args)
    sweep_args.num_spec_tokens = max(num_spec_tokens_values)
    sweep_args.max_model_len = args.max_model_len or max(
        auto_max_model_len(sweep_args, prompt_len) for prompt_len in prompt_lens
    )
    return sweep_args


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Return the start index of ``needle`` inside ``haystack`` or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "tokenizer"


def _use_chat_template(args: Namespace, tokenizer: AutoTokenizer) -> bool:
    if args.chat_template is True:
        if getattr(tokenizer, "chat_template", None) is None:
            raise RuntimeError(
                "--chat-template was forced on but the tokenizer has no "
                "chat_template. Re-run with --no-chat-template or pick a "
                "chat-tuned model."
            )
        return True
    if args.chat_template is False:
        return False
    return getattr(tokenizer, "chat_template", None) is not None


def _chat_overhead(
    tokenizer: AutoTokenizer,
    args: Namespace,
    user_prefix: str,
    user_suffix: str,
) -> tuple[list[int], list[int]]:
    """Return ``(prefix_ids, suffix_ids)`` framing the body for this mode.

    When the chat template is active, the wrapper renders the template to
    text with a placeholder in place of the body, splits on the placeholder
    to recover the exact prefix/suffix strings that frame the body, and
    tokenizes each side independently. Working in text space sidesteps BPE
    seam merges that would otherwise prevent locating the placeholder in a
    tokenized blob. When the chat template is off, the raw prefix/suffix
    text is tokenized directly (with BOS only on the prefix).
    """
    if not _use_chat_template(args, tokenizer):
        prefix_ids = list(tokenizer.encode(user_prefix, add_special_tokens=True))
        suffix_ids = list(tokenizer.encode(user_suffix, add_special_tokens=False))
        return prefix_ids, suffix_ids

    placeholder = CHAT_BODY_PLACEHOLDER
    messages = [
        {"role": "system", "content": args.system_prompt},
        {
            "role": "user",
            "content": f"{user_prefix}{placeholder}{user_suffix}",
        },
    ]
    templated_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if placeholder not in templated_text:
        raise RuntimeError(
            "Chat template stripped the body placeholder. "
            "Pass --no-chat-template to bypass templating."
        )
    prefix_text, suffix_text = templated_text.split(placeholder, 1)
    # The chat template already embeds any required special tokens as
    # literal text (e.g. <|im_start|>, [gMASK]<sop>), so each side is
    # tokenized with add_special_tokens=False to avoid double-adding BOS.
    prefix_ids = list(tokenizer.encode(prefix_text, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(suffix_text, add_special_tokens=False))
    return prefix_ids, suffix_ids


def _check_prompt_len(prompt_len: int, max_prompt_len: int) -> None:
    if prompt_len > max_prompt_len:
        raise ValueError(
            f"--prompt-len={prompt_len} leaves no room for generation under "
            f"--max-model-len; max allowed prompt length is {max_prompt_len}."
        )


def _check_body_budget(prompt_len: int, body_budget: int) -> None:
    if body_budget <= 0:
        raise ValueError(
            f"--prompt-len={prompt_len} is too short for the prompt "
            f"prefix/suffix wrapper ({prompt_len - body_budget} tokens of "
            "overhead)."
        )


_PG19_CORPUS_CACHE: dict[tuple[str, str, int], list[int]] = {}


def _get_pg19_corpus(
    tokenizer: AutoTokenizer, args: Namespace, min_len_needed: int
) -> list[int]:
    """Return a tokenized PG-19 blob of at least ``min_len_needed`` tokens.

    The blob is cached both in-process (across prompts in a sweep) and on
    disk (across script invocations) so the multi-million-token tokenize
    pass only ever runs once per (tokenizer, split, length) key.
    """
    key = (
        f"{args.pg19_dataset}:{tokenizer.name_or_path}",
        args.pg19_split,
        min_len_needed,
    )
    cached = _PG19_CORPUS_CACHE.get(key)
    if cached is not None:
        return cached

    cache_dir = Path(args.pg19_cache_dir)
    cache_path = cache_dir / (
        f"{_safe_name(args.pg19_dataset)}_"
        f"{_safe_name(tokenizer.name_or_path)}_{args.pg19_split}_"
        f"{min_len_needed}.pt"
    )
    if cache_path.exists():
        ids = torch.load(cache_path).tolist()
        _PG19_CORPUS_CACHE[key] = ids
        return ids

    try:
        from datasets import load_dataset  # noqa: F401 (lazy import)
    except ImportError as exc:
        raise RuntimeError(
            "--prompt-mode pg19/niah requires the `datasets` package. "
            "Install it with `uv pip install datasets`."
        ) from exc
    from datasets import load_dataset

    print(
        f"Loading {args.pg19_dataset} split={args.pg19_split} and tokenizing "
        f"up to {min_len_needed} tokens (one-time; cached to {cache_path})."
    )
    ds = load_dataset(args.pg19_dataset, split=args.pg19_split, streaming=True)
    ids: list[int] = []
    for row in ds:
        text = row.get("text") or ""
        if not text:
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) >= min_len_needed:
            break
    if len(ids) < min_len_needed:
        raise RuntimeError(
            f"PG-19 split={args.pg19_split} exhausted at {len(ids)} tokens; "
            f"need at least {min_len_needed}. Try --pg19-split train."
        )
    ids = ids[:min_len_needed]
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(ids, dtype=torch.int32), cache_path)
    _PG19_CORPUS_CACHE[key] = ids
    return ids


def _pg19_body_offset(
    corpus_len: int, body_budget: int, prompt_index: int
) -> int:
    """Pick a deterministic, non-overlapping offset for the per-prompt slice."""
    span = max(1, corpus_len - body_budget)
    return (prompt_index * body_budget) % span


def _extract_prompt_tail(
    tokenizer: AutoTokenizer, body_token_ids: list[int], num_sentences: int
) -> str:
    """Decode the body's tail and return its last ``num_sentences`` sentences."""
    if num_sentences <= 0 or not body_token_ids:
        return ""
    tail_count = min(512, len(body_token_ids))
    tail_text = tokenizer.decode(
        body_token_ids[-tail_count:], skip_special_tokens=True
    )
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", tail_text)
        if s.strip()
    ]
    if not sentences:
        return tail_text.strip()
    return " ".join(sentences[-num_sentences:])


def _build_repeat_prompt(
    tokenizer: AutoTokenizer,
    args: Namespace,
    prompt_len: int,
    max_prompt_len: int,
    prompt_index: int,
) -> BuiltPrompt:
    _check_prompt_len(prompt_len, max_prompt_len)

    user_prefix = f"Self-SWA validation passage {prompt_index}.\n"
    repeat_block = (
        "This deterministic validation text is repeated to create a long "
        "prompt while keeping greedy decoding stable. "
    )
    user_suffix = "\nSummarize the validation passage in one short sentence."

    repeat_ids = tokenizer.encode(repeat_block, add_special_tokens=False)
    if not repeat_ids:
        raise ValueError("repeat block produced no tokens")

    prefix_ids, suffix_ids = _chat_overhead(
        tokenizer, args, user_prefix, user_suffix
    )
    body_budget = prompt_len - len(prefix_ids) - len(suffix_ids)
    _check_body_budget(prompt_len, body_budget)

    full_repeats, remainder = divmod(body_budget, len(repeat_ids))
    body = repeat_ids * full_repeats + repeat_ids[:remainder]
    token_ids = prefix_ids + body + suffix_ids
    return BuiltPrompt(token_ids=token_ids)


def _build_pg19_prompt(
    tokenizer: AutoTokenizer,
    args: Namespace,
    prompt_len: int,
    max_prompt_len: int,
    prompt_index: int,
) -> BuiltPrompt:
    _check_prompt_len(prompt_len, max_prompt_len)

    user_prefix = (
        "Please continue the following text in the same style. "
        "Do not summarize; continue narrating from where it stops.\n\n"
        "<text>\n"
    )
    user_suffix = "\n</text>\n\nContinuation:"

    prefix_ids, suffix_ids = _chat_overhead(
        tokenizer, args, user_prefix, user_suffix
    )
    body_budget = prompt_len - len(prefix_ids) - len(suffix_ids)
    _check_body_budget(prompt_len, body_budget)

    needed = max(body_budget * (args.num_prompts + 1), body_budget + 1)
    corpus = _get_pg19_corpus(tokenizer, args, needed)
    offset = _pg19_body_offset(len(corpus), body_budget, prompt_index)
    body = list(corpus[offset : offset + body_budget])

    token_ids = prefix_ids + body + suffix_ids
    prompt_tail_text = _extract_prompt_tail(
        tokenizer, body, args.prompt_tail_sentences
    )
    return BuiltPrompt(token_ids=token_ids, prompt_tail_text=prompt_tail_text)


def _build_niah_prompt(
    tokenizer: AutoTokenizer,
    args: Namespace,
    prompt_len: int,
    max_prompt_len: int,
    prompt_index: int,
    depth_frac: float | None,
) -> BuiltPrompt:
    _check_prompt_len(prompt_len, max_prompt_len)

    if depth_frac is None:
        depth_frac = args.niah_depth_frac
    if not 0.0 < depth_frac < 1.0:
        raise ValueError(
            f"NIAH depth_frac must be in (0, 1); got {depth_frac}"
        )

    rng = random.Random(NIAH_RNG_SEED ^ prompt_index)
    code = (
        f"PC-{rng.randrange(10**6):06d}-"
        + "".join(rng.choices("ABCDEFGHJKMNPQRSTUVWXYZ", k=4))
    )
    needle_text = (
        f"\n\nIMPORTANT: The magic passcode for participant {prompt_index} "
        f"is {code}. Remember this exact passcode.\n\n"
    )
    user_prefix = (
        "Here is a long document. Read it carefully and answer the "
        "question at the end.\n\n<document>\n"
    )
    user_suffix = (
        f"\n</document>\n\nQuestion: What is the magic passcode for "
        f"participant {prompt_index}? Reply with just the passcode.\nAnswer:"
    )

    prefix_ids, suffix_ids = _chat_overhead(
        tokenizer, args, user_prefix, user_suffix
    )
    body_budget = prompt_len - len(prefix_ids) - len(suffix_ids)
    _check_body_budget(prompt_len, body_budget)

    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    if len(needle_ids) >= body_budget:
        raise ValueError(
            f"NIAH needle ({len(needle_ids)} tokens) does not fit inside the "
            f"haystack body ({body_budget} tokens). Use a larger --prompt-len."
        )

    needed = max(body_budget * (args.num_prompts + 1), body_budget + 1)
    corpus = _get_pg19_corpus(tokenizer, args, needed)
    offset = _pg19_body_offset(len(corpus), body_budget, prompt_index)
    haystack = list(corpus[offset : offset + body_budget - len(needle_ids)])

    pos = int(depth_frac * body_budget)
    pos = max(0, min(len(haystack), pos))
    body = haystack[:pos] + needle_ids + haystack[pos:]
    assert len(body) == body_budget, (len(body), body_budget)

    token_ids = prefix_ids + body + suffix_ids
    return BuiltPrompt(
        token_ids=token_ids,
        expected_answer=code,
        depth_frac=depth_frac,
    )


def build_prompt(
    tokenizer: AutoTokenizer,
    args: Namespace,
    prompt_len: int,
    max_prompt_len: int,
    prompt_index: int,
    depth_frac: float | None = None,
) -> BuiltPrompt:
    """Dispatch to the prompt builder selected by ``args.prompt_mode``."""
    mode = args.prompt_mode
    if mode == "repeat":
        return _build_repeat_prompt(
            tokenizer, args, prompt_len, max_prompt_len, prompt_index
        )
    if mode == "pg19":
        return _build_pg19_prompt(
            tokenizer, args, prompt_len, max_prompt_len, prompt_index
        )
    if mode == "niah":
        return _build_niah_prompt(
            tokenizer,
            args,
            prompt_len,
            max_prompt_len,
            prompt_index,
            depth_frac,
        )
    raise ValueError(f"Unknown --prompt-mode: {mode!r}")


def build_prompts(
    tokenizer: AutoTokenizer,
    args: Namespace,
    prompt_len: int,
    depth_frac: float | None,
) -> tuple[list[dict[str, list[int]]], list[str | None], list[str | None]]:
    """Build ``args.num_prompts`` prompts for a single (prompt_len, depth_frac) cell."""
    max_prompt_len = (
        args.max_model_len - args.output_len - args.num_spec_tokens - 1
    )
    prompts: list[dict[str, list[int]]] = []
    expected: list[str | None] = []
    tails: list[str | None] = []
    for i in range(args.num_prompts):
        built = build_prompt(
            tokenizer,
            args,
            prompt_len,
            max_prompt_len,
            prompt_index=i,
            depth_frac=depth_frac,
        )
        prompts.append({"prompt_token_ids": built.token_ids})
        expected.append(built.expected_answer)
        tails.append(built.prompt_tail_text)
    return prompts, expected, tails


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
    if args.hf_overrides:
        kwargs["hf_overrides"] = args.hf_overrides
    if args.profile_dir is not None:
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": args.profile_dir,
            "torch_profiler_with_stack": args.profile_with_stack,
            "torch_profiler_with_memory": False,
            "torch_profiler_record_shapes": args.profile_record_shapes,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": True,
            "torch_profiler_dump_cuda_time_total": True,
            "ignore_frontend": True,
            "delay_iterations": args.profile_delay_iterations,
            "max_iterations": args.profile_max_iterations,
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


def _profile_prefix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "profile"


def _with_profile_dir(args: Namespace, profile_dir: str | None) -> Namespace:
    if args.profile_dir == profile_dir:
        return args
    updated = copy(args)
    updated.profile_dir = profile_dir
    return updated


def _generate_once(
    llm: LLM,
    prompts: list[dict[str, list[int]]],
    sampling_params: SamplingParams,
):
    return llm.generate(prompts, sampling_params, use_tqdm=True)


def _profiled_generate_once(
    name: str,
    llm: LLM,
    args: Namespace,
    prompts: list[dict[str, list[int]]],
    sampling_params: SamplingParams,
):
    if args.profile_dir is None:
        return _generate_once(llm, prompts, sampling_params)

    print(
        f"Starting torch profiler: dir={args.profile_dir} "
        f"prefix={_profile_prefix(name)}"
    )
    llm.start_profile(profile_prefix=_profile_prefix(name))
    try:
        return _generate_once(llm, prompts, sampling_params)
    finally:
        llm.stop_profile()
        print(f"Stopped torch profiler: dir={args.profile_dir}")


def _run_warmup_generations(
    llm: LLM,
    args: Namespace,
    prompts: list[dict[str, list[int]]],
    sampling_params: SamplingParams,
) -> None:
    if args.profile_warmup_runs <= 0:
        return
    for warmup_idx in range(args.profile_warmup_runs):
        print(
            f"Warmup generate {warmup_idx + 1}/{args.profile_warmup_runs} "
            f"(max_tokens={sampling_params.max_tokens})"
        )
        _generate_once(llm, prompts, sampling_params)


def _run_profile_prefill_probe(
    name: str,
    llm: LLM,
    args: Namespace,
    prompts: list[dict[str, list[int]]],
) -> None:
    if args.profile_dir is None or not args.profile_split_prefill_decode:
        return
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    print(
        "Profiling max_tokens=1 prefill probe. The following decode profile "
        "uses the same prompt so prefix caching can reuse prompt KV."
    )
    _profiled_generate_once(f"{name} prefill", llm, args, prompts, sampling_params)


def _compute_niah_passes(
    expected_answers: list[str | None], texts: list[str]
) -> list[bool | None] | None:
    """Return per-prompt pass/fail for NIAH, or ``None`` outside NIAH mode."""
    if not any(exp is not None for exp in expected_answers):
        return None
    passes: list[bool | None] = []
    for exp, text in zip(expected_answers, texts):
        if exp is None:
            passes.append(None)
        else:
            passes.append(exp.lower() in text.lower())
    return passes


def _niah_pass_rate(passes: list[bool | None] | None) -> float | None:
    if not passes:
        return None
    scored = [p for p in passes if p is not None]
    if not scored:
        return None
    return sum(1 for p in scored if p) / len(scored)


def run_generation(
    name: str,
    llm: LLM,
    args: Namespace,
    case: PromptCase,
) -> RunResult:
    print(f"\n=== {name} ===")
    prompts = case.prompts
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
    )
    if args.profile_explicit:
        _run_profile_prefill_probe(name, llm, args, prompts)
        _run_warmup_generations(llm, args, prompts, sampling_params)

    start = time.perf_counter()
    if args.profile_dir is None:
        outputs = _generate_once(llm, prompts, sampling_params)
    else:
        outputs = _profiled_generate_once(name, llm, args, prompts, sampling_params)
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
    print("\n=== generated text ===")
    for i, text in enumerate(texts):
        tail = case.prompt_tail_texts[i] if i < len(case.prompt_tail_texts) else None
        if tail:
            print(f"[{i}] prompt tail: {tail}")
        print(f"[{i}] {text}")

    niah_passes = _compute_niah_passes(case.expected_answers, texts)
    if niah_passes is not None:
        print("\n=== NIAH retrieval ===")
        for i, (exp, text, passed) in enumerate(
            zip(case.expected_answers, texts, niah_passes)
        ):
            if exp is None:
                continue
            status = "PASS" if passed else "FAIL"
            print(f"[{i}] expected: {exp} | result: {status}")
        rate = _niah_pass_rate(niah_passes)
        if rate is not None:
            scored = sum(1 for p in niah_passes if p is not None)
            hits = sum(1 for p in niah_passes if p)
            print(f"niah_pass_rate: {rate:.2f} ({hits}/{scored})")

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
        niah_passes=niah_passes,
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
    args: Namespace,
    tokenizer: AutoTokenizer,
    prompt_lens: list[int],
    depth_fracs: list[float | None],
) -> list[PromptCase]:
    prompt_cases = []
    for prompt_len in prompt_lens:
        for depth_frac in depth_fracs:
            prompts, expected, tails = build_prompts(
                tokenizer, args, prompt_len, depth_frac
            )
            actual_prompt_lens = [
                len(prompt["prompt_token_ids"]) for prompt in prompts
            ]
            prompt_cases.append(
                PromptCase(
                    prompt_len=prompt_len,
                    actual_prompt_lens=actual_prompt_lens,
                    prompts=prompts,
                    expected_answers=expected,
                    prompt_tail_texts=tails,
                    depth_frac=depth_frac,
                )
            )
            print("\n" + "=" * 80)
            print(f"prompt_len: {prompt_len}")
            print(f"actual_prompt_lens: {actual_prompt_lens}")
            print(f"max_model_len: {args.max_model_len}")
            if depth_frac is not None:
                print(f"niah_depth_frac: {depth_frac}")
            print("=" * 80)
    return prompt_cases


def _case_name(prefix: str, case: PromptCase) -> str:
    parts = [f"{prefix} prompt_len={case.prompt_len}"]
    if case.depth_frac is not None:
        parts.append(f"depth_frac={case.depth_frac:.2f}")
    return " ".join(parts)


def run_baseline_sweep(
    args: Namespace, prompt_cases: list[PromptCase]
) -> dict[tuple[int, float | None], RunResult]:
    baseline_results: dict[tuple[int, float | None], RunResult] = {}
    llm: LLM | None = None
    try:
        baseline_args = args
        if args.profile_dir is not None and args.profile_separate_dirs:
            baseline_args = _with_profile_dir(
                args, str(Path(args.profile_dir) / "baseline")
            )
        llm = create_llm(baseline_args, speculative_config=None)
        for case in prompt_cases:
            baseline_results[(case.prompt_len, case.depth_frac)] = run_generation(
                _case_name("baseline", case),
                llm,
                baseline_args,
                case,
            )
    finally:
        cleanup_llm(llm)
    return baseline_results


def run_window_sweep(
    args: Namespace,
    prompt_cases: list[PromptCase],
    baseline_results: dict[tuple[int, float | None], RunResult],
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
        llm_args = args
        if args.profile_dir is not None and args.profile_separate_dirs:
            llm_args = _with_profile_dir(
                args, str(Path(args.profile_dir) / f"self_swa_window_{window_size}")
            )
        llm = create_llm(llm_args, self_swa_config)
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
                    _case_name(
                        f"self-SWA window_size={window_size}", case
                    ),
                    llm,
                    llm_args,
                    case,
                )
                after_metrics = collect_spec_metrics(
                    self_swa.metrics, args.num_spec_tokens
                )
                spec_metrics = diff_spec_metrics(before_metrics, after_metrics)
                baseline = baseline_results.get(
                    (case.prompt_len, case.depth_frac)
                )
                if baseline is None:
                    print(
                        "\nSkipping baseline comparison because "
                        "--self-swa-only was set."
                    )
                    speedup = None
                    exact_match = None
                    baseline_decode_tokens_per_s = None
                    baseline_niah_pass_rate = None
                else:
                    speedup = print_speed_comparison(baseline, self_swa)
                    exact_match = compare_outputs(
                        baseline,
                        self_swa,
                        raise_on_mismatch=not args.continue_on_error,
                    )
                    baseline_decode_tokens_per_s = baseline.decode_tokens_per_s
                    baseline_niah_pass_rate = _niah_pass_rate(
                        baseline.niah_passes
                    )
                self_swa_niah_pass_rate = _niah_pass_rate(self_swa.niah_passes)
                print_spec_metrics(spec_metrics)
                results.append(
                    SweepResult(
                        prompt_len=case.prompt_len,
                        actual_prompt_lens=case.actual_prompt_lens,
                        max_model_len=args.max_model_len,
                        num_spec_tokens=args.num_spec_tokens,
                        self_swa_window_size=window_size,
                        exact_match=exact_match,
                        baseline_decode_tokens_per_s=baseline_decode_tokens_per_s,
                        self_swa_decode_tokens_per_s=self_swa.decode_tokens_per_s,
                        speedup=speedup,
                        spec_metrics=spec_metrics,
                        depth_frac=case.depth_frac,
                        baseline_niah_pass_rate=baseline_niah_pass_rate,
                        self_swa_niah_pass_rate=self_swa_niah_pass_rate,
                    )
                )
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
    num_spec_tokens_values: list[int],
    depth_fracs: list[float | None],
) -> list[SweepResult]:
    sweep_args = make_sweep_args(args, prompt_lens, num_spec_tokens_values)
    print(f"max_model_len: {sweep_args.max_model_len}")
    prompt_cases = build_prompt_cases(
        sweep_args, tokenizer, prompt_lens, depth_fracs
    )
    baseline_results: dict[tuple[int, float | None], RunResult] = {}
    if not args.self_swa_only:
        baseline_results = run_baseline_sweep(sweep_args, prompt_cases)
    if args.baseline_only:
        return []

    results = []
    for num_spec_tokens in num_spec_tokens_values:
        spec_args = copy(sweep_args)
        spec_args.num_spec_tokens = num_spec_tokens
        for window_size in window_sizes:
            print("\n" + "#" * 80)
            print(f"num_spec_tokens: {num_spec_tokens}")
            print(f"self_swa_window_size: {window_size}")
            print("#" * 80)
            results.extend(
                run_window_sweep(
                    spec_args,
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
        "prompt_len\tmax_model_len\tnum_spec_tokens\twindow_size\tdepth_frac\t"
        "exact_match\tbaseline_decode_tps\tself_swa_decode_tps\t"
        "speedup\tnum_drafts\tmean_acceptance_length\t"
        "baseline_niah_pass\tself_swa_niah_pass"
    )
    for result in results:
        exact_match = "n/a" if result.exact_match is None else str(result.exact_match)
        baseline_tps = (
            "n/a"
            if result.baseline_decode_tokens_per_s is None
            else f"{result.baseline_decode_tokens_per_s:.2f}"
        )
        speedup = "n/a" if result.speedup is None else f"{result.speedup:.2f}"
        depth = (
            "n/a" if result.depth_frac is None else f"{result.depth_frac:.2f}"
        )
        baseline_niah = (
            "n/a"
            if result.baseline_niah_pass_rate is None
            else f"{result.baseline_niah_pass_rate:.2f}"
        )
        self_swa_niah = (
            "n/a"
            if result.self_swa_niah_pass_rate is None
            else f"{result.self_swa_niah_pass_rate:.2f}"
        )
        print(
            f"{result.prompt_len}\t"
            f"{result.max_model_len}\t"
            f"{result.num_spec_tokens}\t"
            f"{result.self_swa_window_size}\t"
            f"{depth}\t"
            f"{exact_match}\t"
            f"{baseline_tps}\t"
            f"{result.self_swa_decode_tokens_per_s:.2f}\t"
            f"{speedup}\t"
            f"{result.spec_metrics.num_drafts}\t"
            f"{result.spec_metrics.mean_acceptance_length:.2f}\t"
            f"{baseline_niah}\t"
            f"{self_swa_niah}"
        )


def main() -> None:
    args = parse_args()
    validate_env(args)
    prompt_lens = resolve_prompt_lens(args)
    window_sizes = resolve_window_sizes(args)
    num_spec_tokens_values = resolve_num_spec_tokens(args)
    depth_fracs = resolve_depth_fracs(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    chat_template_active = _use_chat_template(args, tokenizer)

    print(f"model: {args.model}")
    print(f"prompt_lens: {prompt_lens}")
    print(f"self_swa_window_sizes: {window_sizes}")
    print(f"tensor_parallel_size: {args.tp}")
    print(f"output_len: {args.output_len}")
    print(f"num_prompts: {args.num_prompts}")
    print(f"num_spec_tokens: {num_spec_tokens_values}")
    print(f"prompt_mode: {args.prompt_mode}")
    print(
        "chat_template: "
        f"{'on' if chat_template_active else 'off'} "
        f"(requested={args.chat_template})"
    )
    if args.prompt_mode == "niah":
        print(f"niah_depth_fracs: {depth_fracs}")
    if args.prompt_mode in ("pg19", "niah"):
        print(f"pg19_dataset: {args.pg19_dataset}")
        print(f"pg19_split: {args.pg19_split}")
        print(f"pg19_cache_dir: {args.pg19_cache_dir}")
    if args.prompt_mode == "pg19":
        print(f"prompt_tail_sentences: {args.prompt_tail_sentences}")
    print(f"baseline_only: {args.baseline_only}")
    print(f"self_swa_only: {args.self_swa_only}")
    print(f"profile_dir: {args.profile_dir}")
    print(f"profile_explicit: {args.profile_explicit}")
    print(f"profile_warmup_runs: {args.profile_warmup_runs}")
    print(f"profile_record_shapes: {args.profile_record_shapes}")
    print(f"profile_separate_dirs: {args.profile_separate_dirs}")
    print(f"profile_split_prefill_decode: {args.profile_split_prefill_decode}")
    print(f"hf_overrides: {args.hf_overrides}")
    print(
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN="
        f"{os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN']}"
    )
    print(
        "VLLM_ROCM_USE_AITER="
        f"{os.environ.get('VLLM_ROCM_USE_AITER', '<unset>')}"
    )
    print(
        "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT="
        f"{os.environ['VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT']}"
    )

    results = run_sweep(
        args,
        tokenizer,
        prompt_lens,
        window_sizes,
        num_spec_tokens_values,
        depth_fracs,
    )
    print_sweep_summary(results)


if __name__ == "__main__":
    main()
