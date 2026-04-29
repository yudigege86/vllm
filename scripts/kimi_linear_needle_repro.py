#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproduce Kimi-linear long-context needle retrieval via OpenAI chat API.

Example server:

    vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
        --served-model-name kimi-linear \
        --tensor-parallel-size 8 \
        --max-model-len 65536 \
        --trust-remote-code \
        --dtype bfloat16 \
        --enforce-eager

Example fixed-size repro:

    python scripts/kimi_linear_needle_repro.py --filler-lines 5800

Example sweep for a smaller failing context:

    python scripts/kimi_linear_needle_repro.py \
        --sweep-lines 128,256,512,1024,2048,4096,5800
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "kimi-linear"
DEFAULT_NEEDLE = "COBALT PENGUIN WALKS AT DAWN"


@dataclass
class NeedleResult:
    filler_lines: int
    elapsed_s: float
    prompt_tokens: int | None
    text: str
    passed: bool
    first_top_logprobs: list[tuple[str, float]] | None


def build_prompt(filler_lines: int, needle: str, needle_position: str) -> str:
    filler = "\n".join(f"filler {i}: ignore this line." for i in range(filler_lines))
    prefix = "You are an exact retrieval bot. Return only the uppercase phrase after SECRET_PHRASE.\n"
    suffix = "\nQuestion: What is the exact SECRET_PHRASE? Return only the phrase, no markdown."

    if needle_position == "start":
        body = f"SECRET_PHRASE: {needle}\n" + filler
    elif needle_position == "middle":
        midpoint = filler_lines // 2
        before = "\n".join(f"filler {i}: ignore this line." for i in range(midpoint))
        after = "\n".join(
            f"filler {i}: ignore this line." for i in range(midpoint, filler_lines)
        )
        body = f"{before}\nSECRET_PHRASE: {needle}\n{after}"
    elif needle_position == "end":
        body = f"{filler}\nSECRET_PHRASE: {needle}"
    else:
        raise ValueError(f"Unsupported needle position: {needle_position}")

    return prefix + body + suffix


def post_chat_completion(
    url: str,
    model: str,
    prompt: str,
    timeout_s: int,
    max_tokens: int,
    logprobs: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload.update({"logprobs": True, "top_logprobs": 10})

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code}: {body[:1000]}") from err


def run_case(args: argparse.Namespace, filler_lines: int) -> NeedleResult:
    prompt = build_prompt(filler_lines, args.needle, args.needle_position)
    start = time.time()
    response = post_chat_completion(
        url=args.url,
        model=args.model,
        prompt=prompt,
        timeout_s=args.timeout,
        max_tokens=args.max_tokens,
        logprobs=args.logprobs,
    )
    elapsed_s = time.time() - start

    choice = response["choices"][0]
    text = choice["message"]["content"]
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")

    first_top_logprobs = None
    if args.logprobs:
        first_token_logprobs = choice["logprobs"]["content"][0]["top_logprobs"]
        first_top_logprobs = [
            (entry["token"], round(float(entry["logprob"]), 4))
            for entry in first_token_logprobs
        ]

    return NeedleResult(
        filler_lines=filler_lines,
        elapsed_s=elapsed_s,
        prompt_tokens=prompt_tokens,
        text=text,
        passed=text.strip() == args.needle,
        first_top_logprobs=first_top_logprobs,
    )


def parse_sweep_lines(value: str | None, fallback: int) -> list[int]:
    if not value:
        return [fallback]
    values = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError("--sweep-lines did not contain any integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--needle", default=DEFAULT_NEEDLE)
    parser.add_argument("--filler-lines", type=int, default=5800)
    parser.add_argument(
        "--sweep-lines",
        help="Comma-separated filler line counts, e.g. 128,256,512,1024,2048,4096,5800.",
    )
    parser.add_argument(
        "--needle-position",
        choices=("start", "middle", "end"),
        default="end",
    )
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--logprobs", action="store_true")
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop a sweep at the first failed retrieval.",
    )
    args = parser.parse_args()

    filler_line_counts = parse_sweep_lines(args.sweep_lines, args.filler_lines)
    any_failed = False
    for filler_lines in filler_line_counts:
        result = run_case(args, filler_lines)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] filler_lines={result.filler_lines} "
            f"prompt_tokens={result.prompt_tokens} "
            f"elapsed_s={result.elapsed_s:.2f} "
            f"text={result.text!r}"
        )
        if result.first_top_logprobs is not None:
            print(f"  first_top_logprobs={result.first_top_logprobs}")
        any_failed = any_failed or not result.passed
        if args.stop_on_fail and not result.passed:
            break

    raise SystemExit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
