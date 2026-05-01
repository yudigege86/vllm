# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.spec_decode.self_swa import SELF_SWA_FORWARD_CONTEXT_KEY


class _FakeModelConfig:
    model = "Qwen/Qwen2.5-7B-Instruct-1M"
    quantization = None
    max_model_len = 1024
    hf_config = SimpleNamespace(model_type="qwen2")
    hf_text_config = SimpleNamespace(model_type="qwen2")

    def verify_with_parallel_config(self, parallel_config: ParallelConfig) -> None:
        pass


def test_self_swa_speculative_config_reuses_target_model_config():
    target_model_config = _FakeModelConfig()
    target_parallel_config = ParallelConfig()

    speculative_config = SpeculativeConfig(
        method="self_swa",
        num_speculative_tokens=4,
        target_model_config=target_model_config,
        target_parallel_config=target_parallel_config,
    )

    assert speculative_config.use_self_swa()
    assert speculative_config.draft_model_config is target_model_config
    assert speculative_config.draft_parallel_config is target_parallel_config


def test_self_swa_rejects_non_greedy_draft_sampling():
    with pytest.raises(ValueError, match="greedy draft sampling"):
        SpeculativeConfig(
            method="self_swa",
            num_speculative_tokens=4,
            draft_sample_method="gumbel",
            target_model_config=_FakeModelConfig(),
            target_parallel_config=ParallelConfig(),
        )


def test_self_swa_forward_context_override_is_scoped():
    window = (4095, 0)

    with set_forward_context(
        attn_metadata={},
        vllm_config=VllmConfig(),
        additional_kwargs={SELF_SWA_FORWARD_CONTEXT_KEY: window},
    ):
        assert (
            get_forward_context().additional_kwargs[SELF_SWA_FORWARD_CONTEXT_KEY]
            == window
        )

    with set_forward_context(attn_metadata={}, vllm_config=VllmConfig()):
        assert (
            SELF_SWA_FORWARD_CONTEXT_KEY
            not in get_forward_context().additional_kwargs
        )
