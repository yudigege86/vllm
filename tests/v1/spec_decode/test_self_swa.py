# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.attention.backends.rocm_aiter_fa import (
    _make_self_swa_block_aligned_metadata,
)
from vllm.v1.spec_decode.self_swa import (
    SELF_SWA_FORWARD_CONTEXT_KEY,
    SELF_SWA_SINK_SIZE_FORWARD_CONTEXT_KEY,
)


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
    assert speculative_config.self_swa_sink_size == 4


def test_self_swa_sink_size_can_disable_attention_sink():
    speculative_config = SpeculativeConfig(
        method="self_swa",
        num_speculative_tokens=4,
        self_swa_sink_size=0,
        target_model_config=_FakeModelConfig(),
        target_parallel_config=ParallelConfig(),
    )

    assert speculative_config.self_swa_sink_size == 0


def test_self_swa_rejects_negative_sink_size():
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="self_swa",
            num_speculative_tokens=4,
            self_swa_sink_size=-1,
            target_model_config=_FakeModelConfig(),
            target_parallel_config=ParallelConfig(),
        )


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
    sink_size = 4

    with set_forward_context(
        attn_metadata={},
        vllm_config=VllmConfig(),
        additional_kwargs={
            SELF_SWA_FORWARD_CONTEXT_KEY: window,
            SELF_SWA_SINK_SIZE_FORWARD_CONTEXT_KEY: sink_size,
        },
    ):
        additional_kwargs = get_forward_context().additional_kwargs
        assert additional_kwargs[SELF_SWA_FORWARD_CONTEXT_KEY] == window
        assert additional_kwargs[SELF_SWA_SINK_SIZE_FORWARD_CONTEXT_KEY] == sink_size

    with set_forward_context(attn_metadata={}, vllm_config=VllmConfig()):
        additional_kwargs = get_forward_context().additional_kwargs
        assert SELF_SWA_FORWARD_CONTEXT_KEY not in additional_kwargs
        assert SELF_SWA_SINK_SIZE_FORWARD_CONTEXT_KEY not in additional_kwargs


def test_self_swa_block_aligned_metadata_rounds_sink_to_blocks():
    block_table = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    seq_lens = torch.tensor([100, 33], dtype=torch.int32)

    block_aligned_table, visible_seq_lens, max_visible_seq_len = (
        _make_self_swa_block_aligned_metadata(
            block_table=block_table,
            seq_lens=seq_lens,
            sink_size=17,
            recent_window=20,
            block_size=16,
        )
    )

    assert block_aligned_table.tolist() == [
        [0, 1, 5, 6],
        [16, 17, 18, 16],
    ]
    assert visible_seq_lens.tolist() == [52, 33]
    assert max_visible_seq_len == 52


def test_self_swa_block_aligned_metadata_rounds_recent_start_down():
    block_table = torch.arange(16, dtype=torch.int32).reshape(1, 16)
    seq_lens = torch.tensor([100], dtype=torch.int32)

    block_aligned_table, visible_seq_lens, max_visible_seq_len = (
        _make_self_swa_block_aligned_metadata(
            block_table=block_table,
            seq_lens=seq_lens,
            sink_size=4,
            recent_window=19,
            block_size=16,
        )
    )

    assert block_aligned_table.tolist() == [[0, 5, 6]]
    assert visible_seq_lens.tolist() == [36]
    assert max_visible_seq_len == 36
