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
    SelfSWAProposer,
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


def test_self_swa_local_argmax_uses_generic_logits_processor_path():
    class _FakeLogitsProcessor:
        def __init__(self):
            self.called_with = None

        def get_top_tokens(self, lm_head, hidden_states):
            self.called_with = (lm_head, hidden_states)
            return torch.tensor([7], device=hidden_states.device)

    logits_processor = _FakeLogitsProcessor()
    lm_head = object()
    proposer = object.__new__(SelfSWAProposer)
    proposer.model = SimpleNamespace(
        logits_processor=logits_processor,
        lm_head=lm_head,
    )
    hidden_states = torch.zeros(1, 4)

    top_tokens = proposer._greedy_sample(hidden_states)

    assert top_tokens.tolist() == [7]
    assert logits_processor.called_with == (lm_head, hidden_states)


def test_self_swa_local_argmax_falls_back_to_full_logits():
    class _FakeModel:
        def __init__(self):
            self.called_with = None

        def compute_logits(self, hidden_states):
            self.called_with = hidden_states
            return torch.tensor([[0.0, 4.0, 1.0]], device=hidden_states.device)

    model = _FakeModel()
    proposer = object.__new__(SelfSWAProposer)
    proposer.model = model
    hidden_states = torch.zeros(1, 4)

    top_tokens = proposer._greedy_sample(hidden_states)

    assert top_tokens.tolist() == [1]
    assert model.called_with is hidden_states


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


@pytest.mark.parametrize(
    ("sink_size", "window_size", "block_size", "seq_lens_values"),
    [
        (4, 8192, 16, [100000, 100001, 9000, 33]),
        (17, 20, 16, [100, 64, 48, 33]),
        (0, 8192, 16, [1_000_000, 8193, 4096, 1]),
        (4, 19, 16, [100, 64, 48, 33]),
    ],
)
def test_self_swa_reusable_block_aligned_metadata_matches_helper(
    sink_size: int,
    window_size: int,
    block_size: int,
    seq_lens_values: list[int],
):
    proposer = object.__new__(SelfSWAProposer)
    proposer._self_swa_block_aligned_buffers_initialized = False
    proposer.block_size = block_size
    proposer.self_swa_sink_size = sink_size
    proposer.self_swa_window_size = window_size
    proposer.max_batch_size = len(seq_lens_values)
    proposer.device = torch.device("cpu")
    proposer._self_swa_max_visible_blocks = 0
    proposer._self_swa_max_visible_seq_len = 0
    proposer._self_swa_sink_block_limit = 0
    for attr in (
        "_self_swa_block_aligned_block_table",
        "_self_swa_block_aligned_seq_lens",
    ):
        setattr(proposer, attr, None)

    max_seq_len = max(seq_lens_values)
    table_blocks = (max_seq_len + block_size - 1) // block_size
    block_table = torch.arange(
        len(seq_lens_values) * table_blocks, dtype=torch.int32
    ).reshape(len(seq_lens_values), table_blocks)
    seq_lens = torch.tensor(seq_lens_values, dtype=torch.int32)

    reusable_table, reusable_seq_lens, reusable_max_seq_len = (
        proposer._build_self_swa_block_aligned_metadata(
            block_table=block_table,
            seq_lens=seq_lens,
            batch_size=len(seq_lens_values),
        )
    )
    helper_table, helper_seq_lens, helper_max_seq_len = (
        _make_self_swa_block_aligned_metadata(
            block_table=block_table,
            seq_lens=seq_lens,
            sink_size=sink_size,
            recent_window=window_size,
            block_size=block_size,
        )
    )

    assert torch.equal(reusable_table[:, : helper_table.shape[1]], helper_table)
    assert torch.equal(reusable_seq_lens, helper_seq_lens)
    assert reusable_max_seq_len >= helper_max_seq_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/ROCm")
def test_self_swa_fused_block_aligned_metadata_matches_helper_on_gpu():
    proposer = object.__new__(SelfSWAProposer)
    proposer._self_swa_block_aligned_buffers_initialized = False
    proposer.block_size = 16
    proposer.self_swa_sink_size = 4
    proposer.self_swa_window_size = 8192
    proposer.max_batch_size = 4
    proposer.device = torch.device("cuda")
    proposer._self_swa_max_visible_blocks = 0
    proposer._self_swa_max_visible_seq_len = 0
    proposer._self_swa_sink_block_limit = 0
    proposer._self_swa_block_aligned_block_table = None
    proposer._self_swa_block_aligned_seq_lens = None

    block_table = torch.arange(
        4 * 7000, dtype=torch.int32, device="cuda"
    ).reshape(4, 7000)
    seq_lens = torch.tensor(
        [100000, 100001, 9000, 33], dtype=torch.int32, device="cuda"
    )

    fused_table, fused_seq_lens, fused_max_seq_len = (
        proposer._build_self_swa_block_aligned_metadata(
            block_table=block_table,
            seq_lens=seq_lens,
            batch_size=4,
        )
    )
    helper_table, helper_seq_lens, helper_max_seq_len = (
        _make_self_swa_block_aligned_metadata(
            block_table=block_table,
            seq_lens=seq_lens,
            sink_size=4,
            recent_window=8192,
            block_size=16,
        )
    )

    assert torch.equal(fused_table[:, : helper_table.shape[1]], helper_table)
    assert torch.equal(fused_seq_lens, helper_seq_lens)
    assert fused_max_seq_len >= helper_max_seq_len
