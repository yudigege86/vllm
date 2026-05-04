# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata

# Experimental knobs for the first self-SWA prototype.
SELF_SWA_WINDOW_SIZE = 4096
SELF_SWA_FORWARD_CONTEXT_KEY = "self_swa_sliding_window"
SELF_SWA_DEBUG_ENV = "VLLM_SELF_SWA_DEBUG"

logger = init_logger(__name__)


def _debug_enabled() -> bool:
    return os.environ.get(SELF_SWA_DEBUG_ENV, "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _tensor_debug(tensor: torch.Tensor | None, limit: int = 8):
    if tensor is None:
        return None
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "capturing": True,
        }
    return tensor.detach().flatten()[:limit].cpu().tolist()


class SelfSWAProposer(SpecDecodeBaseProposer):
    """Draft with the target model using a temporary sliding-window mask.

    Unlike DraftModelProposer, this proposer does not load a second model and
    does not maintain a separate draft KV cache. It writes provisional KV for
    drafted tokens into the target request's lookahead slots. The next target
    verification pass recomputes those slots with full attention.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        if self.uses_mrope or self.uses_xdrope_dim > 0:
            raise NotImplementedError("self_swa currently supports 1D positions only.")
        if self.parallel_drafting:
            raise NotImplementedError(
                "self_swa currently supports serial drafting only."
            )
        if self.supports_mm_inputs:
            raise NotImplementedError("self_swa currently supports text-only models.")

        self.self_swa_window_size = self.speculative_config.self_swa_window_size
        self._self_swa_window = (self.self_swa_window_size - 1, 0)
        if _debug_enabled():
            logger.info(
                "self_swa init: window=%s num_speculative_tokens=%s "
                "max_model_len=%s",
                self._self_swa_window,
                self.num_speculative_tokens,
                self.max_model_len,
            )

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = {
            name
            for name, layer in all_attn_layers.items()
            if layer.get_kv_cache_spec(self.vllm_config) is not None
        }
        if _debug_enabled():
            logger.info(
                "self_swa load_model: target_model=%s draft_attn_layers=%s "
                "sample_layers=%s",
                target_model.__class__.__name__,
                len(self._draft_attn_layer_names),
                sorted(self._draft_attn_layer_names)[:8],
            )

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Reuse the base AttentionGroup initialization, but with
        # _draft_attn_layer_names set to the target attention layers.
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    def model_returns_tuple(self) -> bool:
        return False

    @staticmethod
    def _empty_drafts(batch_size: int) -> list[list[int]]:
        return [[] for _ in range(batch_size)]

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor | list[list[int]]:
        del target_hidden_states, mm_embed_inputs
        del slot_mappings

        if not sampling_metadata.all_greedy:
            raise NotImplementedError(
                "self_swa currently supports greedy requests only."
            )

        batch_size = common_attn_metadata.batch_size()
        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        if _debug_enabled():
            logger.info(
                "self_swa propose enter: batch_size=%s target_tokens=%s "
                "target_positions_shape=%s max_seq_len=%s max_query_len=%s "
                "num_actual_tokens=%s num_speculative_tokens=%s "
                "token_indices_to_sample=%s next_token_ids=%s seq_lens=%s "
                "query_start_loc=%s num_rejected=%s block_table_shape=%s",
                batch_size,
                target_token_ids.shape[0],
                tuple(target_positions.shape),
                common_attn_metadata.max_seq_len,
                common_attn_metadata.max_query_len,
                common_attn_metadata.num_actual_tokens,
                self.num_speculative_tokens,
                _tensor_debug(token_indices_to_sample),
                _tensor_debug(next_token_ids),
                _tensor_debug(common_attn_metadata.seq_lens),
                _tensor_debug(common_attn_metadata.query_start_loc),
                _tensor_debug(num_rejected_tokens_gpu),
                tuple(common_attn_metadata.block_table_tensor.shape),
            )

        if (
            common_attn_metadata.max_seq_len <= self.self_swa_window_size
            or common_attn_metadata.max_seq_len + self.num_speculative_tokens
            > self.max_model_len
        ):
            if _debug_enabled():
                logger.info(
                    "self_swa propose skip: max_seq_len=%s window_size=%s "
                    "num_speculative_tokens=%s max_model_len=%s",
                    common_attn_metadata.max_seq_len,
                    self.self_swa_window_size,
                    self.num_speculative_tokens,
                    self.max_model_len,
                )
            return self._empty_drafts(batch_size)

        if target_positions.ndim != 1:
            raise NotImplementedError("self_swa currently supports 1D positions only.")

        seq_lens = common_attn_metadata.seq_lens.clone()
        positions = target_positions[token_indices_to_sample].clone()
        input_ids = next_token_ids.int()
        first_draft_positions = positions + 1
        first_draft_block_numbers = first_draft_positions // self.block_size
        first_draft_block_ids = common_attn_metadata.block_table_tensor.gather(
            dim=1, index=first_draft_block_numbers[:, None]
        ).squeeze(1)
        has_unallocated_lookahead = bool((first_draft_block_ids <= 0).any().item())
        if _debug_enabled():
            logger.info(
                "self_swa propose initial decode state: positions=%s seq_lens=%s "
                "input_ids=%s first_draft_positions=%s "
                "first_draft_block_numbers=%s first_draft_block_ids=%s "
                "has_unallocated_lookahead=%s",
                _tensor_debug(positions),
                _tensor_debug(seq_lens),
                _tensor_debug(input_ids),
                _tensor_debug(first_draft_positions),
                _tensor_debug(first_draft_block_numbers),
                _tensor_debug(first_draft_block_ids),
                has_unallocated_lookahead,
            )
        if has_unallocated_lookahead:
            if _debug_enabled():
                logger.info(
                    "self_swa propose skip: missing lookahead KV block for "
                    "first draft positions=%s block_numbers=%s block_ids=%s",
                    _tensor_debug(first_draft_positions),
                    _tensor_debug(first_draft_block_numbers),
                    _tensor_debug(first_draft_block_ids),
                )
            return self._empty_drafts(batch_size)

        query_start_loc = self.arange[: batch_size + 1]
        query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        draft_token_ids_list: list[torch.Tensor] = []
        for draft_index in range(self.num_speculative_tokens):
            prev_positions = positions.clone() if _debug_enabled() else None
            out_positions = self.positions[:batch_size]
            out_slot_mapping = self._slot_mapping_buffer[:batch_size]
            eagle_step_update_slot_mapping_and_metadata(
                positions_1d=positions,
                block_table_tensor=common_attn_metadata.block_table_tensor,
                seq_lens=seq_lens,
                block_size=self.block_size,
                max_model_len=self.max_model_len,
                out_clamped_positions=out_positions,
                out_slot_mapping=out_slot_mapping,
            )
            positions = out_positions
            if _debug_enabled():
                block_numbers = positions // self.block_size
                expected_block_ids = common_attn_metadata.block_table_tensor.gather(
                    dim=1, index=block_numbers[:, None]
                ).squeeze(1)
                expected_slot_mapping = (
                    expected_block_ids * self.block_size + positions % self.block_size
                )
                logger.info(
                    "self_swa draft step metadata: draft_index=%s "
                    "prev_positions=%s positions=%s block_numbers=%s "
                    "expected_block_ids=%s expected_slot_mapping=%s "
                    "seq_lens=%s slot_mapping=%s block_table_row_head=%s "
                    "block_table_row_at_block=%s",
                    draft_index,
                    _tensor_debug(prev_positions),
                    _tensor_debug(positions),
                    _tensor_debug(block_numbers),
                    _tensor_debug(expected_block_ids),
                    _tensor_debug(expected_slot_mapping),
                    _tensor_debug(seq_lens),
                    _tensor_debug(out_slot_mapping),
                    _tensor_debug(common_attn_metadata.block_table_tensor[0, :8]),
                    _tensor_debug(expected_block_ids),
                )

            draft_common_attn_metadata = common_attn_metadata.replace(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                num_actual_tokens=batch_size,
                max_query_len=1,
                max_seq_len=min(
                    common_attn_metadata.max_seq_len + draft_index + 1,
                    self.max_model_len,
                ),
                slot_mapping=out_slot_mapping,
                seq_lens_cpu_upper_bound=None,
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
            )
            _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
                draft_common_attn_metadata, draft_index=draft_index
            )
            if _debug_enabled():
                logger.info(
                    "self_swa draft attention metadata: draft_index=%s "
                    "max_seq_len=%s num_actual_tokens=%s max_query_len=%s "
                    "query_start_loc=%s per_layer_metadata=%s",
                    draft_index,
                    draft_common_attn_metadata.max_seq_len,
                    draft_common_attn_metadata.num_actual_tokens,
                    draft_common_attn_metadata.max_query_len,
                    _tensor_debug(draft_common_attn_metadata.query_start_loc),
                    len(per_layer_attn_metadata),
                )

            self.input_ids[:batch_size] = input_ids
            model_kwargs: dict[str, Any] = {
                "input_ids": self.input_ids[:batch_size],
                "positions": positions,
                "inputs_embeds": None,
            }
            slot_mapping = {
                name: out_slot_mapping for name in self._draft_attn_layer_names
            }
            if _debug_enabled():
                logger.info(
                    "self_swa draft model input: draft_index=%s input_ids=%s "
                    "positions=%s window=%s slot_mapping_layers=%s",
                    draft_index,
                    _tensor_debug(self.input_ids[:batch_size]),
                    _tensor_debug(positions),
                    self._self_swa_window,
                    len(slot_mapping),
                )
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=batch_size,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=slot_mapping,
                additional_kwargs={
                    SELF_SWA_FORWARD_CONTEXT_KEY: self._self_swa_window,
                },
            ):
                ret_hidden_states = self.model(**model_kwargs)
                if not self.model_returns_tuple():
                    last_hidden_states = ret_hidden_states
                else:
                    last_hidden_states, _ = ret_hidden_states

            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
            if _debug_enabled():
                logger.info(
                    "self_swa draft output: draft_index=%s hidden_shape=%s "
                    "draft_token_ids=%s",
                    draft_index,
                    tuple(last_hidden_states[:batch_size].shape),
                    _tensor_debug(draft_token_ids),
                )
            draft_token_ids_list.append(draft_token_ids)
            input_ids = draft_token_ids.int()

        stacked_drafts = torch.stack(draft_token_ids_list, dim=1)
        if _debug_enabled():
            logger.info(
                "self_swa propose return: draft_ids=%s",
                _tensor_debug(stacked_drafts),
            )
        return stacked_drafts
