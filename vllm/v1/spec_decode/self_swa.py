# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata

# Experimental knobs for the first self-SWA prototype.
SELF_SWA_WINDOW_SIZE = 4096
SELF_SWA_FORWARD_CONTEXT_KEY = "self_swa_sliding_window"


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

        self._self_swa_window = (SELF_SWA_WINDOW_SIZE - 1, 0)

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

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Reuse the base AttentionGroup initialization, but with
        # _draft_attn_layer_names set to the target attention layers.
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)

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
    ) -> torch.Tensor:
        del target_hidden_states, mm_embed_inputs, num_rejected_tokens_gpu
        del slot_mappings

        if not sampling_metadata.all_greedy:
            raise NotImplementedError(
                "self_swa currently supports greedy requests only."
            )

        batch_size = common_attn_metadata.batch_size()
        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        if (
            common_attn_metadata.max_seq_len <= SELF_SWA_WINDOW_SIZE
            or common_attn_metadata.max_seq_len + self.num_speculative_tokens
            > self.max_model_len
        ):
            return torch.zeros(
                (batch_size, self.num_speculative_tokens),
                dtype=torch.int64,
                device=self.device,
            )

        if target_positions.ndim != 1:
            raise NotImplementedError("self_swa currently supports 1D positions only.")

        seq_lens = common_attn_metadata.seq_lens.clone()
        positions = target_positions[token_indices_to_sample].clone()
        input_ids = next_token_ids.int()
        query_start_loc = self.arange[: batch_size + 1]
        query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()

        draft_token_ids_list: list[torch.Tensor] = []
        for draft_index in range(self.num_speculative_tokens):
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

            self.input_ids[:batch_size] = input_ids
            model_kwargs: dict[str, Any] = {
                "input_ids": self.input_ids[:batch_size],
                "positions": positions,
                "inputs_embeds": None,
            }
            slot_mapping = {
                name: out_slot_mapping for name in self._draft_attn_layer_names
            }
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
            draft_token_ids_list.append(draft_token_ids)
            input_ids = draft_token_ids.int()

        return torch.stack(draft_token_ids_list, dim=1)
