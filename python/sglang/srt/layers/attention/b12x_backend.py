"""b12x SM120 paged attention backend for sglang.

This backend replaces FlashInfer/Triton paged attention with b12x's
SM120 kernels for decode and extend on Blackwell GPUs.  It requires
sglang's page_size to be 64, matching b12x's native paged layout.

Usage:
    sglang --attention-backend b12x ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_B12X_PAGE_SIZE = 64

# Maximum split count — the kernel selects the actual value at runtime
# from a precomputed LUT indexed by max_pages.
_B12X_MAX_NUM_SPLITS = 32


@dataclass
class B12xForwardMetadata:
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    page_table: torch.Tensor
    mode: str  # "decode" or "extend"


class B12xAttnBackend(AttentionBackend):
    """Paged attention backend using b12x SM120 kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        from b12x.integration.attention import allocate_paged_attention_workspace_pool

        self.page_size = model_runner.page_size
        if self.page_size != _B12X_PAGE_SIZE:
            raise ValueError(
                f"b12x attention backend requires page_size={_B12X_PAGE_SIZE}, "
                f"got {self.page_size}"
            )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.max_context_len = model_runner.model_config.context_len
        self.max_pages_per_req = (
            (self.max_context_len + self.page_size - 1) // self.page_size
        )

        self.workspace_pool = allocate_paged_attention_workspace_pool()
        self.forward_metadata: Optional[B12xForwardMetadata] = None

        # Pre-allocate page offsets for graph-stable page table builds.
        self.graph_page_offsets = torch.arange(
            0, self.max_pages_per_req * self.page_size, self.page_size,
            dtype=torch.int64, device=self.device,
        )

        # CUDA graph state (allocated in init_cuda_graph_state).
        self.cuda_graph_cu_seqlens_q: Optional[torch.Tensor] = None
        self.cuda_graph_cache_seqlens: Optional[torch.Tensor] = None
        self.cuda_graph_page_table: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Metadata init (eager)
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens[:bs]

        if forward_batch.forward_mode.is_decode_or_idle():
            cu_seqlens_q = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
            cache_seqlens = seq_lens.to(torch.int32)
            mode = "decode"
        else:
            extend_lens = forward_batch.extend_seq_lens[:bs].to(torch.int32)
            cu_seqlens_q = torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            )
            torch.cumsum(extend_lens, dim=0, out=cu_seqlens_q[1:])
            cache_seqlens = seq_lens.to(torch.int32)
            mode = "extend"

        page_table = self._build_page_table(
            forward_batch.req_pool_indices[:bs], cache_seqlens
        )

        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # CUDA graph lifecycle
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_cu_seqlens_q = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cache_seqlens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_page_table = torch.zeros(
            max_bs, self.max_pages_per_req, dtype=torch.int32, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        assert encoder_lens is None, "b12x backend does not support encoder-decoder models"
        assert self.cuda_graph_cu_seqlens_q is not None

        cu_seqlens_q = self.cuda_graph_cu_seqlens_q
        cache_seqlens = self.cuda_graph_cache_seqlens

        if forward_mode.is_decode_or_idle():
            cache_seqlens[:bs] = seq_lens[:bs].to(torch.int32)
            cu_seqlens_q[: bs + 1] = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
            mode = "decode"
        else:
            if spec_info is not None and hasattr(spec_info, "draft_token_num"):
                tokens_per_req = spec_info.draft_token_num
            else:
                tokens_per_req = num_tokens // bs if bs > 0 else 1
            cu_seqlens_q[: bs + 1] = torch.arange(
                0, bs * tokens_per_req + 1, tokens_per_req,
                dtype=torch.int32, device=self.device,
            )
            # Use max_context_len as placeholder cache length during capture
            # so b12x's q_len <= cache_len validation passes.
            cache_seqlens[:bs] = self.max_context_len
            mode = "extend"

        page_table = self.cuda_graph_page_table
        self._build_page_table_into(
            req_pool_indices[:bs], cache_seqlens[:bs], page_table, bs
        )

        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q[: bs + 1],
            cache_seqlens=cache_seqlens[:bs],
            page_table=page_table[:bs],
            mode=mode,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        assert self.cuda_graph_cache_seqlens is not None
        assert self.cuda_graph_page_table is not None

        cache_seqlens = self.cuda_graph_cache_seqlens
        cache_seqlens[:bs] = seq_lens[:bs].to(torch.int32)

        self._build_page_table_into(
            req_pool_indices[:bs],
            cache_seqlens[:bs],
            self.cuda_graph_page_table,
            bs,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        from b12x.integration.attention import (
            b12x_paged_decode,
            create_paged_attention_plan,
        )

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self.forward_metadata
        k_cache, v_cache = self._get_paged_kv_buffers(
            forward_batch.token_to_kv_pool, layer.layer_id
        )

        total_q = q.shape[0]
        q3 = q.view(total_q, layer.tp_q_head_num, layer.qk_head_dim)

        plan = create_paged_attention_plan(
            q3,
            k_cache,
            v_cache,
            md.page_table,
            md.cache_seqlens,
            md.cu_seqlens_q,
            causal=True,
            mode="decode",
            num_splits=_B12X_MAX_NUM_SPLITS,
        )

        k_descale, v_descale = self._get_descale_tensors(
            layer, md.cache_seqlens.shape[0]
        )

        o = torch.empty(
            total_q, layer.tp_q_head_num, layer.v_head_dim,
            dtype=q.dtype, device=q.device,
        )

        output, _lse = b12x_paged_decode(
            q3,
            k_cache,
            v_cache,
            md.page_table,
            md.cache_seqlens,
            md.cu_seqlens_q,
            workspace=self.workspace_pool,
            plan=plan,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=layer.scaling,
            output=o,
        )

        return output.view(total_q, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        from b12x.integration.attention import (
            b12x_paged_extend,
            create_paged_attention_plan,
        )

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self.forward_metadata
        k_cache, v_cache = self._get_paged_kv_buffers(
            forward_batch.token_to_kv_pool, layer.layer_id
        )

        total_q = q.shape[0]
        q3 = q.view(total_q, layer.tp_q_head_num, layer.qk_head_dim)

        plan = create_paged_attention_plan(
            q3,
            k_cache,
            v_cache,
            md.page_table,
            md.cache_seqlens,
            md.cu_seqlens_q,
            causal=True,
            mode="extend",
            num_splits=_B12X_MAX_NUM_SPLITS,
        )

        k_descale, v_descale = self._get_descale_tensors(
            layer, md.cache_seqlens.shape[0]
        )

        o = torch.empty(
            total_q, layer.tp_q_head_num, layer.v_head_dim,
            dtype=q.dtype, device=q.device,
        )

        output, _lse = b12x_paged_extend(
            q3,
            k_cache,
            v_cache,
            md.page_table,
            md.cache_seqlens,
            md.cu_seqlens_q,
            workspace=self.workspace_pool,
            plan=plan,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=layer.scaling,
            output=o,
        )

        return output.view(total_q, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_paged_kv_buffers(
        self,
        token_to_kv_pool,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape sglang's flat KV buffers into b12x's paged layout.

        sglang stores KV as [total_slots, kv_heads, head_dim].
        b12x expects [num_pages, page_size, kv_heads, head_dim].
        Since sglang allocates in page_size chunks, the flat buffer
        can be reshaped without copying.
        """
        k_flat = token_to_kv_pool.get_key_buffer(layer_id)
        v_flat = token_to_kv_pool.get_value_buffer(layer_id)
        total_slots = k_flat.shape[0]
        num_pages = total_slots // self.page_size
        kv_heads = k_flat.shape[1]
        head_dim = k_flat.shape[2]
        k_paged = k_flat[: num_pages * self.page_size].view(
            num_pages, self.page_size, kv_heads, head_dim
        )
        v_paged = v_flat[: num_pages * self.page_size].view(
            num_pages, self.page_size, kv_heads, head_dim
        )
        return k_paged, v_paged

    def _build_page_table(
        self,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Build a [batch, max_pages] page table from sglang's req_to_token pool."""
        bs = req_pool_indices.shape[0]
        max_cache = int(cache_seqlens.max().item()) if bs > 0 else 0
        max_pages = max((max_cache + self.page_size - 1) // self.page_size, 1)

        stride = self.req_to_token.shape[1]
        page_offsets = torch.arange(
            0, max_pages * self.page_size, self.page_size,
            dtype=torch.int64, device=self.device,
        )
        row_indices = req_pool_indices.to(torch.int64).unsqueeze(1) * stride
        flat_indices = (row_indices + page_offsets.unsqueeze(0)).clamp(
            0, self.req_to_token.numel() - 1
        )
        token_indices = self.req_to_token.view(-1)[flat_indices]
        page_table = (token_indices // self.page_size).to(torch.int32)
        return page_table.contiguous()

    def _build_page_table_into(
        self,
        req_pool_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
        dest: torch.Tensor,
        bs: int,
    ) -> None:
        """Build page table in-place into a pre-allocated graph-stable buffer.

        Uses dest's column count directly instead of computing max_pages from
        cache_seqlens, avoiding a GPU->CPU sync on every replay step.
        """
        stride = self.req_to_token.shape[1]
        page_offsets = self.graph_page_offsets[:dest.shape[1]]
        row_indices = req_pool_indices[:bs].to(torch.int64).unsqueeze(1) * stride
        flat_indices = (row_indices + page_offsets.unsqueeze(0)).clamp(
            0, self.req_to_token.numel() - 1
        )
        token_indices = self.req_to_token.view(-1)[flat_indices]
        dest[:bs] = (token_indices // self.page_size).to(torch.int32)

    def _get_descale_tensors(
        self,
        layer: RadixAttention,
        batch_size: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Broadcast sglang's scalar FP8 scales into b12x's [batch, kv_heads] format."""
        if layer.k_scale is None or layer.v_scale is None:
            return None, None
        k_descale = torch.full(
            (batch_size, self.num_kv_heads),
            layer.k_scale_float,
            dtype=torch.float32,
            device=self.device,
        )
        v_descale = torch.full(
            (batch_size, self.num_kv_heads),
            layer.v_scale_float,
            dtype=torch.float32,
            device=self.device,
        )
        return k_descale, v_descale
