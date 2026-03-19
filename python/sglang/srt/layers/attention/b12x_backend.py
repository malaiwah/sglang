"""b12x SM120 paged attention backend for sglang."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from b12x.attention.paged.planner import build_decode_chunk_pages_lut

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from b12x.integration.attention import PagedAttentionWorkspace
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_B12X_PAGE_SIZE = 64
_B12X_DECODE_BLOCK_CHUNKS = 128
_B12X_DECODE_BLOCK_PAGES = 128


@triton.jit
def build_b12x_decode_graph_page_table_triton(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_table_ptr,
    req_to_token_row_stride,
    page_table_row_stride,
    max_pages_per_req,
    PAGE_SIZE: tl.constexpr,
    BLOCK_PAGES: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    page_block_idx = tl.program_id(axis=1)

    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx).to(tl.int64)
    page_offsets = page_block_idx * BLOCK_PAGES + tl.arange(0, BLOCK_PAGES)
    page_mask = page_offsets < max_pages_per_req
    flat_token_offsets = req_pool_idx * req_to_token_row_stride + page_offsets.to(tl.int64) * PAGE_SIZE
    token_indices = tl.load(req_to_token_ptr + flat_token_offsets, mask=page_mask, other=0)
    tl.store(
        page_table_ptr + req_idx * page_table_row_stride + page_offsets,
        (token_indices // PAGE_SIZE).to(tl.int32),
        mask=page_mask,
    )


@triton.jit
def update_b12x_decode_graph_metadata_triton(
    cache_seqlens_ptr,
    merge_indptr_ptr,
    block_valid_mask_ptr,
    chunk_pages_ptr,
    max_chunks_per_req,
    PAGE_SIZE: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    chunk_block_idx = tl.program_id(axis=1)

    cache_len = tl.load(cache_seqlens_ptr + req_idx).to(tl.int32)
    chunk_pages = tl.load(chunk_pages_ptr).to(tl.int32)
    num_pages = tl.maximum((cache_len + (PAGE_SIZE - 1)) // PAGE_SIZE, 1)
    num_chunks = (num_pages + chunk_pages - 1) // chunk_pages

    tl.store(merge_indptr_ptr + req_idx + 1, num_chunks)

    chunk_offsets = chunk_block_idx * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
    chunk_mask = chunk_offsets < max_chunks_per_req
    is_active = chunk_offsets < num_chunks
    tl.store(
        block_valid_mask_ptr + req_idx * max_chunks_per_req + chunk_offsets,
        is_active.to(tl.int32),
        mask=chunk_mask,
    )


@dataclass
class B12xForwardMetadata:
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    page_table: torch.Tensor
    mode: str
    use_cuda_graph: bool
    graph_key: tuple[str, int] | None = None
    req_pool_indices: torch.Tensor | None = None
    decode_chunk_pages: torch.Tensor | None = None


class B12xAttnBackend(AttentionBackend):
    """Paged attention backend using b12x SM120 kernels."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        from b12x.integration.attention import PagedAttentionWorkspace

        self.workspace_cls = PagedAttentionWorkspace
        self.page_size = model_runner.page_size
        if self.page_size != _B12X_PAGE_SIZE:
            raise ValueError(
                f"b12x attention backend requires page_size={_B12X_PAGE_SIZE}, got {self.page_size}"
            )

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.q_dtype = model_runner.dtype
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.device = torch.device(model_runner.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.server_args = model_runner.server_args
        self.max_running_requests = int(getattr(model_runner, "max_running_requests", 1))
        self.max_context_len = model_runner.model_config.context_len
        self.max_pages_per_req = (self.max_context_len + self.page_size - 1) // self.page_size
        self.kv_contract_layer_id = self._select_kv_contract_layer_id()
        self.num_cache_pages = int(
            self.token_to_kv_pool.get_key_buffer(self.kv_contract_layer_id).shape[0] // self.page_size
        )

        self.forward_metadata: Optional[B12xForwardMetadata] = None
        self.eager_workspaces: dict[str, PagedAttentionWorkspace] = {}
        self.cuda_graph_workspaces: dict[tuple[str, int], PagedAttentionWorkspace] = {}

        self.graph_page_offsets = torch.arange(
            0,
            self.max_pages_per_req * self.page_size,
            self.page_size,
            dtype=torch.int64,
            device=self.device,
        )

        self.cuda_graph_cu_seqlens_q: Optional[torch.Tensor] = None
        self.cuda_graph_cache_seqlens: Optional[torch.Tensor] = None
        self.cuda_graph_page_table: Optional[torch.Tensor] = None
        self.cuda_graph_decode_chunk_pages: Optional[torch.Tensor] = None
        self.cuda_graph_decode_chunk_pages_lut: Optional[torch.Tensor] = None
        self.cuda_graph_decode_max_chunks_per_req: Optional[int] = None
        self.cuda_graph_decode_worst_page_count: Optional[int] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        cache_seqlens = forward_batch.seq_lens[:bs].to(torch.int32)
        mode = self._mode_from_forward_mode(forward_batch.forward_mode)
        cu_seqlens_q = self._build_eager_cu_seqlens_q(forward_batch, bs, mode)
        page_table = self._build_page_table(forward_batch.req_pool_indices[:bs], cache_seqlens)
        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            mode=mode,
            use_cuda_graph=False,
        )

    def requires_seq_lens_cpu_for_replay(self, forward_mode: Optional[ForwardMode] = None) -> bool:
        del forward_mode
        return False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        del max_num_tokens
        self.cuda_graph_cu_seqlens_q = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_cu_seqlens_q.copy_(
            torch.arange(0, max_bs + 1, dtype=torch.int32, device=self.device)
        )
        self.cuda_graph_cache_seqlens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_page_table = torch.zeros(
            max_bs, self.max_pages_per_req, dtype=torch.int32, device=self.device
        )
        self.cuda_graph_decode_chunk_pages = torch.ones(
            1, dtype=torch.int32, device=self.device
        )
        decode_chunk_pages_lut = build_decode_chunk_pages_lut(
            q_dtype=self.q_dtype,
            kv_dtype=self.kv_cache_dtype,
            batch=max_bs,
            page_size=self.page_size,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            gqa_group_size=self.num_q_heads // self.num_kv_heads,
            max_effective_kv_pages=self.max_pages_per_req,
        )
        self.cuda_graph_decode_chunk_pages_lut = torch.tensor(
            (decode_chunk_pages_lut[0], *decode_chunk_pages_lut),
            dtype=torch.int32,
            device=self.device,
        )
        self.cuda_graph_decode_chunk_pages[0] = int(decode_chunk_pages_lut[0])
        worst_page_count = 1
        max_chunks_per_req = 1
        for page_count, chunk_pages in enumerate(decode_chunk_pages_lut, start=1):
            num_chunks = (page_count + int(chunk_pages) - 1) // int(chunk_pages)
            if num_chunks > max_chunks_per_req:
                max_chunks_per_req = num_chunks
                worst_page_count = page_count
        self.cuda_graph_decode_max_chunks_per_req = max_chunks_per_req
        self.cuda_graph_decode_worst_page_count = worst_page_count

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
        mode = self._mode_from_forward_mode(forward_mode)
        graph_key = (mode, bs)
        workspace = self._get_or_create_graph_workspace(graph_key, mode=mode, total_q_capacity=num_tokens)
        if mode == "decode":
            self._bind_decode_graph_runtime_buffers(workspace, bs, cache_seqlens=seq_lens[:bs])
            cache_seqlens = workspace.cache_seqlens
            page_table = workspace.page_table
            cu_seqlens_q = workspace.cu_seqlens_q
        else:
            cache_seqlens, page_table, cu_seqlens_q = self._fill_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                total_q_hint=num_tokens,
            )
            workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            mode=mode,
            use_cuda_graph=True,
            graph_key=graph_key,
            req_pool_indices=req_pool_indices[:bs] if mode == "decode" else None,
            decode_chunk_pages=self.cuda_graph_decode_chunk_pages if mode == "decode" else None,
        )

    def init_forward_metadata_replay_cuda_graph_no_cpu(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        del seq_lens_sum
        assert encoder_lens is None, "b12x backend does not support encoder-decoder models"
        mode = self._mode_from_forward_mode(forward_mode)
        graph_key = (mode, bs)
        workspace = self.cuda_graph_workspaces.get(graph_key)
        if workspace is None:
            raise RuntimeError(f"missing captured b12x cuda-graph workspace for {graph_key}")
        if mode == "decode":
            self._bind_decode_graph_runtime_buffers(workspace, bs, cache_seqlens=seq_lens[:bs])
            cache_seqlens = workspace.cache_seqlens
            page_table = workspace.page_table
            cu_seqlens_q = workspace.cu_seqlens_q
        else:
            cache_seqlens, page_table, cu_seqlens_q = self._fill_cuda_graph_metadata(
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                total_q_hint=workspace.total_q_capacity,
            )
            workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            mode=mode,
            use_cuda_graph=True,
            graph_key=graph_key,
            req_pool_indices=req_pool_indices[:bs] if mode == "decode" else None,
            decode_chunk_pages=self.cuda_graph_decode_chunk_pages if mode == "decode" else None,
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
        del seq_lens_cpu
        self.init_forward_metadata_replay_cuda_graph_no_cpu(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        self._validate_layer_contract(layer)
        if save_kv_cache and k is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self._require_forward_metadata("decode")
        workspace = self._get_workspace(md, total_q=q.shape[0])
        if not md.use_cuda_graph:
            workspace.prepare(md.page_table, md.cache_seqlens, md.cu_seqlens_q)
        elif layer.layer_id == self.kv_contract_layer_id:
            self._run_decode_graph_metadata_update(workspace, md)

        k_cache, v_cache = self._get_paged_kv_buffers(forward_batch.token_to_kv_pool, layer.layer_id)
        q3 = q.view(q.shape[0], layer.tp_q_head_num, layer.qk_head_dim)
        output = torch.empty(
            q.shape[0],
            layer.tp_q_head_num,
            layer.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        k_descale, v_descale = self._get_descale_tensors(layer, md.cache_seqlens.shape[0])
        out, _ = workspace.run(
            q3,
            k_cache,
            v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return out.view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        self._validate_layer_contract(layer)
        if save_kv_cache and k is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        md = self._require_forward_metadata("extend")
        workspace = self._get_workspace(md, total_q=q.shape[0])
        if not md.use_cuda_graph:
            workspace.prepare(md.page_table, md.cache_seqlens, md.cu_seqlens_q)

        k_cache, v_cache = self._get_paged_kv_buffers(forward_batch.token_to_kv_pool, layer.layer_id)
        q3 = q.view(q.shape[0], layer.tp_q_head_num, layer.qk_head_dim)
        output = torch.empty(
            q.shape[0],
            layer.tp_q_head_num,
            layer.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        k_descale, v_descale = self._get_descale_tensors(layer, md.cache_seqlens.shape[0])
        out, _ = workspace.run(
            q3,
            k_cache,
            v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return out.view(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def _require_forward_metadata(self, expected_mode: str) -> B12xForwardMetadata:
        if self.forward_metadata is None:
            raise RuntimeError("b12x backend metadata has not been initialized")
        if self.forward_metadata.mode != expected_mode:
            raise RuntimeError(
                f"b12x backend expected {expected_mode} metadata, got {self.forward_metadata.mode}"
            )
        return self.forward_metadata

    def _get_workspace(self, md: B12xForwardMetadata, *, total_q: int) -> PagedAttentionWorkspace:
        if md.use_cuda_graph:
            assert md.graph_key is not None
            workspace = self.cuda_graph_workspaces.get(md.graph_key)
            if workspace is None:
                raise RuntimeError(f"missing captured b12x cuda-graph workspace for {md.graph_key}")
            return workspace
        if md.mode == "extend":
            return self._get_or_create_eager_extend_workspace()
        workspace = self.eager_workspaces.get(md.mode)
        if workspace is None:
            workspace = self.workspace_cls.for_contract(
                mode=md.mode,
                device=self.device,
                dtype=self.q_dtype,
                kv_dtype=self.kv_cache_dtype,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                head_dim_vo=self.head_dim,
                page_size=self.page_size,
                max_total_q=total_q,
                num_cache_pages=self.num_cache_pages,
                use_cuda_graph=False,
            )
            self.eager_workspaces[md.mode] = workspace
        return workspace

    def _get_or_create_eager_extend_workspace(self) -> PagedAttentionWorkspace:
        workspace = self.eager_workspaces.get("extend")
        if workspace is not None:
            return workspace

        total_q_capacity = self._eager_extend_total_q_capacity()
        batch_capacity = self._eager_extend_batch_capacity(total_q_capacity)
        workspace = self.workspace_cls.for_eager_extend_capacity(
            device=self.device,
            dtype=self.q_dtype,
            kv_dtype=self.kv_cache_dtype,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            head_dim_vo=self.head_dim,
            page_size=self.page_size,
            max_total_q=total_q_capacity,
            max_batch=batch_capacity,
            max_page_table_width=self.max_pages_per_req,
            num_cache_pages=self.num_cache_pages,
            use_cuda_graph=False,
        )
        self.eager_workspaces["extend"] = workspace
        return workspace

    def _eager_extend_total_q_capacity(self) -> int:
        chunked_prefill_size = int(getattr(self.server_args, "chunked_prefill_size", -1) or -1)
        if chunked_prefill_size > 0:
            return chunked_prefill_size
        return int(self.server_args.max_prefill_tokens)

    def _eager_extend_batch_capacity(self, total_q_capacity: int) -> int:
        configured_batch = self.server_args.prefill_max_requests
        if configured_batch is None:
            configured_batch = self.max_running_requests
        return max(1, min(int(configured_batch), int(total_q_capacity)))

    def _get_or_create_graph_workspace(
        self,
        graph_key: tuple[str, int],
        *,
        mode: str,
        total_q_capacity: int,
    ) -> PagedAttentionWorkspace:
        workspace = self.cuda_graph_workspaces.get(graph_key)
        if workspace is None:
            workspace = self.workspace_cls.for_contract(
                mode=mode,
                device=self.device,
                dtype=self.q_dtype,
                kv_dtype=self.kv_cache_dtype,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                head_dim_vo=self.head_dim,
                page_size=self.page_size,
                max_total_q=total_q_capacity,
                num_cache_pages=self.num_cache_pages,
                use_cuda_graph=True,
            )
            self._prime_graph_workspace_capacity(
                workspace,
                mode=mode,
                bs=graph_key[1],
                total_q_capacity=total_q_capacity,
            )
            self.cuda_graph_workspaces[graph_key] = workspace
            return workspace
        if workspace.total_q_capacity != total_q_capacity:
            raise ValueError(
                "b12x backend currently expects one cuda-graph q-capacity per (mode, batch) bucket"
            )
        return workspace

    def _mode_from_forward_mode(self, forward_mode: ForwardMode) -> str:
        return "decode" if forward_mode.is_decode_or_idle() else "extend"

    def _select_kv_contract_layer_id(self) -> int:
        mapping = getattr(self.token_to_kv_pool, "full_attention_layer_id_mapping", None)
        if mapping is not None:
            if not mapping:
                raise ValueError("b12x attention backend requires at least one full-attention layer")
            return int(next(iter(mapping)))
        return 0

    def _build_eager_cu_seqlens_q(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        mode: str,
    ) -> torch.Tensor:
        if mode == "decode":
            return torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
        extend_lens = forward_batch.extend_seq_lens[:bs].to(torch.int32)
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(extend_lens, dim=0, out=cu_seqlens_q[1:])
        return cu_seqlens_q

    def _fill_cuda_graph_metadata(
        self,
        *,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        total_q_hint: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.cuda_graph_cu_seqlens_q is not None
        assert self.cuda_graph_cache_seqlens is not None
        assert self.cuda_graph_page_table is not None

        cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
        cache_seqlens.copy_(seq_lens[:bs].to(torch.int32))

        cu_seqlens_q = self.cuda_graph_cu_seqlens_q[: bs + 1]
        mode = self._mode_from_forward_mode(forward_mode)
        if mode == "decode":
            cu_seqlens_q.copy_(torch.arange(0, bs + 1, dtype=torch.int32, device=self.device))
        else:
            if bs <= 0:
                raise ValueError("b12x cuda-graph extend path requires bs > 0")
            if spec_info is not None and hasattr(spec_info, "draft_token_num"):
                tokens_per_req = int(spec_info.draft_token_num)
                total_q = bs * tokens_per_req
            else:
                total_q = int(total_q_hint)
                if total_q % bs != 0:
                    raise ValueError(
                        f"b12x cuda-graph extend path requires uniform q-per-request, got total_q={total_q}, bs={bs}"
                    )
                tokens_per_req = total_q // bs
            cu_seqlens_q.copy_(
                torch.arange(
                    0,
                    bs * tokens_per_req + 1,
                    tokens_per_req,
                    dtype=torch.int32,
                    device=self.device,
                )
            )

        page_table = self.cuda_graph_page_table[:bs]
        self._build_page_table_into(req_pool_indices[:bs], cache_seqlens, page_table, bs)
        return cache_seqlens, page_table, cu_seqlens_q

    def _validate_layer_contract(self, layer: RadixAttention) -> None:
        if layer.tp_q_head_num != self.num_q_heads or layer.tp_k_head_num != self.num_kv_heads:
            raise ValueError("b12x backend expects layer head counts to match the backend contract")
        if layer.tp_v_head_num != self.num_kv_heads:
            raise ValueError("b12x backend expects tp_v_head_num to match tp_k_head_num")
        if layer.qk_head_dim != self.head_dim or layer.v_head_dim != self.head_dim:
            raise ValueError("b12x backend currently expects qk_head_dim == v_head_dim == model head_dim")

    def _prime_graph_workspace_capacity(
        self,
        workspace: PagedAttentionWorkspace,
        *,
        mode: str,
        bs: int,
        total_q_capacity: int,
    ) -> None:
        max_cache_seqlen = self.max_context_len
        if mode == "decode":
            if self.cuda_graph_decode_worst_page_count is None:
                raise RuntimeError("decode graph chunk-pages LUT is not initialized")
            max_cache_seqlen = int(self.cuda_graph_decode_worst_page_count * self.page_size)
        workspace.prepare_for_capacity(
            batch=bs,
            total_q_capacity=total_q_capacity,
            max_page_table_width=self.max_pages_per_req,
            max_cache_seqlen=max_cache_seqlen,
        )
        if mode == "decode":
            self._bind_decode_graph_runtime_buffers(workspace, bs)
            self._validate_decode_graph_chunk_capacity(workspace, bs=bs)

    def _get_paged_kv_buffers(
        self,
        token_to_kv_pool,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        bs = req_pool_indices.shape[0]
        max_cache = int(cache_seqlens.max().item()) if bs > 0 else 0
        max_pages = max((max_cache + self.page_size - 1) // self.page_size, 1)

        stride = self.req_to_token.shape[1]
        page_offsets = torch.arange(
            0, max_pages * self.page_size, self.page_size, dtype=torch.int64, device=self.device
        )
        row_indices = req_pool_indices.to(torch.int64).unsqueeze(1) * stride
        flat_indices = (row_indices + page_offsets.unsqueeze(0)).clamp(0, self.req_to_token.numel() - 1)
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
        del cache_seqlens
        stride = self.req_to_token.shape[1]
        page_offsets = self.graph_page_offsets[: dest.shape[1]]
        row_indices = req_pool_indices[:bs].to(torch.int64).unsqueeze(1) * stride
        flat_indices = (row_indices + page_offsets.unsqueeze(0)).clamp(0, self.req_to_token.numel() - 1)
        token_indices = self.req_to_token.view(-1)[flat_indices]
        dest[:bs] = (token_indices // self.page_size).to(torch.int32)

    def _bind_decode_graph_runtime_buffers(
        self,
        workspace: PagedAttentionWorkspace,
        bs: int,
        *,
        cache_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        assert self.cuda_graph_page_table is not None
        assert self.cuda_graph_cu_seqlens_q is not None
        workspace.page_table = self.cuda_graph_page_table[:bs]
        if cache_seqlens is None:
            assert self.cuda_graph_cache_seqlens is not None
            workspace.cache_seqlens = self.cuda_graph_cache_seqlens[:bs]
        else:
            workspace.cache_seqlens = cache_seqlens
        workspace.cu_seqlens_q = self.cuda_graph_cu_seqlens_q[: bs + 1]

    def _validate_decode_graph_chunk_capacity(
        self,
        workspace: PagedAttentionWorkspace,
        *,
        bs: int,
    ) -> None:
        if workspace.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if bs <= 0:
            raise ValueError("decode graph replay requires bs > 0")

        work_items_capacity = int(workspace.request_indices.shape[0])
        if work_items_capacity % bs != 0:
            raise RuntimeError(
                "decode graph workspace request_indices shape is incompatible with the batch bucket"
            )
        max_chunks_per_req = work_items_capacity // bs
        if max_chunks_per_req <= 0:
            raise RuntimeError("decode graph workspace must allocate at least one chunk per request")
        if self.cuda_graph_decode_max_chunks_per_req is None:
            raise RuntimeError("decode graph chunk-pages LUT is not initialized")
        if self.cuda_graph_decode_max_chunks_per_req > max_chunks_per_req:
            raise RuntimeError(
                "decode graph workspace capacity is too small for the current chunking policy"
            )

    def _run_decode_graph_metadata_update(
        self,
        workspace: PagedAttentionWorkspace,
        md: B12xForwardMetadata,
    ) -> None:
        if workspace.page_table is None:
            raise RuntimeError("decode graph workspace is missing page_table")
        if workspace.cache_seqlens is None:
            raise RuntimeError("decode graph workspace is missing cache_seqlens")
        if workspace.request_indices is None:
            raise RuntimeError("decode graph workspace is missing request indices")
        if workspace.block_valid_mask is None:
            raise RuntimeError("decode graph workspace is missing block_valid_mask")
        if workspace.merge_indptr is None or workspace.o_indptr is None:
            raise RuntimeError("decode graph workspace is missing indptr buffers")
        if workspace.kv_chunk_size_ptr is None:
            raise RuntimeError("decode graph workspace is missing kv_chunk_size_ptr")
        if md.req_pool_indices is None or md.decode_chunk_pages is None:
            raise RuntimeError("decode graph metadata is missing runtime graph inputs")
        if self.cuda_graph_decode_chunk_pages_lut is None:
            raise RuntimeError("decode graph chunk-pages LUT is not initialized")

        bs = int(workspace.cache_seqlens.shape[0])
        max_cache_pages = torch.div(
            workspace.cache_seqlens[:bs].amax() + (self.page_size - 1),
            self.page_size,
            rounding_mode="floor",
        ).clamp_(min=1, max=self.max_pages_per_req).to(torch.int64)
        md.decode_chunk_pages.copy_(
            torch.index_select(self.cuda_graph_decode_chunk_pages_lut, 0, max_cache_pages.view(1))
        )
        page_blocks = triton.cdiv(int(workspace.page_table.shape[1]), _B12X_DECODE_BLOCK_PAGES)
        build_b12x_decode_graph_page_table_triton[(bs, page_blocks)](
            self.req_to_token,
            md.req_pool_indices,
            workspace.page_table,
            self.req_to_token.stride(0),
            workspace.page_table.stride(0),
            workspace.page_table.shape[1],
            PAGE_SIZE=self.page_size,
            BLOCK_PAGES=_B12X_DECODE_BLOCK_PAGES,
        )

        workspace.block_valid_mask.zero_()
        workspace.merge_indptr.zero_()
        max_chunks_per_req = int(workspace.request_indices.shape[0]) // bs
        chunk_blocks = triton.cdiv(max_chunks_per_req, _B12X_DECODE_BLOCK_CHUNKS)
        update_b12x_decode_graph_metadata_triton[(bs, chunk_blocks)](
            workspace.cache_seqlens,
            workspace.merge_indptr,
            workspace.block_valid_mask,
            md.decode_chunk_pages,
            max_chunks_per_req,
            PAGE_SIZE=self.page_size,
            BLOCK_CHUNKS=_B12X_DECODE_BLOCK_CHUNKS,
        )
        torch.cumsum(
            workspace.merge_indptr[1 : bs + 1],
            dim=0,
            out=workspace.merge_indptr[1 : bs + 1],
        )
        workspace.o_indptr[: bs + 1].copy_(workspace.merge_indptr[: bs + 1])
        workspace.kv_chunk_size_ptr.copy_(md.decode_chunk_pages * self.page_size)

    def _get_descale_tensors(
        self,
        layer: RadixAttention,
        batch_size: int,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.kv_cache_dtype != torch.float8_e4m3fn:
            return None, None
        k_scale = 1.0 if layer.k_scale is None else float(layer.k_scale_float)
        v_scale = 1.0 if layer.v_scale is None else float(layer.v_scale_float)
        k_descale = torch.full(
            (batch_size, self.num_kv_heads),
            k_scale,
            dtype=torch.float32,
            device=self.device,
        )
        v_descale = torch.full(
            (batch_size, self.num_kv_heads),
            v_scale,
            dtype=torch.float32,
            device=self.device,
        )
        return k_descale, v_descale
