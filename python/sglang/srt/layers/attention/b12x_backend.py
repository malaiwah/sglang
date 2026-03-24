"""b12x SM120 paged attention backend for sglang."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from b12x.integration.attention import PagedAttentionWorkspace
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

_B12X_PAGE_SIZE = 64


@dataclass
class B12xForwardMetadata:
    cu_seqlens_q: torch.Tensor
    cache_seqlens: torch.Tensor
    page_table: torch.Tensor
    mode: str
    use_cuda_graph: bool
    graph_key: tuple[str, int] | None = None


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
        mode = self._mode_from_forward_mode(forward_mode)
        cache_seqlens, page_table, cu_seqlens_q = self._fill_cuda_graph_metadata(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            total_q_hint=num_tokens,
        )
        graph_key = (mode, bs)
        workspace = self._get_or_create_graph_workspace(graph_key, mode=mode, total_q_capacity=num_tokens)
        workspace.prepare(page_table, cache_seqlens, cu_seqlens_q)
        self.forward_metadata = B12xForwardMetadata(
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            mode=mode,
            use_cuda_graph=True,
            graph_key=graph_key,
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
        del seq_lens_sum, seq_lens_cpu
        assert encoder_lens is None, "b12x backend does not support encoder-decoder models"
        mode = self._mode_from_forward_mode(forward_mode)
        graph_key = (mode, bs)
        workspace = self.cuda_graph_workspaces.get(graph_key)
        if workspace is None:
            raise RuntimeError(f"missing captured b12x cuda-graph workspace for {graph_key}")
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
        max_cache_seqlens = torch.full(
            (bs,),
            self.max_context_len,
            dtype=torch.int32,
            device=self.device,
        )
        max_page_ids = (
            torch.arange(self.max_pages_per_req, dtype=torch.int32, device=self.device)
            % self.num_cache_pages
        )
        max_page_table = max_page_ids.unsqueeze(0).expand(bs, -1).contiguous()
        max_cu_seqlens_q = self._build_graph_capacity_cu_seqlens_q(
            mode=mode,
            bs=bs,
            total_q_capacity=total_q_capacity,
        )
        workspace.prepare(max_page_table, max_cache_seqlens, max_cu_seqlens_q)

    def _build_graph_capacity_cu_seqlens_q(
        self,
        *,
        mode: str,
        bs: int,
        total_q_capacity: int,
    ) -> torch.Tensor:
        if mode == "decode":
            return torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)

        if total_q_capacity < bs:
            raise ValueError(
                f"b12x extend graph bucket requires total_q_capacity >= bs, got {total_q_capacity} < {bs}"
            )
        q_lens = torch.ones(bs, dtype=torch.int32, device=self.device)
        q_lens[-1] = int(total_q_capacity - (bs - 1))
        cu_seqlens_q = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(q_lens, dim=0, out=cu_seqlens_q[1:])
        return cu_seqlens_q

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
