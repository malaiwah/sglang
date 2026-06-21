from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, TypeAlias

import torch

from sglang.srt.configs.model_config import get_nsa_index_topk, is_deepseek_nsa
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache_paged
from sglang.srt.layers.attention.nsa.nsa_backend_mtp_precompute import (
    NativeSparseAttnBackendMTPPrecomputeMixin,
    PrecomputedMetadata,
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache
from sglang.srt.layers.attention.nsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_prefill_cp_round_robin_split,
    compute_nsa_seqlens,
    is_nsa_enable_prefill_cp,
    nsa_cp_round_robin_split_data,
    nsa_cp_round_robin_split_q_seqs,
    pad_nsa_cache_seqlens,
)
from sglang.srt.layers.attention.utils import (
    concat_mla_absorb_q_general,
    mla_quantize_and_rope_for_fp8,
    seqlens_expand_triton,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


_is_hip = is_hip()

if _is_hip:
    from sglang.srt.layers.attention.nsa.triton_kernel import get_valid_kv_indices

    try:
        from aiter import (  # noqa: F401
            flash_attn_varlen_func,
            mha_batch_prefill_func,
            paged_attention_ragged,
        )
        from aiter.mla import mla_decode_fwd, mla_prefill_fwd  # noqa: F401
    except ImportError:
        print(
            "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
        )
else:
    from sglang.jit_kernel.flash_attention import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )


# Reuse this workspace buffer across all NSA backend instances
global_workspace_buffer = None

# Control whether to use fused metadata copy kernel for cuda graph replay (default: enabled)
# Set SGLANG_USE_FUSED_METADATA_COPY=0 or false to disable
_USE_FUSED_METADATA_COPY = envs.SGLANG_USE_FUSED_METADATA_COPY.get() and not _is_hip


def _get_b12x_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
) -> Optional[torch.Tensor]:
    try:
        from b12x.integration.nsa_indexer import get_paged_mqa_logits_metadata
    except (ImportError, ModuleNotFoundError):
        return None
    return get_paged_mqa_logits_metadata(context_lens, 64)


@dataclass(frozen=True)
class NSAFlashMLAMetadata:
    """Metadata only needed by FlashMLA"""

    flashmla_metadata: torch.Tensor
    num_splits: torch.Tensor

    def slice(self, sli):
        return NSAFlashMLAMetadata(
            flashmla_metadata=self.flashmla_metadata,
            num_splits=self.num_splits[sli],
        )

    def copy_(self, other: "NSAFlashMLAMetadata"):
        self.flashmla_metadata.copy_(other.flashmla_metadata)
        self.num_splits.copy_(other.num_splits)


@dataclass(frozen=True)
class NSAMetadata:
    page_size: int

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor
    # Maximum sequence length for query
    max_seq_len_q: int
    # Maximum sequence length for key
    max_seq_len_k: int
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor
    # Page table, the index of KV Cache Tables/Blocks
    # this table is always with page_size = 1
    page_table_1: torch.Tensor

    # NOTE(dark): This will property be used in:
    # 1. dense decode/prefill, we use paged flash attention, need real_page_table
    # 2. sparse decode/prefill, indexer need real_page_table to compute the score
    real_page_table: torch.Tensor

    # NSA metadata (nsa prefill are expanded)
    nsa_cache_seqlens_int32: torch.Tensor  # this seqlens is clipped to `topk`
    nsa_cu_seqlens_q: torch.Tensor  # must be arange(0, len(nsa_cu_seqlens_k))
    nsa_cu_seqlens_k: torch.Tensor  # cumsum of `nsa_cache_seqlens_int32`
    nsa_extend_seq_lens_list: List[int]
    nsa_seqlens_expanded: torch.Tensor  # expanded, unclipped `seqlens`
    nsa_max_seqlen_q: Literal[1] = 1  # always 1 for decode, variable for extend

    flashmla_metadata: Optional[NSAFlashMLAMetadata] = None
    # DeepGEMM schedule metadata for paged MQA logits (decode/target_verify/draft_extend only).
    # Precomputed once per forward batch and reused across layers.
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    # The sum of sequence lengths for key, prefill only
    seq_lens_sum: Optional[int] = None
    # The flattened 1D page table with shape (seq_lens_sum,), prefill only
    # this table is always with page_size = 1
    page_table_1_flattened: Optional[torch.Tensor] = None
    # The offset of topk indices in ragged kv, prefill only
    # shape: (seq_lens_sum,)
    topk_indices_offset: Optional[torch.Tensor] = None

    # k_start and k_end in kv cache for each token.
    indexer_k_start_end: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # seq lens for each batch.
    indexer_seq_lens_cpu: Optional[torch.Tensor] = None
    # seq lens for each batch.
    indexer_seq_lens: Optional[torch.Tensor] = None
    # batch index for each token.
    token_to_batch_idx: Optional[torch.Tensor] = None


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


@torch.compile
def _compiled_cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    return torch.cat(tensors, dim=dim)


def _cat(tensors: list[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Concatenate two tensors along the last dimension.
    Use this function to concatenate q_nope and q_rope or k_nope and k_rope.
    """
    assert len(tensors) == 2

    qk_nope, qk_rope = tensors
    assert qk_nope.ndim == 3 and qk_rope.ndim == 3

    torch._dynamo.mark_dynamic(qk_nope, 0)
    torch._dynamo.mark_dynamic(qk_rope, 0)

    return _compiled_cat([qk_nope, qk_rope], dim=dim)


@dataclass(frozen=True)
class NSAIndexerMetadata(BaseIndexerMetadata):
    attn_metadata: NSAMetadata
    topk_transform_method: TopkTransformMethod
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    force_unfused_topk: bool = False

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.nsa_seqlens_expanded

    def get_cu_seqlens_k(self) -> torch.Tensor:
        return self.attn_metadata.cu_seqlens_k

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn_metadata.indexer_k_start_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self.attn_metadata.indexer_seq_lens_cpu

    def get_nsa_extend_len_cpu(self) -> List[int]:
        return self.attn_metadata.nsa_extend_seq_lens_list

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.attn_metadata.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: Optional[torch.Tensor] = None,
        cu_seqlens_q: torch.Tensor = None,
        ke_offset: torch.Tensor = None,
        batch_idx_list: List[int] = None,
        topk_indices_offset_override: Optional[torch.Tensor] = None,
        page_table_1_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sgl_kernel import (
            fast_topk_transform_fused,
            fast_topk_transform_ragged_fused,
            fast_topk_v2,
        )

        if topk_indices_offset_override is not None:
            cu_topk_indices_offset = topk_indices_offset_override
            cu_seqlens_q_topk = None
        elif cu_seqlens_q is not None:
            cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            cu_seqlens_q_topk = compute_cu_seqlens(cu_seqlens_q)
            cu_topk_indices_offset = torch.repeat_interleave(
                cu_seqlens_q_topk[:-1],
                cu_seqlens_q,
            )
        else:
            cu_seqlens_q_topk = self.attn_metadata.cu_seqlens_q
            cu_topk_indices_offset = self.attn_metadata.topk_indices_offset
        if ke_offset is not None:
            seq_lens_topk = ke_offset
        else:
            seq_lens_topk = self.get_seqlens_expanded()
        if page_table_1_override is not None:
            # DCP Stage 2: rank-LOCAL token->slot table so selected columns map to LOCAL latent
            # slots (the indexer scored only this rank's owned pages). The ragged chunk path indexes
            # the per-sequence table by per-token batch idx, so honor batch_idx_list here too.
            page_table_size_1 = (
                page_table_1_override[batch_idx_list]
                if batch_idx_list is not None
                else page_table_1_override
            )
        elif batch_idx_list is not None:
            page_table_size_1 = self.attn_metadata.page_table_1[batch_idx_list]
        else:
            page_table_size_1 = self.attn_metadata.page_table_1

        if not envs.SGLANG_NSA_FUSE_TOPK.get() or self.force_unfused_topk:
            return fast_topk_v2(logits, seq_lens_topk, topk, row_starts=ks)
        elif self.topk_transform_method == TopkTransformMethod.PAGED:
            # NOTE(dark): if fused, we return a transformed page table directly
            return fast_topk_transform_fused(
                score=logits,
                lengths=seq_lens_topk,
                page_table_size_1=page_table_size_1,
                cu_seqlens_q=cu_seqlens_q_topk,
                topk=topk,
                row_starts=ks,
            )
        elif self.topk_transform_method == TopkTransformMethod.RAGGED:
            if cu_topk_indices_offset is None:
                raise RuntimeError(
                    "RAGGED topk_transform requires topk_indices_offset; "
                    "expected extend-without-speculative metadata."
                )
            return fast_topk_transform_ragged_fused(
                score=logits,
                lengths=seq_lens_topk,
                topk_indices_offset=cu_topk_indices_offset,
                topk=topk,
                row_starts=ks,
            )
        else:
            assert False, f"Unsupported {self.topk_transform_method = }"


_NSA_IMPL_T: TypeAlias = Literal[
    "flashmla_sparse", "flashmla_kv", "b12x", "fa3", "tilelang", "trtllm"
]


class NativeSparseAttnBackend(
    NativeSparseAttnBackendMTPPrecomputeMixin, AttentionBackend
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.forward_metadata: NSAMetadata
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        self.real_page_size = model_runner.page_size
        self.num_splits = (
            1 if model_runner.server_args.enable_deterministic_inference else 0
        )
        self.use_nsa = is_deepseek_nsa(model_runner.model_config.hf_config)
        assert self.use_nsa, "NSA backend only supports DeepSeek NSA"
        self.nsa_kv_cache_store_fp8 = (
            model_runner.token_to_kv_pool.nsa_kv_cache_store_fp8
        )
        self.nsa_index_topk = get_nsa_index_topk(model_runner.model_config.hf_config)
        self.max_context_len = model_runner.model_config.context_len
        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.kv_cache_dim = model_runner.token_to_kv_pool.kv_cache_dim
        self.qk_nope_head_dim = model_runner.model_config.qk_nope_head_dim
        self.kv_lora_rank = model_runner.model_config.kv_lora_rank
        self.qk_rope_head_dim = model_runner.model_config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.model_v_head_dim = getattr(
            model_runner.model_config, "v_head_dim", self.qk_head_dim
        )
        self.q_dtype = model_runner.dtype

        # --- vLLM-style decode-context-parallel (DCP) ---------------------------------
        # Shard the latent KV pool across the attention-TP ranks while KEEPING attn_tp
        # (heads stay TP-sharded) — the vLLM mechanism, which avoids SGLang attn_cp's
        # attn_tp=1 weight-replication tax. DCP group == the attention-TP group
        # (dcp_size == attn_tp). Env-gated; non-CP path byte-identical when off.
        # Decode flow (validated tpxdcp_decode_probe.py, cos 0.999994):
        #   all_gather q over heads -> each rank decodes ALL heads over its OWN KV shard
        #   (return_lse) -> LSE-merge across ranks -> reduce-scatter heads back to attn_tp.
        import os as _os_dcp
        self.dcp_enabled = _os_dcp.environ.get("SGLANG_NSA_DECODE_DCP", "0") not in (
            "0", "", "false", "False",
        )
        self.dcp_size = get_attention_tp_size() if self.dcp_enabled else 1
        self.dcp_rank = 0
        self.dcp_group = None
        if self.dcp_enabled and self.dcp_size > 1:
            from sglang.srt.layers.dp_attention import (
                get_attention_tp_group as _gatg,
                get_attention_tp_rank as _gatr,
            )
            self.dcp_rank = _gatr()
            self.dcp_attn_tp_group = _gatg()
            self.dcp_group = getattr(_gatg(), "device_group", None)
            # cuda-graph-safe constant: the merge's num_chunks_ptr is just [dcp]. Build it ONCE
            # here on-device — a per-call torch.tensor([cp], device=cuda) is a host->device copy
            # that is ILLEGAL during cuda-graph capture (the capture blocker we hit).
            self._cp_num_chunks = torch.tensor(
                [self.dcp_size], device=model_runner.device, dtype=torch.int32
            )
        # Step B (default): the latent pool is physically sharded /dcp -> per-rank-local
        # slot remap in decode. Step A (=0): replicated pool, decode-only correctness A/B.
        self.dcp_shard_pool = self.dcp_enabled and _os_dcp.environ.get(
            "SGLANG_NSA_DCP_SHARD_POOL", "1"
        ) not in ("0", "", "false", "False")
        # DCP Stage 2: also shard the indexer index_k pool /dcp by page + per-rank-local top-k
        # (fixes the O(context^2) replicated-indexer prefill tail AND unlocks ~4x/809k capacity).
        # Default OFF until end-to-end validated; must match memory_pool / cell-sizer gating.
        self.dcp_shard_index = self.dcp_shard_pool and _os_dcp.environ.get(
            "SGLANG_NSA_DCP_SHARD_INDEX", "0"
        ) not in ("0", "", "false", "False")
        # DCP merge collective: default merge_cp_reduce_scatter (all_gather(LSE)+reduce_scatter).
        # When ON, use merge_cp_a2a (single all-to-all of out||lse + local online-softmax) — fewer
        # PCIe collectives per attention layer (the no-NVLink decode-throughput win, vLLM a2a path).
        # Default OFF so the merge can be A/B'd at the same DCP layout. Same result either way.
        self.dcp_a2a_merge = self.dcp_enabled and _os_dcp.environ.get(
            "SGLANG_NSA_DCP_A2A_MERGE", "0"
        ) not in ("0", "", "false", "False")
        # DCP decode merge via the b12x FUSED kernel (merge_cp_decode_output ->
        # run_sparse_mla_split_decode_merge): all_gather the partials + ONE fused online-softmax
        # kernel (vs ~6 torch ops in reduce_scatter). For DECODE (bs=1, tiny tensors) the all_gather
        # is cheap and the fused kernel cuts per-layer kernel launches -> candidate decode win.
        # Default OFF (A/B). Prefill keeps reduce_scatter (4x less traffic on big tensors).
        self.dcp_fused_decode_merge = self.dcp_enabled and _os_dcp.environ.get(
            "SGLANG_NSA_DCP_FUSED_DECODE", "0"
        ) not in ("0", "", "false", "False")
        # DCP merge via the vLLM FUSED correct-attn path (merge_cp_correct_rs): all_gather ONLY
        # the tiny LSE via single-buffer all_gather_into_tensor + ONE fused Triton kernel
        # (global-LSE + in-place exp2 rescale) + single-buffer reduce_scatter_tensor. Closes the
        # decode gap vs vLLM (ours 30 Stage-2 / 36 Stage-1 vs 45): same 3 collectives/layer but
        # drops the ~8-10 eager torch kernels of reduce_scatter AND the list-collective copies.
        # Same online-softmax (base-2) result as merge_cp_reduce_scatter; used for BOTH decode and
        # extend when ON. Default OFF (A/B). Takes precedence over a2a when both are set.
        self.dcp_correct_merge = self.dcp_enabled and _os_dcp.environ.get(
            "SGLANG_NSA_DCP_CORRECT_MERGE", "0"
        ) not in ("0", "", "false", "False")

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mha: bool = False
        self.nsa_prefill_impl: _NSA_IMPL_T = (
            model_runner.server_args.nsa_prefill_backend
        )
        self.nsa_decode_impl: _NSA_IMPL_T = model_runner.server_args.nsa_decode_backend
        self.enable_auto_select_prefill_impl = self.nsa_prefill_impl == "flashmla_auto"

        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        if _is_hip:
            max_bs = model_runner.req_to_token_pool.size

            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.kv_indices = torch.zeros(
                max_bs * self.nsa_index_topk,
                dtype=torch.int32,
                device=self.device,
            )

        # Speculative decoding
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        self.device_capability = torch.cuda.get_device_capability()
        self.device_sm_major = self.device_capability[0]
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.b12x_workspaces: Dict[str, object] = {}

        # Allocate global workspace buffer for TRT-LLM kernels (ragged attention on SM100/B200, or trtllm decode)
        if self.device_sm_major >= 10 or self.nsa_decode_impl == "trtllm":
            global global_workspace_buffer
            if global_workspace_buffer is None:
                global_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
            self.workspace_buffer = global_workspace_buffer
        else:
            self.workspace_buffer = None

        if self.nsa_prefill_impl == "b12x" or self.nsa_decode_impl == "b12x":
            self._validate_b12x_contract(model_runner)

    def get_device_int32_arange(self, l: int) -> torch.Tensor:
        if l > len(self._arange_buf):
            next_pow_of_2 = 1 << (l - 1).bit_length()
            self._arange_buf = torch.arange(
                next_pow_of_2, device=self.device, dtype=torch.int32
            )
        return self._arange_buf[:l]

    def _validate_b12x_contract(self, model_runner: ModelRunner) -> None:
        model_arch = model_runner.model_config.hf_config.architectures[0]
        model_type = getattr(model_runner.model_config.hf_config, "model_type", None)
        if model_type != "glm_moe_dsa":
            raise ValueError(
                "b12x only supports the GLM DSA model family in v1 "
                f"(model_type=glm_moe_dsa); got architecture={model_arch}, "
                f"model_type={model_type}."
            )
        if self.device_capability != (12, 0):
            raise ValueError(
                f"b12x requires SM120, got sm_{self.device_capability[0]}{self.device_capability[1]}."
            )
        if self.real_page_size != 64:
            raise ValueError(
                f"b12x requires page_size=64, got {self.real_page_size}."
            )
        if self.kv_cache_dtype != torch.float8_e4m3fn or not self.nsa_kv_cache_store_fp8:
            raise ValueError("b12x requires FP8 NSA KV cache storage.")
        if (
            self.qk_nope_head_dim != 192
            or self.kv_lora_rank != 512
            or self.qk_rope_head_dim != 64
            or self.model_v_head_dim != 256
            or self.nsa_index_topk != 2048
        ):
            raise ValueError(
                "b12x v1 only supports the GLM-5.1 MLA geometry "
                "(qk_nope=192, kv_lora_rank=512, qk_rope=64, v=256, topk=2048)."
            )
        if model_runner.server_args.enable_nsa_prefill_context_parallel:
            raise ValueError("b12x does not support NSA context parallel in v1.")
        if getattr(model_runner, "enable_hisparse", False):
            raise ValueError("b12x does not support HiSparse in v1.")

    def _get_b12x_workspace(
        self,
        *,
        mode: str,
        total_q: int,
        batch: int,
        v_head_dim: int,
        num_q_heads_override: int = 0,
    ):
        # b12x 0.20.0 removed MLAWorkspace.for_fixed_capacity; the new lifecycle is
        # caps -> plan_sparse_mla_scratch -> allocate a scratch buffer -> plan.bind().
        # We cache a {plan, buf} holder per mode, grown to the largest capacity seen.
        from b12x.integration.sparse_mla_scratch import (
            B12XSparseMLAScratchCaps,
            plan_sparse_mla_scratch,
        )

        workspace_mode = "verify" if mode == "target_verify" else mode
        _nqh = num_q_heads_override or self.num_q_heads
        # cache key carries the head count (all-H DCP scratch is separate); the b12x caps
        # `mode=` MUST stay a valid mode string ("decode"/"extend"/"verify").
        holder_key = workspace_mode if _nqh == self.num_q_heads else f"{workspace_mode}_h{_nqh}"
        holder = self.b12x_workspaces.get(holder_key)
        if (
            holder is not None
            and holder["max_total_q"] >= total_q
            and holder["max_batch"] >= batch
        ):
            return holder

        if holder is not None:
            total_q = max(total_q, holder["max_total_q"])
            batch = max(batch, holder["max_batch"])

        caps = B12XSparseMLAScratchCaps(
            device=self.device,
            num_q_heads=_nqh,
            max_q_rows=max(total_q, 1),
            max_width=self.nsa_index_topk,
            dtype=self.q_dtype,
            kv_dtype=self.kv_cache_dtype,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mode=workspace_mode,
            max_batch=max(batch, 1),
            page_size=self.real_page_size,
        )
        plan = plan_sparse_mla_scratch(caps)
        _shape, _dt = plan.shapes_and_dtypes()[0]
        _buf = torch.empty(_shape, dtype=_dt, device=self.device)
        holder = {
            "plan": plan,
            "buf": _buf,
            "max_total_q": max(total_q, 1),
            "max_batch": max(batch, 1),
        }
        self.b12x_workspaces[holder_key] = holder
        return holder

    def _forward_b12x(
        self,
        *,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        metadata: NSAMetadata,
        sm_scale: float,
        v_head_dim: int,
        mode: Literal["decode", "extend", "target_verify", "draft_extend"],
    ) -> torch.Tensor:
        from b12x.integration.mla import (
            MLASparseDecodeMetadata,
            MLASparseExtendMetadata,
            sparse_mla_decode_forward,
            sparse_mla_extend_forward,
        )

        workspace = self._get_b12x_workspace(
            mode=mode,
            total_q=q_all.shape[0],
            batch=metadata.cache_seqlens_int32.shape[0],
            v_head_dim=v_head_dim,
        )
        if mode == "decode":
            # GLM-OPTIONB DEBUG (2026-06-19): one-shot dump of real decode inputs.
            import os as _os
            cs = metadata.cache_seqlens_int32
            if (
                _os.environ.get("SGLANG_NSA_DEBUG_DUMP")
                and getattr(type(self), "_dbg_decode_dumped", 0) < 4
                and not torch.cuda.is_current_stream_capturing()
                and int(cs.max()) > 2
            ):
                try:
                    pt = page_table_1
                    nsa = metadata.nsa_cache_seqlens_int32
                    with open("/root/.cache/huggingface/nsa_dump.txt", "a") as _f:
                        _f.write(
                            f"[decode] q_all {tuple(q_all.shape)} {q_all.dtype} "
                            f"norm={q_all.float().norm().item():.3f} "
                            f"nan={bool(q_all.isnan().any())}\n"
                            f"  kv_cache {tuple(kv_cache.shape)} {kv_cache.dtype}\n"
                            f"  page_table_1(selected) {tuple(pt.shape)} {pt.dtype} "
                            f"min={int(pt.min())} max={int(pt.max())} "
                            f"row0[:20]={pt[0][:20].tolist()}\n"
                            f"  cache_seqlens {cs.tolist()[:8]} "
                            f"nsa_cache_seqlens {nsa.tolist()[:8]}\n"
                        )
                    type(self)._dbg_decode_dumped = (
                        getattr(type(self), "_dbg_decode_dumped", 0) + 1
                    )
                except Exception as _e:
                    pass
            # --- decode context-parallel (DCP) -------------------------------------
            # When --attn-cp-size>1 (attn_tp=1 layout), shard the selected (top-k)
            # tokens across the attn-cp ranks, partial-decode each shard with
            # return_lse, and LSE-merge the per-rank partials into the exact global
            # output. Active only when cp_size>1; byte-identical to the legacy path
            # otherwise. SGLANG_NSA_DECODE_CP=0 forces the full per-rank decode (for
            # A/B isolation of the merge at the same attn_tp=1 layout).
            # vLLM-style DCP (keep attn_tp): shard the latent KV pool across the TP ranks,
            # all-gather q over heads, decode all-H over the owned shard, merge, head-slice.
            # Takes precedence over the legacy attn_cp path when SGLANG_NSA_DECODE_DCP is set.
            if self.dcp_enabled and self.dcp_size > 1:
                return self._decode_dcp(
                    q_all=q_all,
                    kv_cache=kv_cache,
                    page_table_1=page_table_1,
                    metadata=metadata,
                    sm_scale=sm_scale,
                    v_head_dim=v_head_dim,
                )
            import os as _os_cp
            try:
                from sglang.srt.layers.dp_attention import (
                    get_attention_cp_size as _gcs,
                )

                _cp_size = _gcs()
            except Exception:
                _cp_size = 1
            if _cp_size > 1 and _os_cp.environ.get(
                "SGLANG_NSA_DECODE_CP", "1"
            ) not in ("0", "", "false", "False"):
                return self._decode_cp(
                    q_all=q_all,
                    kv_cache=kv_cache,
                    page_table_1=page_table_1,
                    metadata=metadata,
                    sm_scale=sm_scale,
                    v_head_dim=v_head_dim,
                    workspace=workspace,
                    cp_size=_cp_size,
                )
            # b12x 0.20.0: flattened API — bind tensors to the scratch plan, then
            # pass the binding (no MLASparseDecodeMetadata / workspace= anymore).
            binding = workspace["plan"].bind(
                scratch=workspace["buf"],
                q=q_all,
                selected_indices=page_table_1,
                cache_seqlens_int32=metadata.cache_seqlens_int32,
                nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
            )
            _o_dec = sparse_mla_decode_forward(
                kv_cache=kv_cache,
                binding=binding,
                sm_scale=sm_scale,
                v_head_dim=v_head_dim,
            )
            if (
                _os.environ.get("SGLANG_NSA_DEBUG_SAVE")
                and not getattr(type(self), "_dbg_saved", 0)
                and not torch.cuda.is_current_stream_capturing()
                and int(cs.max()) > 2
            ):
                try:
                    _valid = page_table_1[0][page_table_1[0] >= 0]
                    torch.save(
                        {
                            "q_all": q_all.detach().cpu(),
                            "kv_rows": kv_cache[_valid.long()].detach().cpu(),
                            "valid_slots": _valid.detach().cpu(),
                            "selected": page_table_1.detach().cpu(),
                            "cache_seqlens": cs.detach().cpu(),
                            "nsa": metadata.nsa_cache_seqlens_int32.detach().cpu(),
                            "O": _o_dec.detach().cpu(),
                            "sm_scale": float(sm_scale),
                            "v_head_dim": int(v_head_dim),
                        },
                        "/root/.cache/huggingface/real_decode.pt",
                    )
                    type(self)._dbg_saved = 1
                except Exception:
                    pass
            return _o_dec

        # vLLM-style DCP for EXTEND/prefill: each rank extends over its OWNED selected KV shard
        # (return_lse), then LSE-merge across ranks + head reduce-scatter — required so the
        # prefill attention is correct over the /dcp-sharded latent pool.
        if self.dcp_enabled and self.dcp_size > 1:
            return self._extend_dcp(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                metadata=metadata,
                sm_scale=sm_scale,
                v_head_dim=v_head_dim,
            )
        # b12x 0.20.0: extend resolves the binding exactly like decode —
        # binding.selected_indices is mapped to `selected_token_offsets` inside
        # sparse_mla_extend_forward (_resolve_sparse_mla_binding, selected_name=
        # "selected_token_offsets"). So the same plan.bind() as decode applies; the
        # per-query topk selection lives in page_table_1.
        binding = workspace["plan"].bind(
            scratch=workspace["buf"],
            q=q_all,
            selected_indices=page_table_1,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
        )
        return sparse_mla_extend_forward(
            kv_cache=kv_cache,
            binding=binding,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )

    def _decode_cp(
        self,
        *,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        metadata: NSAMetadata,
        sm_scale: float,
        v_head_dim: int,
        workspace: dict,
        cp_size: int,
    ) -> torch.Tensor:
        """Decode-context-parallel: each attn-cp rank attends to the subset of the
        selected (top-k) tokens it owns; the per-rank partial (out, lse) are then
        all-gathered and LSE-merged into the EXACT global sparse-MLA output.

        M1 (capacity-neutral): the KV pool is still replicated, so page_table_1 is
        identical on every rank and we partition it by COLUMN (rank r owns selected
        columns r::cp_size — a front-valid prefix per row). M2 will instead shard the
        KV pool by token ownership, making this the natural per-rank-local selection.

        Validated standalone (b12x_cp_merge_probe.py) and across 4 NCCL ranks
        (cp_nsa_dist_test.py). The cross-rank merge reuses b12x's standalone
        run_sparse_mla_split_decode_merge via cp_nsa.merge_cp_decode_output.
        """
        import torch.nn.functional as _F
        from b12x.integration.mla import sparse_mla_decode_forward
        from sglang.srt.layers.attention.nsa.cp_nsa import merge_cp_decode_output
        from sglang.srt.layers.dp_attention import (
            get_attention_cp_group,
            get_attention_cp_rank,
        )

        cp_rank = get_attention_cp_rank()
        pg = getattr(get_attention_cp_group(), "device_group", None)
        dev = page_table_1.device
        W = page_table_1.shape[1]
        nsa = metadata.nsa_cache_seqlens_int32  # [batch] valid selected count per row

        # This rank owns selected columns cp_rank::cp_size. Because orig_col grows
        # monotonically along the stride, the owned valid entries are a prefix.
        owned_slice = page_table_1[:, cp_rank::cp_size]  # [batch, Wr]
        Wr = owned_slice.shape[1]
        orig_col = cp_rank + torch.arange(Wr, device=dev, dtype=torch.int64) * cp_size
        owned_nsa = (
            (orig_col.unsqueeze(0) < nsa.unsqueeze(1).to(torch.int64)).sum(dim=1).to(torch.int32)
        )
        # Pad owned columns back to the planned width W (kernel reads owned_nsa from front).
        owned_pt = (
            _F.pad(owned_slice, (0, W - Wr), value=-1).contiguous()
            if Wr < W
            else owned_slice.contiguous()
        )

        # Empty-owner rows: feed 1 dummy entry so the kernel stays valid, then mark the
        # row's lse = -inf so the merge skips this rank's contribution for that row.
        empty = owned_nsa == 0
        safe_nsa = torch.where(empty, torch.ones_like(owned_nsa), owned_nsa)

        binding = workspace["plan"].bind(
            scratch=workspace["buf"],
            q=q_all,
            selected_indices=owned_pt,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=safe_nsa,
        )
        o_r, lse_r = sparse_mla_decode_forward(
            kv_cache=kv_cache,
            binding=binding,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
            return_lse=True,
            lse_scale="base2",
        )
        rows = q_all.shape[0]
        o_r = o_r.reshape(rows, self.num_q_heads, v_head_dim)
        lse_r = lse_r.reshape(rows, self.num_q_heads).float()
        lse_r[empty] = float("-inf")  # unconditional masked write — no host sync (perf #1)
        return merge_cp_decode_output(o_r, lse_r, cp_group=pg)

    def _decode_dcp(
        self,
        *,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        metadata: NSAMetadata,
        sm_scale: float,
        v_head_dim: int,
    ) -> torch.Tensor:
        """vLLM-style decode context-parallel (KEEP attn_tp). `q_all` is THIS rank's TP head
        shard [rows, num_q_heads, head_dim]; the latent KV pool is sharded /dcp (Step B) or
        replicated (Step A). Flow (validated tpxdcp_decode_probe.py, cos 0.999994):
          all_gather q over heads -> decode ALL heads over this rank's OWNED selected slots
          (page-ownership) with return_lse -> LSE-merge across dcp -> slice this rank's heads.
        """
        import torch.nn.functional as _F
        from b12x.integration.mla import sparse_mla_decode_forward
        from sglang.srt.layers.attention.nsa.cp_nsa import (
            merge_cp_a2a,
            merge_cp_correct_rs,
            merge_cp_decode_output,
            merge_cp_reduce_scatter,
            page_owned_local_selection,
        )

        dcp = self.dcp_size
        rank = self.dcp_rank
        pg = self.dcp_group
        rows = q_all.shape[0]
        h_local = q_all.shape[1]

        # 1) all-gather q over the head axis -> all-H query replicated on every rank
        q_gathered = self.dcp_attn_tp_group.all_gather(q_all.contiguous(), dim=1)
        h_all = q_gathered.shape[1]  # == h_local * dcp == num_attention_heads

        # 2) page-level owned selection (+ global->local slot remap when pool is sharded)
        # DCP Stage 2 now produces a GLOBAL top-k (global KV slots, identical on every rank) just
        # like Stage 1 (the indexer merges its rank-local top-k via a candidate all-gather), so BOTH
        # stages take the same carve: page_owned_local_selection slices this rank's owned pages and
        # (when the latent pool is sharded) remaps global->local slots. nsa_cache_seqlens_int32 is
        # the global valid count (min(seqlen, topk)); the (ar<nsa)&(pt>=0) guard inside tolerates
        # rows with fewer than nsa valid global-topk entries.
        nsa = metadata.nsa_cache_seqlens_int32
        owned_pt, owned_cnt = page_owned_local_selection(
            page_table_1, nsa, rank, dcp, self.real_page_size,
            remap_local=self.dcp_shard_pool,
        )
        # owned_pt is already [rows, W] and contiguous (page_owned_local_selection gathers over
        # the full W columns and returns .contiguous()): the old pad branch was dead and the extra
        # .contiguous() a redundant rows×W copy/layer. Dropped (perf #4).
        empty = owned_cnt == 0
        # Empty-owner rows read dummy slot 0 (valid), discarded via lse=-inf after step 3. The
        # masked write is a no-op when no row is empty, so do it UNCONDITIONALLY: the old
        # `if bool(empty.any())` forced a GPU->CPU sync (pipeline stall) every layer (perf #1).
        owned_pt[empty, 0] = 0
        safe_cnt = torch.where(empty, torch.ones_like(owned_cnt), owned_cnt)

        # 3) decode ALL heads over this rank's owned KV shard (all-H scratch), return LSE
        ws = self._get_b12x_workspace(
            mode="decode",
            total_q=rows,
            batch=metadata.cache_seqlens_int32.shape[0],
            v_head_dim=v_head_dim,
            num_q_heads_override=h_all,
        )
        binding = ws["plan"].bind(
            scratch=ws["buf"],
            q=q_gathered,
            selected_indices=owned_pt,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=safe_cnt,
        )
        o_r, lse_r = sparse_mla_decode_forward(
            kv_cache=kv_cache,
            binding=binding,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
            return_lse=True,
            lse_scale="base2",
        )
        o_r = o_r.reshape(rows, h_all, v_head_dim)
        lse_r = lse_r.reshape(rows, h_all).float()
        lse_r[empty] = float("-inf")  # unconditional masked write — no host sync (perf #1)

        # 4) LSE-merge the per-rank all-H partials -> exact global all-H attention
        # 4+5) merge + reduce-scatter in one (vLLM cp_lse_ag_out_rs): each rank gets its TP
        # heads directly, moving cp× less data than the all-gather merge. When the a2a gate is on,
        # use the single all-to-all merge (vLLM dcp_a2a_lse_reduce) — same result, fewer PCIe
        # collectives per layer (the no-NVLink decode win).
        if self.dcp_correct_merge:
            # vLLM fused correct-attn: all_gather_into_tensor(LSE) + 1 Triton kernel + reduce_scatter_tensor.
            return merge_cp_correct_rs(o_r, lse_r, cp_group=pg, rank=rank, h_local=h_local)
        if self.dcp_fused_decode_merge:
            # all_gather + b12x fused split-merge kernel -> [rows, h_all, V]; slice this rank's heads.
            merged = merge_cp_decode_output(
                o_r, lse_r, cp_group=pg, num_chunks=self._cp_num_chunks
            )
            return merged[:, rank * h_local : (rank + 1) * h_local, :].contiguous()
        if self.dcp_a2a_merge:
            return merge_cp_a2a(o_r, lse_r, cp_group=pg, rank=rank, h_local=h_local)
        return merge_cp_reduce_scatter(o_r, lse_r, cp_group=pg, rank=rank, h_local=h_local)

    def _extend_dcp(
        self,
        *,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        metadata: NSAMetadata,
        sm_scale: float,
        v_head_dim: int,
    ) -> torch.Tensor:
        """vLLM-style DCP for the EXTEND/prefill path. Mirrors _decode_dcp but uses the b12x
        extend kernel; rows = total_q query tokens. Each rank extends ALL heads over its OWNED
        selected KV shard (page-ownership, return_lse) -> LSE-merge across dcp -> head slice.
        """
        import torch.nn.functional as _F
        from b12x.integration.mla import sparse_mla_extend_forward
        from sglang.srt.layers.attention.nsa.cp_nsa import (
            merge_cp_a2a,
            merge_cp_correct_rs,
            merge_cp_decode_output,
            merge_cp_reduce_scatter,
            page_owned_local_selection,
        )

        dcp = self.dcp_size
        rank = self.dcp_rank
        pg = self.dcp_group
        rows = q_all.shape[0]
        h_local = q_all.shape[1]

        q_gathered = self.dcp_attn_tp_group.all_gather(q_all.contiguous(), dim=1)
        h_all = q_gathered.shape[1]

        nsa = metadata.nsa_cache_seqlens_int32
        # DCP Stage 2: page_table_1 is now a GLOBAL top-k (the skip path returns global dense slots;
        # the ragged shard branch merges its rank-local top-k into a global one — same as decode), so
        # BOTH stages take Stage-1's carve. page_owned_local_selection slices this rank's owned pages
        # and (sharded pool) remaps global->local slots. (Was: a per-rank-local bypass that treated
        # the global slots as local -> corrupted the prompt encoding -> garbage decode.)
        owned_pt, owned_cnt = page_owned_local_selection(
            page_table_1, nsa, rank, dcp, self.real_page_size,
            remap_local=self.dcp_shard_pool,
        )
        # owned_pt already [rows, W] + contiguous (see _decode_dcp): pad branch dead, extra
        # .contiguous() redundant -> dropped (perf #4).
        empty = owned_cnt == 0
        # Unconditional masked write (no-op when no empty row) — no GPU->CPU sync/layer (perf #1).
        owned_pt[empty, 0] = 0
        safe_cnt = torch.where(empty, torch.ones_like(owned_cnt), owned_cnt)

        ws = self._get_b12x_workspace(
            mode="extend",
            total_q=rows,
            batch=metadata.cache_seqlens_int32.shape[0],
            v_head_dim=v_head_dim,
            num_q_heads_override=h_all,
        )
        binding = ws["plan"].bind(
            scratch=ws["buf"],
            q=q_gathered,
            selected_indices=owned_pt,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            nsa_cache_seqlens_int32=safe_cnt,
        )
        o_r, lse_r = sparse_mla_extend_forward(
            kv_cache=kv_cache,
            binding=binding,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
            return_lse=True,
            lse_scale="base2",
        )
        o_r = o_r.reshape(rows, h_all, v_head_dim)
        lse_r = lse_r.reshape(rows, h_all).float()
        lse_r[empty] = float("-inf")  # unconditional masked write — no host sync (perf #1)
        # vLLM-style merge: all-gather LSE (tiny) + reduce_scatter the weighted output. THIS is
        # the prefill fix — 1024-row merge traffic drops 4× + no stack/merge-kernel per layer.
        # correct gate: vLLM fused correct-attn (single-buffer collectives + 1 Triton kernel),
        # same result, fewer eager kernels/list-copies (Stage-1 prefill candidate too).
        # a2a gate: single all-to-all merge (vLLM dcp_a2a_lse_reduce), same result, fewer collectives.
        if self.dcp_correct_merge:
            return merge_cp_correct_rs(o_r, lse_r, cp_group=pg, rank=rank, h_local=h_local)
        if self.dcp_a2a_merge:
            return merge_cp_a2a(o_r, lse_r, cp_group=pg, rank=rank, h_local=h_local)
        return merge_cp_reduce_scatter(o_r, lse_r, cp_group=pg, rank=rank, h_local=h_local)

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.real_page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = self.speculative_num_draft_tokens
        else:
            draft_token_num = 0

        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        assert forward_batch.seq_lens_cpu is not None
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item() + draft_token_num)
        # [b, max_seqlen_k]
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        page_table_1_flattened = None
        topk_indices_offset = None

        # Centralized dispatch: decide all strategies for this batch
        self.set_nsa_prefill_impl(forward_batch)
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        # Batch indices selected when cp enabled: After splitting multiple sequences,
        # a certain cp rank may not have some of these sequences.
        # We use bs_idx_cpu to mark which sequences are finally selected by the current cp rank,
        # a default value of None indicates that all sequences are selected.
        bs_idx_cpu = None
        # seq_len_cpu of selected sequences
        indexer_seq_lens_cpu = forward_batch.seq_lens_cpu
        indexer_seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_target_verify():
            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                batch_size * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
            forward_batch.extend_seq_lens_cpu = extend_seq_lens_cpu

            seqlens_expanded = seqlens_expand_triton(
                torch.tensor(extend_seq_lens_cpu, dtype=torch.int32, device=device),
                cache_seqlens_int32,
                self.speculative_num_draft_tokens * batch_size,
                self.speculative_num_draft_tokens,
            )
            page_table = torch.repeat_interleave(
                page_table, repeats=self.speculative_num_draft_tokens, dim=0
            )
        elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"

            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None

            max_seqlen_q = 1
            cu_seqlens_q = torch.arange(
                0,
                forward_batch.extend_num_tokens + 1,
                1,
                dtype=torch.int32,
                device=device,
            )

            seqlens_expanded = seqlens_expand_triton(
                forward_batch.extend_seq_lens,
                cache_seqlens_int32,
                sum(extend_seq_lens_cpu),
                self.speculative_num_draft_tokens,
            )
            if forward_batch.forward_mode.is_draft_extend_v2():
                # DRAFT_EXTEND_V2: V2 worker pre-fills draft KV cache with ALL speculated
                # tokens upfront. All requests extend by the same fixed
                # (speculative_num_draft_tokens). Use scalar to avoid GPU sync.
                page_table = torch.repeat_interleave(
                    page_table, repeats=self.speculative_num_draft_tokens, dim=0
                )
            else:
                # DRAFT_EXTEND (v1): V1 worker extends by (accept_length + 1) per request
                # after verification. Lengths vary per request based on how many tokens
                # were accepted.
                page_table = torch.repeat_interleave(
                    page_table, repeats=forward_batch.extend_seq_lens, dim=0
                )
        elif forward_batch.forward_mode.is_extend():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
                and forward_batch.extend_prefix_lens_cpu is not None
            ), "All of them must not be None"
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            assert forward_batch.extend_seq_lens is not None
            extend_seq_lens = forward_batch.extend_seq_lens

            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )

            if can_nsa_prefill_cp_round_robin_split(forward_batch):
                seqlens_expanded = nsa_cp_round_robin_split_data(seqlens_expanded)
                extend_seq_lens_cpu, extend_seq_lens, bs_idx_cpu, bs_idx = (
                    nsa_cp_round_robin_split_q_seqs(
                        extend_seq_lens_cpu, extend_seq_lens
                    )
                )
                indexer_seq_lens_cpu = indexer_seq_lens_cpu[bs_idx_cpu]
                indexer_seq_lens = indexer_seq_lens[bs_idx]
                cache_seqlens_int32 = cache_seqlens_int32[bs_idx]
                cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
                max_seqlen_k = (
                    int(indexer_seq_lens_cpu.max().item() + draft_token_num)
                    if len(indexer_seq_lens_cpu) != 0
                    else 0
                )
                page_table = page_table[bs_idx, :max_seqlen_k]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
                or bs_idx_cpu is not None
            ):
                max_seqlen_q = (
                    max(extend_seq_lens_cpu) if len(extend_seq_lens_cpu) != 0 else 1
                )
                cu_seqlens_q = compute_cu_seqlens(extend_seq_lens.to(torch.int32))
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k

            # Check if MHA FP8 dequantization is needed
            mha_dequantize_needed = (
                self.use_mha
                and forward_batch.token_to_kv_pool.dtype == torch.float8_e4m3fn
            )
            forward_batch.using_mha_one_shot_fp8_dequant = mha_dequantize_needed

            # page_table_1_flattened is only used when prefix sharing is enabled:
            has_prefix_sharing = any(forward_batch.extend_prefix_lens_cpu)
            if has_prefix_sharing and (
                topk_transform_method == TopkTransformMethod.RAGGED
                or mha_dequantize_needed
            ):
                page_table_1_flattened = torch.cat(
                    [
                        page_table[i, :kv_len]
                        for i, kv_len in enumerate(
                            indexer_seq_lens_cpu.tolist(),
                        )
                    ]
                )
                assert page_table_1_flattened.shape[0] == sum(
                    indexer_seq_lens_cpu
                ), f"{page_table_1_flattened.shape[0] = } must be the same as {sum(indexer_seq_lens_cpu) = }"

                # Validate indices when logical tokens exceed physical capacity
                # This is likely to be triggered by PP with high kv reuse & parallelism
                kv_cache_capacity = (
                    forward_batch.token_to_kv_pool.size
                    + forward_batch.token_to_kv_pool.page_size
                )
                if forward_batch.seq_lens_sum > kv_cache_capacity:
                    max_idx = page_table_1_flattened.max().item()
                    assert max_idx < kv_cache_capacity, (
                        f"Invalid page table index: max={max_idx}, "
                        f"kv_cache_capacity={kv_cache_capacity}"
                    )

            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = torch.repeat_interleave(
                    cu_seqlens_k[:-1],
                    extend_seq_lens,
                )
        else:
            assert False, f"Unsupported {forward_batch.forward_mode = }"

        indexer_k_start_end, token_to_batch_idx = self._cal_indexer_k_start_end(
            forward_batch, bs_idx_cpu
        )
        # 1D, expanded seqlens (1D means cheap to compute, so always compute it)
        nsa_cache_seqlens_int32 = compute_nsa_seqlens(
            original_seq_lens=seqlens_expanded,
            nsa_index_topk=self.nsa_index_topk,
        )
        nsa_cache_seqlens_int32 = pad_nsa_cache_seqlens(
            forward_batch, nsa_cache_seqlens_int32
        )
        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))

        paged_mqa_schedule_metadata = None
        use_b12x_paged_indexer = (
            (
                forward_batch.forward_mode.is_decode_or_idle()
                and self.nsa_decode_impl == "b12x"
            )
            or (
                (
                    forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                )
                and self.nsa_prefill_impl == "b12x"
            )
        )
        if use_b12x_paged_indexer and is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            seqlens_32 = (
                seqlens_expanded
                if (
                    forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                )
                else cache_seqlens_int32
            )
            paged_mqa_schedule_metadata = _get_b12x_paged_mqa_logits_metadata(
                seqlens_32
            )
        elif is_cuda() and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                # NOTE: DeepGEMM paged path uses block_size=64.
                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_batch.forward_mode.is_target_verify()
                        or forward_batch.forward_mode.is_draft_extend()
                    )
                    else cache_seqlens_int32
                )
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
            except (ImportError, ModuleNotFoundError):
                paged_mqa_schedule_metadata = None

        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seq_lens_sum=forward_batch.seq_lens_sum,
            page_table_1=page_table,
            page_table_1_flattened=page_table_1_flattened,
            flashmla_metadata=(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens_int32,
                    seq_len_q=1,
                )
                if self.nsa_decode_impl == "flashmla_kv"
                else None
            ),
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_seq_lens_cpu,
            real_page_table=self._transform_table_1_to_real(page_table),
            nsa_max_seqlen_q=1,
            topk_indices_offset=topk_indices_offset,
            indexer_k_start_end=indexer_k_start_end,
            indexer_seq_lens_cpu=indexer_seq_lens_cpu,
            indexer_seq_lens=indexer_seq_lens,
            token_to_batch_idx=token_to_batch_idx,
        )
        self.forward_metadata = metadata

    def _cal_indexer_k_start_end(
        self,
        forward_batch: ForwardBatch,
        bs_idx: Optional[List[int]] = None,
    ):
        if not forward_batch.forward_mode.is_extend_without_speculative():
            return None, None
        if forward_batch.batch_size == 0 or (bs_idx is not None and len(bs_idx) == 0):
            empty_t = torch.empty(0, dtype=torch.int32, device=self.device)
            return (empty_t, empty_t), empty_t

        # Suppose there are two requests, with extend_seq_len = [3, 2]
        # and seq_lens = [10, 4]
        # The logits matrix looks like this, with * representing the valid logits
        # and - representing the invalid logits:
        #
        #  ********--|----
        #  *********-|----
        #  **********|----
        #  ----------|***-
        #  ----------|****
        #
        # ks = [0, 0, 0, 10, 10]
        # ke = [8, 9, 10, 13, 14]
        ks_list = []
        ke_list = []
        token_to_batch_idx = []

        q_offset = 0
        k_offset = 0

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens_cpu[i].item()
            assert isinstance(seq_len, int)
            extend_seq_len = forward_batch.extend_seq_lens_cpu[i]
            ks = torch.full(
                (extend_seq_len,), k_offset, dtype=torch.int32, device=self.device
            )
            kv_len = seq_len
            if forward_batch.forward_mode.is_target_verify():
                kv_len += self.speculative_num_draft_tokens
            seq_lens_expanded = torch.arange(
                kv_len - extend_seq_len + 1,
                kv_len + 1,
                dtype=torch.int32,
                device=self.device,
            )
            ke = ks + seq_lens_expanded
            ks_list.append(ks)
            ke_list.append(ke)

            # bi: The index within the selected batch bs_idx. Entries that were not selected are ignored.
            bi = bs_idx.index(i) if (bs_idx is not None and i in bs_idx) else i
            tb = torch.full(
                (extend_seq_len,), bi, dtype=torch.int32, device=self.device
            )
            token_to_batch_idx.append(tb)

            if bs_idx is None or i in bs_idx:  # skip batch not included in bs_idx
                q_offset += extend_seq_len
                k_offset += seq_len

        ks = torch.cat(ks_list, dim=0)
        ke = torch.cat(ke_list, dim=0)
        token_to_batch_idx = torch.cat(token_to_batch_idx, dim=0)
        if bs_idx is not None:
            assert can_nsa_prefill_cp_round_robin_split(forward_batch)
            ks = nsa_cp_round_robin_split_data(ks)
            ke = nsa_cp_round_robin_split_data(ke)
            token_to_batch_idx = nsa_cp_round_robin_split_data(token_to_batch_idx)
        return (ks, ke), token_to_batch_idx

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        self.decode_cuda_graph_metadata: Dict = {
            "cache_seqlens": torch.ones(
                max_num_tokens, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            # fake page_table for sparse_prefill
            # Add extra columns for speculative draft tokens to avoid
            # overflow during target_verify when max_seqlen_k = seq_len + num_draft_tokens
            "page_table": torch.zeros(
                max_num_tokens,
                self.max_context_len + (self.speculative_num_draft_tokens or 0),
                dtype=torch.int32,
                device=self.device,
            ),
            "flashmla_metadata": (
                self._compute_flashmla_metadata(
                    cache_seqlens=torch.ones(
                        max_num_tokens, dtype=torch.int32, device=self.device
                    ),
                    seq_len_q=1,
                )
                if self.nsa_decode_impl == "flashmla_kv"
                else None
            ),
        }

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
        self.set_nsa_prefill_impl(forward_batch=None)

        """Initialize forward metadata for capturing CUDA graph."""
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            # Get sequence information
            cache_seqlens_int32 = seq_lens.to(torch.int32)
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)

            # Use max context length for seq_len_k
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][:bs, :]
            max_seqlen_q = 1
            max_seqlen_k = page_table_1.shape[1]

            # Precompute page table
            # Precompute cumulative sequence lengths

            # NOTE(dark): this is always arange, since we are decoding
            cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][: bs + 1]
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(
                cache_seqlens_int32, nsa_index_topk=self.nsa_index_topk
            )

            seqlens_expanded = cache_seqlens_int32
            nsa_extend_seq_lens_list = [1] * num_tokens
            if self.nsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, num_tokens + 1))
                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend(
            include_v2=True
        ):
            cache_seqlens_int32 = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
            max_seqlen_q = 1
            page_table_1 = self.decode_cuda_graph_metadata["page_table"][
                : bs * self.speculative_num_draft_tokens, :
            ]
            max_seqlen_k = page_table_1.shape[1]

            cu_seqlens_q = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens + 1,
                1,
                dtype=torch.int32,
                device=self.device,
            )

            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            seqlens_int32_cpu = [
                self.speculative_num_draft_tokens + kv_len
                for kv_len in seq_lens.tolist()
            ]
            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    for qo_len, kv_len in zip(
                        extend_seq_lens_cpu,
                        seqlens_int32_cpu,
                        strict=True,
                    )
                ]
            )
            nsa_cache_seqlens_int32 = compute_nsa_seqlens(
                seqlens_expanded, nsa_index_topk=self.nsa_index_topk
            )
            nsa_extend_seq_lens_list = [1] * bs * self.speculative_num_draft_tokens

            if self.nsa_decode_impl == "flashmla_kv":
                flashmla_metadata = self.decode_cuda_graph_metadata[
                    "flashmla_metadata"
                ].slice(slice(0, bs * self.speculative_num_draft_tokens + 1))

                flashmla_metadata.copy_(
                    self._compute_flashmla_metadata(
                        cache_seqlens=nsa_cache_seqlens_int32,
                        seq_len_q=1,
                    )
                )
            else:
                flashmla_metadata = None

        nsa_cu_seqlens_k = compute_cu_seqlens(nsa_cache_seqlens_int32)
        nsa_cu_seqlens_q = self.get_device_int32_arange(len(nsa_cu_seqlens_k))
        real_page_table = self._transform_table_1_to_real(page_table_1)

        paged_mqa_schedule_metadata = None
        use_b12x_paged_indexer = (
            (forward_mode.is_decode_or_idle() and self.nsa_decode_impl == "b12x")
            or (
                (forward_mode.is_target_verify() or forward_mode.is_draft_extend())
                and self.nsa_prefill_impl == "b12x"
            )
        )
        if use_b12x_paged_indexer and is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
        ):
            seqlens_32 = (
                seqlens_expanded
                if (
                    forward_mode.is_target_verify()
                    or forward_mode.is_draft_extend()
                )
                else cache_seqlens_int32
            )
            paged_mqa_schedule_metadata = _get_b12x_paged_mqa_logits_metadata(
                seqlens_32
            )
        elif is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend()
                    )
                    else cache_seqlens_int32
                )
                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
            except (ImportError, ModuleNotFoundError):
                paged_mqa_schedule_metadata = None

        metadata = NSAMetadata(
            page_size=self.real_page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            page_table_1=page_table_1,
            flashmla_metadata=flashmla_metadata,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
            nsa_cu_seqlens_q=nsa_cu_seqlens_q,
            nsa_cu_seqlens_k=nsa_cu_seqlens_k,
            nsa_seqlens_expanded=seqlens_expanded,
            real_page_table=real_page_table,
            nsa_extend_seq_lens_list=nsa_extend_seq_lens_list,
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

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
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        assert seq_lens_cpu is not None

        self.set_nsa_prefill_impl(forward_batch=None)

        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        # Normal Decode
        metadata: NSAMetadata = self.decode_cuda_graph_metadata[bs]
        if forward_mode.is_decode_or_idle():
            # Normal Decode
            max_len = int(seq_lens_cpu.max().item())

            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[req_pool_indices, :max_len]
            metadata.page_table_1[:, :max_len].copy_(page_indices)
            nsa_cache_seqlens = compute_nsa_seqlens(
                cache_seqlens, nsa_index_topk=self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
            seqlens_expanded = cache_seqlens
        elif forward_mode.is_target_verify():
            max_seqlen_k = int(
                seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
            )

            cache_seqlens = (seq_lens + self.speculative_num_draft_tokens).to(
                torch.int32
            )
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(
                page_indices, repeats=self.speculative_num_draft_tokens, dim=0
            )
            metadata.page_table_1[:, :max_seqlen_k].copy_(page_indices)
            extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * bs

            seqlens_expanded = seqlens_expand_triton(
                torch.tensor(
                    extend_seq_lens_cpu, dtype=torch.int32, device=self.device
                ),
                cache_seqlens,
                self.speculative_num_draft_tokens * bs,
                self.speculative_num_draft_tokens,
            )
            metadata.nsa_seqlens_expanded.copy_(seqlens_expanded)
            nsa_cache_seqlens = compute_nsa_seqlens(
                seqlens_expanded, self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32.copy_(nsa_cache_seqlens)
        elif forward_mode.is_draft_extend(include_v2=True):
            max_seqlen_k = int(seq_lens_cpu.max().item())
            cache_seqlens = seq_lens.to(torch.int32)
            metadata.cache_seqlens_int32.copy_(cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32)
            )

            extend_seq_lens = spec_info.accept_length[:bs]
            extend_seq_lens_cpu = extend_seq_lens.tolist()

            page_indices = self.req_to_token[req_pool_indices, :max_seqlen_k]
            page_indices = torch.repeat_interleave(
                page_indices, repeats=extend_seq_lens, dim=0
            )
            metadata.page_table_1[: page_indices.shape[0], :max_seqlen_k].copy_(
                page_indices
            )

            seqlens_expanded = seqlens_expand_triton(
                extend_seq_lens,
                cache_seqlens,
                sum(extend_seq_lens_cpu),
                self.speculative_num_draft_tokens,
            )
            metadata.nsa_seqlens_expanded[: seqlens_expanded.shape[0]].copy_(
                seqlens_expanded
            )
            nsa_cache_seqlens = compute_nsa_seqlens(
                seqlens_expanded, self.nsa_index_topk
            )
            metadata.nsa_cache_seqlens_int32[: seqlens_expanded.shape[0]].copy_(
                nsa_cache_seqlens
            )

        # Update DeepGEMM paged MQA schedule metadata outside the captured graph.
        use_b12x_paged_indexer = (
            (forward_mode.is_decode_or_idle() and self.nsa_decode_impl == "b12x")
            or (
                (forward_mode.is_target_verify() or forward_mode.is_draft_extend())
                and self.nsa_prefill_impl == "b12x"
            )
        )
        if use_b12x_paged_indexer:
            seqlens_32 = (
                seqlens_expanded
                if (
                    forward_mode.is_target_verify()
                    or forward_mode.is_draft_extend()
                )
                else metadata.cache_seqlens_int32
            )
            new_schedule = _get_b12x_paged_mqa_logits_metadata(
                seqlens_32
            )
            if new_schedule is None:
                object.__setattr__(metadata, "paged_mqa_schedule_metadata", None)
            elif metadata.paged_mqa_schedule_metadata is None:
                object.__setattr__(metadata, "paged_mqa_schedule_metadata", new_schedule)
            else:
                metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
        elif is_cuda() and (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
        ):
            try:
                import deep_gemm

                seqlens_32 = (
                    seqlens_expanded
                    if (
                        forward_mode.is_target_verify()
                        or forward_mode.is_draft_extend()
                    )
                    else metadata.cache_seqlens_int32
                )
                new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, 64, deep_gemm.get_num_sms()
                )
                if metadata.paged_mqa_schedule_metadata is None:
                    object.__setattr__(metadata, "paged_mqa_schedule_metadata", new_schedule)
                else:
                    metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
            except (ImportError, ModuleNotFoundError):
                object.__setattr__(metadata, "paged_mqa_schedule_metadata", None)
        seqlens_expanded_size = seqlens_expanded.shape[0]
        assert (
            metadata.nsa_cache_seqlens_int32 is not None
            and metadata.nsa_cu_seqlens_k is not None
            and self.nsa_index_topk is not None
        )

        metadata.nsa_cu_seqlens_k[1 : 1 + seqlens_expanded_size].copy_(
            torch.cumsum(nsa_cache_seqlens, dim=0, dtype=torch.int32)
        )
        # NOTE(dark): (nsa-) cu_seqlens_q is always arange, no need to copy

        assert self.real_page_size == metadata.page_size
        if self.real_page_size > 1:
            real_table = self._transform_table_1_to_real(page_indices)
            new_rows = real_table.shape[0]
            new_cols = real_table.shape[1]
            metadata.real_page_table[:new_rows, :new_cols].copy_(real_table)
        else:
            assert metadata.real_page_table is metadata.page_table_1

        if self.nsa_decode_impl == "flashmla_kv":
            flashmla_metadata = metadata.flashmla_metadata.slice(
                slice(0, seqlens_expanded_size + 1)
            )
            flashmla_metadata.copy_(
                self._compute_flashmla_metadata(
                    cache_seqlens=nsa_cache_seqlens,
                    seq_len_q=1,
                )
            )

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph_from_precomputed(
        self,
        bs: int,
        precomputed: PrecomputedMetadata,
        forward_mode: ForwardMode,
    ):
        """Fast path: copy precomputed metadata to this backend's metadata.

        This function only performs copy operations, no computation.

        Args:
            bs: Batch size
            precomputed: Precomputed metadata to copy from
            forward_mode: Forward mode
        """
        self.set_nsa_prefill_impl(forward_batch=None)

        metadata = self.decode_cuda_graph_metadata[bs]

        # Track whether fused kernel succeeded
        fused_kernel_succeeded = False

        # Use fused CUDA kernel for all copy operations
        if _USE_FUSED_METADATA_COPY:
            try:
                from sglang.jit_kernel.fused_metadata_copy import (
                    fused_metadata_copy_cuda,
                )

                # Map forward_mode to integer enum
                if forward_mode.is_decode_or_idle():
                    mode_int = 0  # DECODE
                elif forward_mode.is_target_verify():
                    mode_int = 1  # TARGET_VERIFY
                elif forward_mode.is_draft_extend():
                    mode_int = 2  # DRAFT_EXTEND
                else:
                    raise ValueError(f"Unsupported forward_mode: {forward_mode}")

                # Prepare FlashMLA tensors if needed
                flashmla_num_splits_src = None
                flashmla_num_splits_dst = None
                flashmla_metadata_src = None
                flashmla_metadata_dst = None
                if precomputed.flashmla_metadata is not None:
                    flashmla_num_splits_src = precomputed.flashmla_metadata.num_splits
                    flashmla_num_splits_dst = metadata.flashmla_metadata.num_splits
                    flashmla_metadata_src = (
                        precomputed.flashmla_metadata.flashmla_metadata
                    )
                    flashmla_metadata_dst = metadata.flashmla_metadata.flashmla_metadata

                # Call fused kernel
                fused_metadata_copy_cuda(
                    # Source tensors
                    precomputed.cache_seqlens,
                    precomputed.cu_seqlens_k,
                    precomputed.page_indices,
                    precomputed.nsa_cache_seqlens,
                    precomputed.seqlens_expanded,
                    precomputed.nsa_cu_seqlens_k,
                    precomputed.real_page_table,
                    flashmla_num_splits_src,
                    flashmla_metadata_src,
                    # Destination tensors
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table_1,
                    metadata.nsa_cache_seqlens_int32,
                    metadata.nsa_seqlens_expanded,
                    metadata.nsa_cu_seqlens_k,
                    (
                        metadata.real_page_table
                        if precomputed.real_page_table is not None
                        else None
                    ),
                    flashmla_num_splits_dst,
                    flashmla_metadata_dst,
                    # Parameters
                    mode_int,
                    bs,
                    precomputed.max_len,
                    precomputed.max_seqlen_k,
                    precomputed.seqlens_expanded_size,
                )

                # Successfully used fused kernel
                fused_kernel_succeeded = True

            except ImportError:
                print(
                    "Warning: Fused metadata copy kernel not available, falling back to individual copies."
                )
            except Exception as e:
                print(
                    f"Warning: Fused metadata copy kernel failed with error: {e}, falling back to individual copies."
                )

        # Fallback to individual copy operations if fused kernel disabled or failed
        if not fused_kernel_succeeded:
            # Copy basic seqlens
            metadata.cache_seqlens_int32.copy_(precomputed.cache_seqlens)
            metadata.cu_seqlens_k[1:].copy_(precomputed.cu_seqlens_k[1:])

            # Mode-specific copy logic
            if forward_mode.is_decode_or_idle():
                # Decode mode
                metadata.page_table_1[:, : precomputed.max_len].copy_(
                    precomputed.page_indices
                )
                metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)
                # seqlens_expanded is same as cache_seqlens (already copied)

            elif forward_mode.is_target_verify():
                # Target verify mode
                metadata.page_table_1[:, : precomputed.max_seqlen_k].copy_(
                    precomputed.page_indices
                )
                metadata.nsa_seqlens_expanded.copy_(precomputed.seqlens_expanded)
                metadata.nsa_cache_seqlens_int32.copy_(precomputed.nsa_cache_seqlens)

            elif forward_mode.is_draft_extend():
                # Draft extend mode
                rows = precomputed.page_indices.shape[0]
                cols = precomputed.max_seqlen_k
                metadata.page_table_1[:rows, :cols].copy_(precomputed.page_indices)

                size = precomputed.seqlens_expanded_size
                metadata.nsa_seqlens_expanded[:size].copy_(precomputed.seqlens_expanded)
                metadata.nsa_cache_seqlens_int32[:size].copy_(
                    precomputed.nsa_cache_seqlens
                )

            # Copy NSA cu_seqlens
            size = precomputed.seqlens_expanded_size
            metadata.nsa_cu_seqlens_k[1 : 1 + size].copy_(
                precomputed.nsa_cu_seqlens_k[1 : 1 + size]
            )

            # Copy real page table
            if precomputed.real_page_table is not None:
                rows, cols = precomputed.real_page_table.shape
                metadata.real_page_table[:rows, :cols].copy_(
                    precomputed.real_page_table
                )

            # Copy FlashMLA metadata in fallback path
            if precomputed.flashmla_metadata is not None:
                size = precomputed.seqlens_expanded_size
                flashmla_metadata = metadata.flashmla_metadata.slice(slice(0, size + 1))
                flashmla_metadata.copy_(precomputed.flashmla_metadata)

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "NSA is causal only"

        nsa_impl = (
            self.nsa_decode_impl
            if (
                forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            )
            else self.nsa_prefill_impl
        )

        if nsa_impl == "trtllm" and not self.use_mha:
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.nsa_cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
                is_prefill=True,
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                import os as _os
                if (
                    _os.environ.get("SGLANG_NSA_DEBUG_DUMP")
                    and getattr(type(self), "_dbg_wr_dumped", 0) < 3
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    try:
                        with open("/root/.cache/huggingface/nsa_dump.txt", "a") as _f:
                            _f.write(
                                f"[extend-write] k(c_kv) {tuple(k.shape)} {k.dtype} "
                                f"absmean={k.float().abs().mean().item():.5f} "
                                f"rms={k.float().pow(2).mean().sqrt().item():.5f} | "
                                f"k_rope absmean={k_rope.float().abs().mean().item():.5f}\n"
                            )
                        type(self)._dbg_wr_dumped = getattr(type(self), "_dbg_wr_dumped", 0) + 1
                    except Exception:
                        pass
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Use MHA kernel if in MHA_ONE_SHOT mode
        if self.use_mha:
            assert k is not None and v is not None
            assert q_rope is None, "MHA_ONE_SHOT path should not pass q_rope"
            assert (
                layer.tp_k_head_num == layer.tp_q_head_num > 1
            ), "MHA_ONE_SHOT requires dense multi-head config"
            return self._forward_standard_mha(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
            )

        # Do absorbed multi-latent attention (MLA path)
        assert q_rope is not None
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)

        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        # Align topk_indices with q dimensions
        # This handles cases where q is padded (TP + partial DP attention)
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        # NOTE(dark): here, we use page size = 1
        topk_transform_method = self.get_topk_transform_method(
            forward_batch.forward_mode
        )
        if nsa_impl == "b12x":
            # GLM-OPTIONB (2026-06-19): EXTEND/prefill keeps RAW request-local topk_indices.
            # The b12x prefill kernel (run_unified_prefill_mg) does its own page-table
            # handling and expects request-local indices — gathering to global slots here
            # triggers a device-side assert. This is ASYMMETRIC with decode: the b12x
            # DECODE kernel (run_unified_decode) needs PRE-gathered global slots (validated
            # by b12x_sparse_mla_probe.py: cos 1.0 global vs 0.08 raw), so only the decode
            # branch gathers. Do NOT "fix" this to match decode.
            page_table_1 = topk_indices
        elif envs.SGLANG_NSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        else:
            if topk_transform_method == TopkTransformMethod.RAGGED:
                topk_indices_offset = metadata.topk_indices_offset
                assert topk_indices_offset is not None
                mask = topk_indices != -1
                topk_indices_offset = (
                    topk_indices_offset.unsqueeze(1)
                    if topk_indices_offset.ndim == 1
                    else topk_indices_offset
                )
                topk_indices = torch.where(
                    mask, topk_indices + topk_indices_offset, topk_indices
                )
            elif topk_transform_method == TopkTransformMethod.PAGED:
                assert metadata.nsa_extend_seq_lens_list is not None
                page_table_1 = transform_index_page_table_prefill(
                    page_table=metadata.page_table_1,
                    topk_indices=topk_indices,
                    extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                    page_size=1,
                )

        # todo hisparse: to cover more backends
        if forward_batch.hisparse_coordinator is not None:
            page_table_1 = (
                forward_batch.token_to_kv_pool.translate_loc_to_hisparse_device(
                    page_table_1
                )
            )

        if nsa_impl == "b12x":
            if forward_batch.hisparse_coordinator is not None:
                raise ValueError("b12x does not support HiSparse in v1.")
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            extend_mode: Literal["extend", "target_verify", "draft_extend"]
            if forward_batch.forward_mode.is_target_verify():
                extend_mode = "target_verify"
            elif forward_batch.forward_mode.is_draft_extend(include_v2=True):
                extend_mode = "draft_extend"
            else:
                extend_mode = "extend"
            return self._forward_b12x(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                metadata=metadata,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                mode=extend_mode,
            )
        if nsa_impl == "tilelang":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif nsa_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)

            if topk_transform_method == TopkTransformMethod.RAGGED:
                if any(forward_batch.extend_prefix_lens_cpu):
                    page_table_1_flattened = (
                        self.forward_metadata.page_table_1_flattened
                    )
                    assert page_table_1_flattened is not None
                    kv_cache = dequantize_k_cache_paged(
                        kv_cache, page_table_1_flattened
                    )
                else:
                    kv_cache = _cat([k, k_rope], dim=-1)
                page_table_1 = topk_indices

            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif nsa_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif nsa_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif nsa_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter_extend(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
            )
        else:
            raise ValueError(
                f"Unsupported {nsa_impl = } for forward_extend. Consider using an other attention backend."
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        causal = not layer.is_cross_attention
        metadata = self.forward_metadata
        assert causal, "NSA is causal only"

        if self.nsa_decode_impl == "trtllm":
            return self._forward_trtllm(
                q,
                k,
                v,
                layer,
                forward_batch,
                metadata.cache_seqlens_int32,
                save_kv_cache,
                q_rope,
                k_rope,
                topk_indices,
                cos_sin_cache,
                is_neox,
                llama_4_scaling,
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                forward_batch.token_to_kv_pool.set_mla_kv_buffer(  # type: ignore
                    layer,
                    cache_loc,
                    k,
                    k_rope,
                )

        # Do absorbed multi-latent attention
        kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if q_rope is not None:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
        else:
            q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            q_nope = q_all[:, :, : layer.v_head_dim]
            q_rope = q_all[:, :, layer.v_head_dim :]

        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q_nope.shape[0])

        if forward_batch.hisparse_coordinator is not None:
            page_table_1 = forward_batch.hisparse_coordinator.swap_in_selected_pages(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                topk_indices,
                layer.layer_id,
            )
        elif self.nsa_decode_impl == "b12x":
            import os as _os
            if (
                _os.environ.get("SGLANG_NSA_DEBUG_DUMP")
                and getattr(type(self), "_dbg_pre_dumped", 0) < 4
                and not torch.cuda.is_current_stream_capturing()
                and topk_indices is not None
                and int(metadata.cache_seqlens_int32.max()) > 2
            ):
                try:
                    pt = metadata.page_table_1
                    ti = topk_indices
                    with open("/root/.cache/huggingface/nsa_dump.txt", "a") as _f:
                        _f.write(
                            f"[pre-transform] topk_indices {tuple(ti.shape)} {ti.dtype} "
                            f"min={int(ti.min())} max={int(ti.max())} "
                            f"row0valid={ti[0][ti[0]>=0][:12].tolist()}\n"
                            f"  metadata.page_table_1 {tuple(pt.shape)} {pt.dtype} "
                            f"min={int(pt.min())} max={int(pt.max())} row0[:12]={pt[0][:12].tolist()}\n"
                            f"  cache_seqlens={metadata.cache_seqlens_int32.tolist()[:6]} "
                            f"nsa={metadata.nsa_cache_seqlens_int32.tolist()[:6]}\n"
                        )
                    type(self)._dbg_pre_dumped = getattr(type(self), "_dbg_pre_dumped", 0) + 1
                except Exception:
                    pass
            # GLM-OPTIONB (2026-06-19): with SGLANG_NSA_FUSE_TOPK=True (default) + PAGED
            # method (forced for b12x), topk_transform already returns a TRANSFORMED
            # global page table (fast_topk_transform_fused). topk_indices ARE the global
            # KV-slot ids the MLA decode needs -> pass directly. (transform here would
            # double-transform -> OOB assert.)
            page_table_1 = topk_indices
        elif envs.SGLANG_NSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        if self.nsa_decode_impl == "b12x":
            if forward_batch.hisparse_coordinator is not None:
                raise ValueError("b12x does not support HiSparse in v1.")
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_b12x(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                metadata=metadata,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                mode="decode",
            )
        if self.nsa_decode_impl == "flashmla_sparse":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_sparse(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_decode_impl == "flashmla_kv":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_flashmla_kv(
                q_all=q_all,
                kv_cache=kv_cache,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
                # TODO optimize args
                layer=layer,
                metadata=metadata,
                page_table_1=page_table_1,
            )
        elif self.nsa_decode_impl == "tilelang":
            if q_rope is not None:
                q_all = concat_mla_absorb_q_general(q_nope, q_rope)
            return self._forward_tilelang(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                sm_scale=layer.scaling,
                v_head_dim=layer.v_head_dim,
            )
        elif self.nsa_decode_impl == "fa3":
            return self._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=layer.v_head_dim,
                q_nope=q_nope,
                page_table=page_table_1,
                cache_seqlens=metadata.nsa_cache_seqlens_int32,
                cu_seqlens_q=metadata.nsa_cu_seqlens_q,
                cu_seqlens_k=metadata.nsa_cu_seqlens_k,
                max_seqlen_q=metadata.nsa_max_seqlen_q,
                sm_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                page_size=1,
            )
        elif self.nsa_decode_impl == "aiter":
            if q_rope is not None:
                q_all = torch.cat([q_nope, q_rope], dim=-1)
            return self._forward_aiter(
                q_all=q_all,
                kv_cache=kv_cache,
                page_table_1=page_table_1,
                layer=layer,
                metadata=metadata,
                bs=forward_batch.batch_size,
            )

        else:
            assert False, f"Unsupported {self.nsa_decode_impl = }"

    def _forward_fa3(
        self,
        q_rope: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        q_nope: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        sm_scale: float,
        logit_cap: float,
        page_size: int,
    ) -> torch.Tensor:
        k_rope_cache = kv_cache[:, :, v_head_dim:]
        c_kv_cache = kv_cache[:, :, :v_head_dim]
        qk_rope_dim = k_rope_cache.shape[-1]
        k_rope_cache = k_rope_cache.view(-1, page_size, 1, qk_rope_dim)
        c_kv_cache = c_kv_cache.view(-1, page_size, 1, v_head_dim)
        o = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope_cache,
            v_cache=c_kv_cache,
            qv=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=sm_scale,
            causal=True,
            softcap=logit_cap,
            return_softmax_lse=False,
            num_splits=self.num_splits,
        )
        return o  # type: ignore

    def _forward_flashmla_sparse(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        # FlashMLA sparse kernel requires num_heads to be a multiple of 64 (Hopper) or 128 (Blackwell)
        # When using TP, num_heads might be smaller (e.g., 256//8=32)
        num_tokens, num_heads, head_dim = q_all.shape

        # Determine required padding based on GPU architecture (use cached value)
        required_padding = 128 if self.device_sm_major >= 10 else 64

        need_padding = num_heads % required_padding != 0

        if need_padding:
            assert required_padding % num_heads == 0, (
                f"num_heads {num_heads} cannot be padded to {required_padding}. "
                f"TP size may be too large for this model."
            )

            # Pad q to required size
            q_padded = q_all.new_zeros((num_tokens, required_padding, head_dim))
            q_padded[:, :num_heads, :] = q_all
            q_input = q_padded
        else:
            q_input = q_all

        # indices shape must be (s_q, h_kv=1, topk), keep h_kv=1 unchanged
        indices_input = page_table_1.unsqueeze(1)

        o, _, _ = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv_cache,
            indices=indices_input,
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

        # Trim output back to original num_heads if we padded
        if need_padding:
            o = o[:, :num_heads, :]

        return o

    def _forward_flashmla_kv(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        sm_scale: float,
        layer,
        metadata: NSAMetadata,
        page_table_1,
    ) -> torch.Tensor:
        from sgl_kernel.flash_mla import flash_mla_with_kvcache

        cache_seqlens = metadata.nsa_cache_seqlens_int32

        # TODO the 2nd dim is seq_len_q, need to be >1 when MTP
        q_all = q_all.view(-1, 1, layer.tp_q_head_num, layer.head_dim)
        kv_cache = kv_cache.view(-1, self.real_page_size, 1, self.kv_cache_dim)
        assert self.real_page_size == 64, "only page size 64 is supported"

        if not self.nsa_kv_cache_store_fp8:
            # inefficiently quantize the whole cache
            kv_cache = quantize_k_cache(kv_cache)

        indices = page_table_1.unsqueeze(1)
        assert (
            indices.shape[-1] == self.nsa_index_topk
        )  # requirement of FlashMLA decode kernel

        o, _ = flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv_cache,
            cache_seqlens=cache_seqlens,
            head_dim_v=v_head_dim,
            tile_scheduler_metadata=metadata.flashmla_metadata.flashmla_metadata,
            num_splits=metadata.flashmla_metadata.num_splits,
            softmax_scale=sm_scale,
            indices=indices,
            # doc says it is not used, but if pass in None then error
            block_table=torch.empty(
                (q_all.shape[0], 0), dtype=torch.int32, device=q_all.device
            ),
            is_fp8_kvcache=True,
        )
        return o

    def _forward_standard_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: NSAMetadata,
    ) -> torch.Tensor:
        """Standard MHA using FlashAttention varlen for MHA_ONE_SHOT mode."""
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        # MHA_ONE_SHOT: k/v include all tokens (prefix + current)
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        max_seqlen_k = metadata.max_seq_len_k
        causal = True

        # Verify batch sizes match (length of cu_seqlens should be batch_size + 1)
        assert len(cu_seqlens_q) == len(cu_seqlens_k), (
            f"batch_size mismatch: cu_seqlens_q has {len(cu_seqlens_q)-1} requests, "
            f"cu_seqlens_k has {len(cu_seqlens_k)-1} requests"
        )

        # Use TRTLLm ragged attention for SM100 (Blackwell/B200) to avoid FA4 accuracy issues
        if self.device_sm_major >= 10:
            import flashinfer

            seq_lens = metadata.cache_seqlens_int32
            return flashinfer.prefill.trtllm_ragged_attention_deepseek(
                query=q,
                key=k,
                value=v,
                workspace_buffer=self.workspace_buffer,
                seq_lens=seq_lens,
                max_q_len=metadata.max_seq_len_q,
                max_kv_len=max_seqlen_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=forward_batch.batch_size,
                window_left=-1,
                cum_seq_lens_q=cu_seqlens_q,
                cum_seq_lens_kv=cu_seqlens_k,
                enable_pdl=False,
                is_causal=causal,
                return_lse=False,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
            )

        # Use FA3 for SM90 (Hopper/H200)
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=layer.scaling,
            causal=causal,
        )

    def _forward_tilelang(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        v_head_dim: int,
        page_table_1: torch.Tensor,
        sm_scale: float,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.tilelang_kernel import tilelang_sparse_fwd

        return tilelang_sparse_fwd(
            q=q_all,
            kv=kv_cache,
            indices=page_table_1.unsqueeze(1),
            sm_scale=sm_scale,
            d_v=v_head_dim,
        )

    def _forward_aiter(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
        metadata: NSAMetadata,
        bs: int,
    ) -> torch.Tensor:
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        kv_indptr = self.kv_indptr

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)
        kv_indptr[1 : bs + 1] = torch.cumsum(non_minus1_counts, dim=0)

        kv_indices = self.kv_indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)

        mla_decode_fwd(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            metadata.cu_seqlens_q,
            kv_indptr,
            kv_indices,
            metadata.cu_seqlens_q,
            metadata.max_seq_len_q,
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
        )
        # kv_cache = kv_cache.view(-1, 1, layer.head_dim)
        return o

    def _forward_aiter_extend(
        self,
        q_all: torch.Tensor,
        kv_cache: torch.Tensor,
        page_table_1: torch.Tensor,
        layer: RadixAttention,
    ) -> torch.Tensor:
        num_tokens = q_all.shape[0]
        q = q_all.reshape(-1, layer.tp_q_head_num * layer.head_dim)

        if layer.head_dim != layer.v_head_dim:
            o = q.new_empty((num_tokens, layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        non_minus1_mask = page_table_1 != -1
        non_minus1_counts = non_minus1_mask.sum(dim=1)

        kv_indptr = torch.zeros(num_tokens + 1, dtype=torch.int32, device=self.device)
        kv_indptr[1:] = torch.cumsum(non_minus1_counts, dim=0)

        # Allocate kv_indices with upper-bound size (num_tokens * topk)
        topk = page_table_1.shape[1]
        kv_indices = torch.zeros(
            num_tokens * topk, dtype=torch.int32, device=self.device
        )

        # Use get_valid_kv_indices kernel to extract valid indices
        get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, num_tokens)

        # Build cu_seqlens_q for extend: each token is treated as seq_len_q=1
        cu_seqlens_q = torch.arange(
            0, num_tokens + 1, dtype=torch.int32, device=self.device
        )
        # TODO support more forward_mode
        mla_decode_fwd(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_cache.view(-1, 1, 1, layer.head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            cu_seqlens_q,
            1,  # max_seq_len_q = 1 for per-token attention
            sm_scale=layer.scaling,
            logit_cap=layer.logit_cap,
        )
        return o

    def _forward_trtllm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        seq_lens: torch.Tensor,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        is_neox: Optional[bool] = False,
        llama_4_scaling: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """Forward using TRT-LLM sparse MLA kernel."""
        import flashinfer.decode

        metadata = self.forward_metadata

        merge_query = q_rope is not None
        if self.kv_cache_dtype == torch.float8_e4m3fn:
            # For FP8 path, we quantize the query and rope parts and merge them into a single tensor
            # Note: rope application in deepseek_v2.py:forward_absorb_prepare is skipped for FP8 decode path of this trtllm_mla backend
            assert q_rope is not None, "For FP8 path q_rope should not be None."
            assert k_rope is not None, "For FP8 path k_rope should not be None."
            assert (
                cos_sin_cache is not None
            ), "For FP8 path cos_sin_cache should not be None."

            q, k, k_rope = mla_quantize_and_rope_for_fp8(
                q,
                q_rope,
                k.squeeze(1),
                k_rope.squeeze(1),
                forward_batch.positions,
                cos_sin_cache,
                is_neox,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
            merge_query = False

            # Save KV cache if requested
        if save_kv_cache:
            assert (
                k is not None and k_rope is not None
            ), "For populating trtllm_mla kv cache, both k_nope and k_rope should be not None."
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                layer, cache_loc, k, k_rope
            )

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        kv_cache = k_cache.view(-1, self.real_page_size, self.kv_cache_dim).unsqueeze(1)

        if merge_query:
            q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            q_rope_reshaped = q_rope.view(
                -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
            )
            q_all = concat_mla_absorb_q_general(q_nope, q_rope_reshaped)
        else:
            q_all = q.view(-1, layer.tp_q_head_num, layer.head_dim)

        # Align topk_indices with q dimensions
        if topk_indices is not None:
            topk_indices = self._pad_topk_indices(topk_indices, q.shape[0])

        if envs.SGLANG_NSA_FUSE_TOPK.get():
            page_table_1 = topk_indices
        elif is_prefill:
            page_table_1 = transform_index_page_table_prefill(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                extend_lens_cpu=metadata.nsa_extend_seq_lens_list,
                page_size=1,
            )
        else:
            page_table_1 = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )

        q_scale = 1.0
        k_scale = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        bmm1_scale = q_scale * k_scale * layer.scaling

        batch_size = page_table_1.shape[0]
        _, num_heads, head_dim = q_all.shape

        q = q_all.view(batch_size, 1, num_heads, head_dim)
        kv = kv_cache.view(-1, 1, self.real_page_size, self.kv_cache_dim)
        block_tables = page_table_1.unsqueeze(1)
        seq_lens = metadata.cache_seqlens_int32 if seq_lens is None else seq_lens

        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv,
            workspace_buffer=self.workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=metadata.max_seq_len_k,
            sparse_mla_top_k=self.nsa_index_topk,
            bmm1_scale=bmm1_scale,
            backend="trtllm-gen",
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
        )
        # Output: [batch, q_len=1, heads, v_dim] -> [batch, heads, v_dim]
        return out.squeeze(1)

    def _pad_topk_indices(
        self, topk_indices: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        current_tokens = topk_indices.shape[0]
        if current_tokens == num_tokens:
            return topk_indices

        assert current_tokens <= num_tokens, (
            f"topk_indices rows ({current_tokens}) > num_tokens ({num_tokens}); "
            "this indicates a mismatch between indexer output and q layout."
        )

        pad_size = num_tokens - current_tokens
        padding = torch.full(
            (pad_size, topk_indices.shape[1]),
            -1,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        return torch.cat([topk_indices, padding], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def set_nsa_prefill_impl(self, forward_batch: Optional[ForwardBatch] = None):
        """
        Decide all attention prefill dispatch strategies for this batch.
        """
        from sglang.srt.utils import get_device_sm, is_blackwell

        if self.nsa_prefill_impl == "b12x":
            self.use_mha = False
            return

        # Decide MHA vs MLA
        if forward_batch and forward_batch.forward_mode.is_extend_without_speculative():
            # Check if sequence meets criteria for MHA_ONE_SHOT
            assert forward_batch.seq_lens_cpu is not None
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            sum_seq_lens = sum(forward_batch.seq_lens_cpu)
            device_sm = get_device_sm()

            # Requirements: H200/B200, short sequences, supported dtype, fits in chunk
            self.use_mha = (
                (
                    device_sm == 90 or (device_sm >= 100 and device_sm < 110)
                )  # SM90/SM100 only
                and max_kv_len
                <= envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()  # Short enough for MHA
                and forward_batch.token_to_kv_pool.dtype
                in [torch.bfloat16, torch.float8_e4m3fn]
                and sum_seq_lens
                <= forward_batch.get_max_chunk_capacity()  # Fits in chunk
                and (not is_nsa_enable_prefill_cp())  # CP not enabled
                and (forward_batch.hisparse_coordinator is None)
            )
        else:
            self.use_mha = False  # Decode/verify always use MLA

        # Set MLA implementation only if not using MHA
        if not self.use_mha and self.enable_auto_select_prefill_impl:
            if self.nsa_kv_cache_store_fp8:
                if (
                    is_blackwell()
                    and forward_batch is not None
                    and forward_batch.forward_mode == ForwardMode.EXTEND
                ):
                    total_kv_tokens = forward_batch.seq_lens_sum
                    total_q_tokens = forward_batch.extend_num_tokens
                    # Heuristic based on benchmarking flashmla_kv vs flashmla_sparse + dequantize_k_cache_paged
                    if total_kv_tokens < total_q_tokens * 512:
                        self.nsa_prefill_impl = "flashmla_sparse"
                        return
                self.nsa_prefill_impl = "flashmla_kv"
            else:
                # bf16 kv cache
                self.nsa_prefill_impl = "flashmla_sparse"

    def get_topk_transform_method(
        self, forward_mode: Optional[ForwardMode] = None
    ) -> TopkTransformMethod:
        """
        SGLANG_NSA_FUSE_TOPK controls whether to fuse the topk transform into the topk kernel.
        This method is used to select the topk transform method which can be fused or unfused.
        """
        if self.nsa_prefill_impl == "b12x" or self.nsa_decode_impl == "b12x":
            return TopkTransformMethod.PAGED
        if (
            # disable for MTP
            self.nsa_kv_cache_store_fp8
            and self.nsa_prefill_impl == "flashmla_sparse"
            and forward_mode == ForwardMode.EXTEND
        ):
            topk_transform_method = TopkTransformMethod.RAGGED
        else:
            topk_transform_method = TopkTransformMethod.PAGED
        return topk_transform_method

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> NSAIndexerMetadata:
        force_unfused = (
            forward_batch.hisparse_coordinator is not None
            and forward_batch.forward_mode.is_decode_or_idle()
        )
        return NSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            topk_transform_method=self.get_topk_transform_method(
                forward_batch.forward_mode
            ),
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
            force_unfused_topk=force_unfused,
        )

    def _compute_flashmla_metadata(self, cache_seqlens: torch.Tensor, seq_len_q: int):
        from sgl_kernel.flash_mla import get_mla_metadata

        flashmla_metadata, num_splits = get_mla_metadata(
            cache_seqlens=cache_seqlens,
            # TODO doc says `num_q_tokens_per_q_seq * num_heads_q // num_heads_k`
            #      but the name looks like need seq_len_q?
            num_q_tokens_per_head_k=seq_len_q * self.num_q_heads // 1,
            num_heads_k=1,
            num_heads_q=self.num_q_heads,
            is_fp8_kvcache=True,
            topk=self.nsa_index_topk,
        )

        return NSAFlashMLAMetadata(
            flashmla_metadata=flashmla_metadata,
            num_splits=num_splits,
        )


class NativeSparseAttnMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                NativeSparseAttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if envs.SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA.get():
            # Precompute metadata once (shared across all backends)
            precomputed = self.attn_backends[0]._precompute_replay_metadata(
                bs=bs,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

            # Use multi-backend fused copy when we have 3 or more backends
            # This is 3x faster than calling the single-backend copy 3 times
            if self.speculative_num_steps > 3:
                try:
                    from sglang.jit_kernel.fused_metadata_copy import (
                        fused_metadata_copy_multi_cuda,
                    )

                    metadata0 = self.attn_backends[0].decode_cuda_graph_metadata[bs]
                    metadata1 = self.attn_backends[1].decode_cuda_graph_metadata[bs]
                    metadata2 = self.attn_backends[2].decode_cuda_graph_metadata[bs]

                    # Set nsa_prefill_impl for first 3 backends (required by the method)
                    for i in range(3):
                        self.attn_backends[i].set_nsa_prefill_impl(forward_batch=None)

                    # Prepare FlashMLA tensors if needed
                    flashmla_num_splits_src = None
                    flashmla_metadata_src = None
                    flashmla_num_splits_dst0 = None
                    flashmla_num_splits_dst1 = None
                    flashmla_num_splits_dst2 = None
                    flashmla_metadata_dst0 = None
                    flashmla_metadata_dst1 = None
                    flashmla_metadata_dst2 = None

                    if precomputed.flashmla_metadata is not None:
                        flashmla_num_splits_src = (
                            precomputed.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_src = (
                            precomputed.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_num_splits_dst0 = (
                            metadata0.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst1 = (
                            metadata1.flashmla_metadata.num_splits
                        )
                        flashmla_num_splits_dst2 = (
                            metadata2.flashmla_metadata.num_splits
                        )
                        flashmla_metadata_dst0 = (
                            metadata0.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst1 = (
                            metadata1.flashmla_metadata.flashmla_metadata
                        )
                        flashmla_metadata_dst2 = (
                            metadata2.flashmla_metadata.flashmla_metadata
                        )

                    # Call the multi-backend fused kernel for first 3 backends
                    fused_metadata_copy_multi_cuda(
                        # Source tensors
                        precomputed.cache_seqlens,
                        precomputed.cu_seqlens_k,
                        precomputed.page_indices,
                        precomputed.nsa_cache_seqlens,
                        precomputed.nsa_cu_seqlens_k,
                        precomputed.real_page_table,
                        flashmla_num_splits_src,
                        flashmla_metadata_src,
                        # Destination tensors for backend 0
                        metadata0.cache_seqlens_int32,
                        metadata0.cu_seqlens_k,
                        metadata0.page_table_1,
                        metadata0.nsa_cache_seqlens_int32,
                        metadata0.nsa_cu_seqlens_k,
                        (
                            metadata0.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst0,
                        flashmla_metadata_dst0,
                        # Destination tensors for backend 1
                        metadata1.cache_seqlens_int32,
                        metadata1.cu_seqlens_k,
                        metadata1.page_table_1,
                        metadata1.nsa_cache_seqlens_int32,
                        metadata1.nsa_cu_seqlens_k,
                        (
                            metadata1.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst1,
                        flashmla_metadata_dst1,
                        # Destination tensors for backend 2
                        metadata2.cache_seqlens_int32,
                        metadata2.cu_seqlens_k,
                        metadata2.page_table_1,
                        metadata2.nsa_cache_seqlens_int32,
                        metadata2.nsa_cu_seqlens_k,
                        (
                            metadata2.real_page_table
                            if precomputed.real_page_table is not None
                            else None
                        ),
                        flashmla_num_splits_dst2,
                        flashmla_metadata_dst2,
                        # Parameters
                        bs,
                        precomputed.max_len,
                        precomputed.seqlens_expanded_size,
                    )

                    # Copy remaining backends one by one (if > 3 backends)
                    for i in range(3, self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                        )
                except (ImportError, Exception) as e:
                    # Fallback to loop if multi-backend kernel not available or fails
                    if isinstance(e, ImportError):
                        print(
                            "Warning: Multi-backend fused metadata copy kernel not available, falling back to loop."
                        )
                    else:
                        print(
                            f"Warning: Multi-backend fused metadata copy kernel failed with error: {e}, falling back to loop."
                        )
                    for i in range(self.speculative_num_steps - 1):
                        self.attn_backends[
                            i
                        ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                            bs=bs,
                            precomputed=precomputed,
                            forward_mode=ForwardMode.DECODE,
                        )
            else:
                # Less than 3 backends: copy to each backend individually
                for i in range(self.speculative_num_steps - 1):
                    self.attn_backends[
                        i
                    ].init_forward_metadata_replay_cuda_graph_from_precomputed(
                        bs=bs,
                        precomputed=precomputed,
                        forward_mode=ForwardMode.DECODE,
                    )
        else:
            # Fallback: compute metadata separately for each backend
            for i in range(self.speculative_num_steps - 1):
                self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                    bs=bs,
                    req_pool_indices=forward_batch.req_pool_indices,
                    seq_lens=forward_batch.seq_lens,
                    seq_lens_sum=forward_batch.seq_lens_sum,
                    encoder_lens=None,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    out_cache_loc=None,
                )
