from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange

from sglang.jit_kernel.fused_store_index_cache import (
    can_use_nsa_fused_store,
    fused_store_index_k_cache,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import (
    add_prefix,
    ceil_align,
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
)

def _install_b12x_indexer_keyfix():
    """Stop the b12x CUTLASS-DSL indexer from recompiling on EVERY new context length.

    Root cause (proven in-container): extend_kernel._tensor_compile_key keys the OUTPUT logits
    scratch with its dim-1 size (k_rows) + dim-0 stride, so each distinct context length misses
    both the memory + on-disk JIT cache -> a full ~13-22s host stall (GPUs 0%) PER prefill. The
    compiled binary is layout-dynamic and does NOT depend on k_rows, so the recompiles are
    functionally identical. Fix: make the logits-output key shape-generic (all dims dynamic +
    neutralize the leaked strides). Result: compile ONCE per variant, cache-hit (~0.5ms) for all
    lengths, persisted on disk across restarts. Gated by B12X_INDEXER_KEYFIX (default on); fully
    fail-safe (any error -> original key, never crashes the kernel).
    """
    import os as _os
    if _os.environ.get("B12X_INDEXER_KEYFIX", "1") in ("0", "false", "False"):
        return
    _OUTPUT_KEYS = {"extend_logits", "logits"}
    mods = []
    try:
        import b12x.attention.indexer.extend_kernel as _ek
        mods.append(_ek)
    except Exception:
        pass
    try:
        import b12x.attention.indexer.kernel as _pk
        mods.append(_pk)
    except Exception:
        pass
    for mod in mods:
        if getattr(mod, "_glm_keyfix", False) or not hasattr(mod, "_tensor_compile_key"):
            continue
        _orig = mod._tensor_compile_key

        def _patched(name, tensor, *, dynamic_dims=(), _orig=_orig):
            try:
                if name in _OUTPUT_KEYS:
                    dynamic_dims = tuple(range(tensor.ndim))
                tk = _orig(name, tensor, dynamic_dims=dynamic_dims)
                if name in _OUTPUT_KEYS and any(
                    getattr(d, "kind", None) == "dynamic" for d in tk.dims
                ):
                    ns = tuple(
                        -1 if getattr(d, "kind", None) == "dynamic" else s
                        for d, s in zip(tk.dims, tk.stride)
                    )
                    tk = tk.__class__(
                        name=tk.name, dtype=tk.dtype, rank=tk.rank, dims=tk.dims,
                        stride=ns, device=tk.device, align=tk.align, layout=tk.layout,
                    )
                return tk
            except Exception:
                return _orig(name, tensor, dynamic_dims=dynamic_dims)

        mod._tensor_compile_key = _patched
        mod._glm_keyfix = True
        try:
            import logging as _l
            _l.getLogger("glm_b12x_indexer").warning(
                "B12X indexer JIT key-fix installed on %s", mod.__name__
            )
        except Exception:
            pass


try:
    _install_b12x_indexer_keyfix()
except Exception:
    pass

global _use_multi_stream
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_gfx95_supported = is_gfx95_supported()

# NextN/MTP draft global-topk index-share (vLLM index_share_for_mtp_iteration parity).
# When the EAGLE/NextN draft decode runs deeper than 1 step, its hidden state drifts from
# the verifier's, so the per-step indexer top-k drifts -> the draft attends to a DIFFERENT
# sparse KV set than the verifier -> distributions diverge -> acceptance collapses past
# ~2 tokens. vLLM freezes the selection: draft step 0 computes its top-k, steps 1+ REUSE
# step-0's cached global slots (set_skip_topk). Mirror that here: cache step-0's [rows,
# index_topk] int32 selection per Indexer layer, and on draft decode steps>=1 return the
# cached slots instead of recomputing. DEFAULT OFF, fail-safe (any mismatch -> recompute).
_NSA_DCP_DRAFT_GLOBAL_TOPK = (
    __import__("os").environ.get("SGLANG_NSA_DCP_DRAFT_GLOBAL_TOPK", "0")
    not in ("0", "false", "False", "")
)
if _is_cuda:
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if is_npu():
    import torch_npu
    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

from sglang.srt.distributed import (
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa.utils import (
    cp_all_gather_rerange_output,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_in_seq_split,
)
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool


DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_page_table_1(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 1.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return: (tokens, ), (tokens, ) int32, k_start and k_end in kv cache(token,xxx) for each token.
        """

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_indexer_seq_len(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_nsa_extend_len_cpu(self) -> List[int]:
        """
        Return: extend seq lens for each batch.
        """

    def get_token_to_batch_idx(self) -> torch.Tensor:
        """
        Return: batch idx for each token.
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    # from sgl_kernel import hadamard_transform
    if _is_hip:
        from fast_hadamard_transform import hadamard_transform
    else:
        from sglang.jit_kernel.hadamard import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class Indexer(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attn_context_model_parallel_world_size()
            self.cp_rank = get_attn_context_model_parallel_rank()
        else:
            self.cp_size = None
            self.cp_rank = None
        if _is_cuda:
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
            pp_size = get_global_server_args().pp_size
            self.logits_with_pp_recv = pp_size > 1 and not get_pp_group().is_last_rank
        else:
            self.logits_with_pp_recv = False

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.bfloat16 if _is_cuda else torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.k_norm = LayerNorm(
            self.head_dim, dtype=torch.bfloat16 if _use_aiter else torch.float32
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device=get_global_server_args().device,
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5
        # MTP draft global-topk index-share: per-layer persistent cache of the draft's
        # step-0 top-k selection ([rows, index_topk] int32 GLOBAL slots). Lazily allocated
        # (shape-static once seen) and reused across draft decode steps 1+ within one MTP
        # iteration (cuda-graph-safe: a fixed-address static buffer written in step-0's
        # captured region and read in later steps via in-graph copy_, no .item()/H2D).
        self._draft_topk_cache: Optional[torch.Tensor] = None

    @contextlib.contextmanager
    def _with_real_sm_count(self):
        # When pipeline parallelism is enabled, each PP rank initiates a recv operation after the _pp_launch_batch
        # request to receive the PP proxy tensor or output from the previous stage, occupying one SM resource.
        # Model execution runs in parallel with the recv operation, so the SMs available to the indexer must be reduced
        # by 1. Currently, the last rank starts the send result + recv request only after waiting for execution results.
        if self.logits_with_pp_recv:
            pp_recv_sm_count = 1
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.sm_count - pp_recv_sm_count
            ):
                yield
        else:
            yield

    def _weights_proj_bf16_in_fp32_out(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        # aiter (ROCm gfx95): extract the passthrough bf16 tensor from the
        # 3-tuple (fp8, scale, bf16) produced by fused_rms_fp8_group_quant,
        # avoiding an expensive FP8-to-bf16 dequantization.
        if _use_aiter and _is_gfx95_supported and isinstance(x, tuple) and len(x) == 3:
            x = x[2]
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            weight = self.weights_proj.weight
            out = torch.empty(
                (x.shape[0], weight.shape[0]),
                dtype=torch.float32,
                device=x.device,
            )
            deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, weight, out)
            return out

        if _is_hip:
            x = x.to(self.weights_proj.weight.dtype)
        weights, _ = self.weights_proj(x)
        return weights.float()

    @torch.compile(dynamic=True)
    def _project_and_scale_head_gates(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        weights = self._weights_proj_bf16_in_fp32_out(x)
        weights = weights * self.n_heads**-0.5
        return weights

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], q_scale: torch.Tensor
    ):
        weights = self._weights_proj_bf16_in_fp32_out(x)
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
    ):
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        self._update_rope_guarded(query[..., : self.rope_head_dim], q_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        elif (
            self.alt_stream is not None
            and forward_batch.nsa_cp_metadata is not None
            and self.nsa_enable_prefill_cp
        ):
            key = rotate_activation(key)
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = cp_all_gather_rerange_output(
                    key.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
            current_stream.wait_stream(self.alt_stream)
            return query, key
        else:
            query = rotate_activation(query)
            key = rotate_activation(key)

        # allgather+rerrange
        if forward_batch.nsa_cp_metadata is not None and self.nsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):
        # Compute only key, skip query
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)
        key = rotate_activation(key)

        return key

    @staticmethod
    def _update_rope_guarded(dst: torch.Tensor, src: torch.Tensor) -> None:
        # On AMD with in-place RoPE kernels, self-aliasing can occur;
        # skip write-back when src/dst tensors point to a single memory.
        if src.data_ptr() == dst.data_ptr():
            return
        dst.copy_(src)

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        # MTP draft global-topk index-share (vLLM index_share_for_mtp_iteration parity).
        # Resolve whether this is a NextN/EAGLE *draft decode* step and, if so, which step.
        # During a NextN/EAGLE draft decode, eagle_worker_v2.draft_forward sets
        #   forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
        # i.e. the per-step NativeSparseAttnBackend built by NativeSparseAttnMultiStepBackend
        # with speculative_step_id=i. We mark those per-step backends with `.is_draft = True`
        # (and stash `.model_runner`) in nsa_backend.py so we can robustly tell them apart from
        # the TARGET backend (which also carries speculative_step_id=0 by default). The whole
        # draft multi-step loop runs in ForwardMode.DECODE, so is_decode_or_idle() also keeps
        # the target VERIFIER (target_verify mode) out of this path.
        #
        # On draft decode steps>=1 we REUSE the frozen step-0 global top-k selection instead of
        # recomputing a drifting per-step selection (the cause of acceptance collapse past ~2
        # tokens). On step 0 we compute normally then cache it. Only valid at
        # speculative_eagle_topk==1 (chain/NextN topology); for topk>1 the draft expands a tree
        # whose per-step rows belong to different branches, so a row-indexed cache would alias —
        # skip the optimization there (recompute, byte-identical to the legacy path).
        # Fully fail-safe + cuda-graph-safe (static-shape buffer, in-graph copy_/clone, no
        # .item()/H2D; speculative_step_id is a Python int known at trace time and each draft
        # step is its own captured region). DEFAULT OFF via SGLANG_NSA_DCP_DRAFT_GLOBAL_TOPK.
        _draft_step = None
        if _NSA_DCP_DRAFT_GLOBAL_TOPK and forward_batch.forward_mode.is_decode_or_idle():
            _be = getattr(forward_batch, "attn_backend", None)
            _be_step = getattr(_be, "speculative_step_id", None)
            _be_is_draft = bool(getattr(_be, "is_draft", False)) or bool(
                getattr(getattr(_be, "model_runner", None), "is_draft_worker", False)
            )
            # eagle_topk==1 guard: the row-indexed cache is only valid for the chain/NextN
            # (topk==1) topology. _be.topk == speculative_eagle_topk on the per-step backend.
            _be_topk1 = int(getattr(_be, "topk", 0) or 0) <= 1
            if (
                _be is not None
                and _be_is_draft
                and _be_step is not None
                and _be_topk1
            ):
                _draft_step = int(_be_step)
        if (
            _draft_step is not None
            and _draft_step >= 1
            and self._draft_topk_cache is not None
            and self._draft_topk_cache.shape[0] == q_fp8.shape[0]
            and self._draft_topk_cache.shape[1] == self.index_topk
        ):
            # Reuse the frozen step-0 selection (already padded to q_fp8.shape[0]); skip the
            # entire logits->topk recompute. clone() so the caller's downstream in-place ops
            # never scribble the cache; it's a static-shape in-graph kernel -> graph-safe.
            return self._draft_topk_cache.clone()

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        if _is_hip:
            assert page_size == 1, "only support page size 1"
            block_tables = metadata.get_page_table_1()
        else:
            assert page_size == 64, "only support page size 64"
            # NOTE(dark): this support extend/decode/decode+graph
            block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        _is_verify_or_draft = (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        )
        if _is_verify_or_draft:
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()

        # DCP Stage 2: when the index_k pool is sharded /dcp, the indexer must score ONLY this
        # rank's owned pages. Remap the GLOBAL (block_tables, seqlens) -> RANK-LOCAL page tables +
        # owned token counts. The logits are then computed over the rank-LOCAL block_tables, but the
        # SELECTION is made GLOBAL: a candidate all-gather over the DCP group merges the rank-local
        # top-k into the GLOBAL top-k in GLOBAL KV-SLOT space (see the shard branch after the logits
        # below). This produces a [q, index_topk] int32 of GLOBAL slots IDENTICAL in format to the
        # Stage-1 (non-shard) topk_transform output, so _decode_dcp reuses page_owned_local_selection.
        # Cuda-graph-safe (pure tensor ops + one collective; dcp*index_topk*bs is static).
        local_pt1_override = None
        _local_real_pt = None
        _pool = forward_batch.token_to_kv_pool
        # MTP+DCP Stage-2 coexistence: allow target_verify into the shard-read path. The helpers
        # (dcp_local_index_paged_tables + two_stage_global_topk_paged) are row-wise, so verify's
        # expanded per-draft-token rows (page_table repeat_interleave'd per draft token) flow through
        # verbatim — each row gets its own rank-local owned pages then the global top-k merge. Keep
        # draft_extend OUT (it uses the Triton ragged fill path, not this b12x paged read).
        _shard_ok = (not _is_verify_or_draft) or forward_batch.forward_mode.is_target_verify()
        if getattr(_pool, "_shard_index", False) and _shard_ok:
            from sglang.srt.layers.attention.nsa.cp_nsa import (
                dcp_local_index_paged_tables,
            )

            block_tables, seqlens_32, local_pt1_override = dcp_local_index_paged_tables(
                block_tables, seqlens_32, _pool._dcp_rank, _pool._dcp_size, page_size
            )
            # block_tables is now this rank's COMPACTED local page table (local_real_pt); keep a
            # handle for the global-slot remap of the selected columns after the logits.
            _local_real_pt = block_tables
            max_seq_len = block_tables.shape[1] * page_size
            # the precomputed schedule_metadata was built for GLOBAL seqlens -> force a rebuild
            # from the rank-local seqlens below.
            schedule_metadata = None
        else:
            # Reuse pre-computed schedule metadata if available (from init_forward_metadata),
            # otherwise fall back to computing it here.
            schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
        if _is_cuda and schedule_metadata is None:
            # b12x metadata builder (sm120-native) — deep_gemm's asserts on sm120.
            if __import__("os").environ.get("SGLANG_NSA_B12X_LOGITS", "1") == "1":
                try:
                    from b12x.integration.indexer import (
                        build_paged_mqa_schedule_metadata as _b12x_build_meta,
                    )

                    schedule_metadata = _b12x_build_meta(
                        seqlens_32, blocksize, self.sm_count
                    )
                except Exception as _e:  # noqa
                    import logging as _l

                    _l.getLogger("glm_b12x_indexer").warning(
                        "GLM-B12X build_paged_mqa_schedule_metadata failed (%r); "
                        "falling back to deep_gemm",
                        _e,
                    )
                    schedule_metadata = None
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32, blocksize, self.sm_count
                )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 1 if _is_hip else 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        if _is_hip:
            kv_cache_fp8 = kv_cache_fp8.view(
                -1, block_kv, num_heads_kv, head_dim_with_sf
            )
        else:
            kv_cache_fp8 = kv_cache_fp8.view(
                kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
            )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        # When attn_tp_size > 1 or in the MAX_LEN padding mode, padding may exist in the hidden states,
        # and it is necessary to extract the actual q length.
        q_offset = sum(metadata.get_nsa_extend_len_cpu())
        if _is_hip:
            from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

            batch_size, next_n, heads, _ = q_fp8.shape
            logits = torch.full(
                (batch_size * next_n, max_seq_len),
                float("-inf"),
                device=q_fp8.device,
                dtype=torch.float32,
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                logits,
                seqlens_32,
                block_tables,
                max_seq_len,
                Preshuffle=False,
                KVBlockSize=block_kv,
                ChunkK=128,
                TotalCuCount=256,
                WavePerEU=5,
            )
        else:
            # GLM-DSA on sm120: DeepGEMM's fp8_paged_mqa_logits asserts
            # "Unsupported architecture". b12x has a native sm120 indexer; route to
            # it. Falls back to deep_gemm (and logs why) if the b12x call mismatches.
            _b12x_logits = None
            if __import__("os").environ.get("SGLANG_NSA_B12X_LOGITS", "1") == "1":
                try:
                    from b12x.integration.indexer import (
                        build_paged_mqa_schedule_metadata as _b12x_build_meta,
                        paged_decode_logits as _b12x_paged_decode_logits,
                    )
                    from b12x.attention.indexer.api import (
                        IndexerPagedDecodeMetadata as _B12XPagedMeta,
                    )

                    _sched = schedule_metadata
                    if _sched is None:
                        _sched = _b12x_build_meta(seqlens_32, block_kv, self.sm_count)
                    # b12x wants q_fp8 rank-3 (q_rows, heads, head_dim); SGLang
                    # unsqueeze(1)'d a next_n=1 dim in for deep_gemm — squeeze it out.
                    _q_b12x = q_fp8[:q_offset]
                    if _q_b12x.ndim == 4 and _q_b12x.shape[1] == 1:
                        _q_b12x = _q_b12x.squeeze(1)
                    # b12x wants index_k_cache rank-2 (num_pages, page*(head+scale));
                    # SGLang view()'d it to 4D (n,64,1,132) for deep_gemm — flatten.
                    _ikc_b12x = kv_cache_fp8
                    if _ikc_b12x.ndim != 2:
                        _ikc_b12x = _ikc_b12x.reshape(_ikc_b12x.shape[0], -1)
                    _b12x_logits = _b12x_paged_decode_logits(
                        q_fp8=_q_b12x,
                        weights=weights[:q_offset],
                        index_k_cache=_ikc_b12x,
                        metadata=_B12XPagedMeta(
                            real_page_table=block_tables,
                            cache_seqlens_int32=seqlens_32,
                            paged_mqa_schedule_metadata=_sched,
                        ),
                        page_size=page_size,
                    )
                except Exception as _e:  # noqa
                    import logging as _l

                    _l.getLogger("glm_b12x_indexer").warning(
                        "GLM-B12X indexer: paged_decode_logits failed (%r); "
                        "falling back to deep_gemm",
                        _e,
                    )
                    _b12x_logits = None
            if _b12x_logits is not None:
                logits = _b12x_logits
            else:
                logits = deep_gemm.fp8_paged_mqa_logits(
                    q_fp8[:q_offset],
                    kv_cache_fp8,
                    weights[:q_offset],
                    seqlens_32,
                    block_tables,
                    schedule_metadata,
                    max_seq_len,
                    clean_logits=False,
                )

        # NOTE(dark): logits should be cleaned in topk_transform
        if _local_real_pt is not None:
            # DCP Stage 2 (GLOBAL selection): the logits were computed over this rank's RANK-LOCAL
            # page table, so a plain top-k would be rank-local. Instead merge the rank-local top-k
            # into the GLOBAL top-k in GLOBAL KV-SLOT space via a single candidate all-gather over
            # the DCP group. The result is [q, index_topk] int32 of GLOBAL slots (-1 padded),
            # IDENTICAL in format to the Stage-1 topk_transform output -> _decode_dcp's
            # page_owned_local_selection carves each rank's owned share verbatim. seqlens_32 is the
            # rank-LOCAL owned-token count (the valid logit-column count) here.
            from sglang.srt.layers.attention.nsa.cp_nsa import (
                two_stage_global_topk_paged,
            )
            from sglang.srt.layers.dp_attention import (
                get_attention_tp_group as _dcp_gatg,
            )

            _dcp_pg = getattr(_dcp_gatg(), "device_group", None)
            topk_result = two_stage_global_topk_paged(
                logits,
                seqlens_32,
                _local_real_pt,
                _pool._dcp_rank,
                _pool._dcp_size,
                page_size,
                self.index_topk,
                cp_group=_dcp_pg,
            )
        else:
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
            )
        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
            padding = torch.full(
                (pad_len, topk_result.shape[1]),
                -1,
                dtype=topk_result.dtype,
                device=topk_result.device,
            )
            topk_result = torch.cat([topk_result, padding], dim=0)
        # MTP draft global-topk index-share: on a draft decode step (typically step 0, or any
        # step where the cache was stale/absent so we recomputed), freeze this selection so the
        # remaining deeper steps in THIS MTP iteration reuse it. The whole draft multi-step loop
        # is captured as ONE cuda graph (eagle_draft_cuda_graph_runner.run_once -> draft_forward),
        # so the step-0 clone() and the step>=1 reader's clone() bind to consistent allocations
        # within that single graph -> cuda-graph-safe. detach() keeps autograd out (inference).
        if _draft_step is not None:
            self._draft_topk_cache = topk_result.detach().clone()
        return topk_result

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device: torch.device
    ) -> Tuple[bool, int]:
        """
        Detect whether we need to chunk the MQA logits computation to avoid OOM
        Return: (need_chunk, free_mem)
        """
        # Quick static check for normal batches
        if num_q * num_k < 8_000_000:  # 8M elements ≈ 32MB logits
            return False, 0

        free_mem, total_mem = torch.cuda.mem_get_info(device)
        bytes_per_elem = 4  # float32
        logits_bytes = num_q * num_k * bytes_per_elem

        # Logits should not exceed 50% of free memory or 30% of total memory
        need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
        return need_chunk, free_mem

    def _ragged_mqa_logits(self, q_fp8_slice, kv_fp8, weights_slice, ks_slice, ke_slice):
        """Ragged/prefill indexer MQA logits.

        GLM-OPTIONB (2026-06-19): the original DSA port wired only the PAGED/decode
        indexer to b12x; this RAGGED/prefill path still called deep_gemm.fp8_mqa_logits,
        which asserts "Unsupported architecture" on sm120 -> any prompt longer than
        index_topk (2048) crashed. b12x has the sm120-native equivalent
        (b12x.attention.indexer.extend_logits + IndexerExtendMetadata). Route to it
        (gated by SGLANG_NSA_B12X_LOGITS, default on), fall back to deep_gemm/aiter.
        """
        if not _is_hip and __import__("os").environ.get("SGLANG_NSA_B12X_LOGITS", "1") == "1":
            try:
                from b12x.attention.indexer.api import (
                    extend_logits as _b12x_extend_logits,
                    IndexerExtendMetadata as _B12XExtendMeta,
                )

                _q = q_fp8_slice
                if _q.ndim == 4 and _q.shape[1] == 1:
                    _q = _q.squeeze(1)
                # CHUNK-PADDING FIX: the b12x prefill512 extend kernel writes the
                # (q_rows, k_total_rows) fp32 logits tile with st.global.v4.f32 (16B) stores at
                # base q_row*k_total_rows, requiring k_total_rows % 4 == 0. k_total_rows =
                # kv_fp8[0].shape[0] = the running KV length. chunk=1024 keeps it %4; chunk>=2048's
                # partial tail leaves it arbitrary -> CUDA misaligned address. Pad the KV token
                # count up to a multiple of 16 (covers the fp32 v4 store + fp8 K tile), run the
                # kernel on the padded width, then SLICE logits back to the true K before topk
                # (padded cols are beyond k_end and discarded -> correctness preserved). Gated;
                # default ON since it's a pure-safety pad. Lets chunk 2048/4096/8192 run.
                _kv_t, _scale_t = kv_fp8
                _k_real = int(_kv_t.shape[0])
                _do_pad = (
                    __import__("os").environ.get("SGLANG_NSA_PAD_EXTEND_KWIDTH", "1") == "1"
                    and (_k_real % 16) != 0
                )
                if _do_pad:
                    _k_pad = ((_k_real + 15) // 16) * 16
                    _n = _k_pad - _k_real
                    _kv_p = torch.cat(
                        [_kv_t, _kv_t.new_zeros((_n,) + tuple(_kv_t.shape[1:]))], dim=0
                    )
                    _scale_p = torch.cat(
                        [_scale_t, _scale_t.new_zeros((_n,) + tuple(_scale_t.shape[1:]))], dim=0
                    )
                    _kv_in = (_kv_p, _scale_p)
                else:
                    _kv_in = kv_fp8
                _lg = _b12x_extend_logits(
                    q_fp8=_q,
                    weights=weights_slice,
                    kv_fp8=_kv_in,
                    metadata=_B12XExtendMeta(k_start=ks_slice, k_end=ke_slice),
                )
                if _do_pad and _lg.shape[-1] != _k_real:
                    _lg = _lg[..., :_k_real]
                return _lg
            except Exception as _e:  # noqa
                import logging as _l

                _l.getLogger("glm_b12x_indexer").warning(
                    "GLM-B12X extend_logits failed (%r); falling back to deep_gemm", _e
                )
        if _is_hip:
            from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

            _kv, _scale = kv_fp8
            return fp8_mqa_logits(
                q_fp8_slice, _kv, _scale, weights_slice, ks_slice, ke_slice
            )
        return deep_gemm.fp8_mqa_logits(
            q_fp8_slice, kv_fp8, weights_slice, ks_slice, ke_slice, clean_logits=False
        )

    def _get_topk_ragged(
        self,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        assert forward_batch.forward_mode.is_extend_without_speculative()

        page_size = forward_batch.token_to_kv_pool.page_size
        if _is_hip:
            assert page_size == 1, "only support page size 1"
        else:
            assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        weights = weights.squeeze(-1)

        if _is_hip:
            block_tables = metadata.get_page_table_1()
        else:
            block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        batch_size = len(block_tables)
        token_nums, _, _ = q_fp8.shape
        device = q_fp8.device

        topk_result = torch.full(
            (token_nums, self.index_topk), -1, device=device, dtype=torch.int32
        )
        if batch_size == 0:
            return topk_result

        # DCP Stage 2 (prefill), vLLM cp_gather: index_k is sharded /dcp for STORAGE (capacity win),
        # but the ragged/extend indexer kernel derives token POSITION from packed-K column order, so it
        # CANNOT run over compacted-owned pages (the abandoned candidate-gather approach scrambled
        # positions -> recency-window collapse). Instead GATHER the sharded index_k back into a
        # transient GLOBAL physical-page-order view, then run the EXACT non-shard ragged path (global
        # block_tables + global ks/ke + plain topk_transform) — correct by construction. Eager prefill,
        # so the all_gather + host syncs are fine; vLLM pays the same O(ctx) gather + O(ctx^2) indexer.
        _pool_r = forward_batch.token_to_kv_pool
        _shard_r = getattr(_pool_r, "_shard_index", False)
        # NOTE: cp_gather keeps the indexer GLOBAL, so local_pt1_override / seqlens_expanded_r stay None
        # (the non-shard latent-slot mapping via attn_metadata.page_table_1 is correct as-is).
        local_pt1_override = None
        seqlens_expanded_r = None
        _index_buf_override = None
        # GLOBAL ragged metadata (identical for shard + non-shard — this is the whole point of cp_gather)
        ks, ke = metadata.get_indexer_kvcache_range()
        indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
        seq_len_sum = torch.sum(indexer_seq_lens_cpu).item()
        max_seq_len = torch.max(indexer_seq_lens_cpu).item()
        indexer_seq_len_dev = metadata.get_indexer_seq_len()
        if _shard_r:
            # Gather this LAYER's sharded index_k into a transient GLOBAL physical-page-order buffer and
            # rewrite block_tables to the compacted dense page ids that address it. The gather runs PER
            # LAYER (each F-layer's index_k page content differs); the union/searchsorted geometry is
            # cheap relative to the per-layer all_gather + O(ctx^2) logits. Eager prefill: collectives
            # + .item() host syncs are fine (this path is never cuda-graph-captured).
            from sglang.srt.layers.attention.nsa.cp_nsa import (
                dcp_gather_global_index_k,
            )
            from sglang.srt.layers.dp_attention import (
                get_attention_tp_group as _dcp_gatg_r,
            )

            _dcp_pg_r = getattr(_dcp_gatg_r(), "device_group", None)
            local_index_buf = _pool_r.get_index_k_with_scale_buffer(layer_id=layer_id)
            _index_buf_override, block_tables = dcp_gather_global_index_k(
                local_index_buf,
                block_tables,
                indexer_seq_len_dev,
                _pool_r._dcp_rank,
                _pool_r._dcp_size,
                page_size,
                cp_group=_dcp_pg_r,
            )
        k_fp8, k_scale = _pool_r.get_index_k_scale_buffer(
            layer_id,
            indexer_seq_len_dev,
            block_tables,
            seq_len_sum,
            max_seq_len,
            buf_override=_index_buf_override,
        )
        if _is_fp8_fnuz:
            k_fp8 = k_fp8.view(torch.float8_e4m3fnuz)
        else:
            k_fp8 = k_fp8.view(torch.float8_e4m3fn)

        k_scale = k_scale.view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)

        # Check if we need to chunk to avoid OOM.
        # cp_gather: the indexer is GLOBAL (gathered global-order index_k), so seq_lens_expanded is the
        # GLOBAL per-query causal length for BOTH shard and non-shard (seqlens_expanded_r is None here).
        seq_lens_expanded = metadata.get_seqlens_expanded()
        token_to_batch_idx = metadata.get_token_to_batch_idx()
        q_offset = ks.shape[0]
        k_offset = k_fp8.shape[0]
        need_chunk, free_mem = self._should_chunk_mqa_logits(q_offset, k_offset, device)

        if not need_chunk:
            assert q_fp8[:q_offset].shape[0] != 0
            with self._with_real_sm_count():
                logits = self._ragged_mqa_logits(
                    q_fp8[:q_offset], kv_fp8, weights[:q_offset], ks, ke
                )
            assert logits.shape[0] == len(seq_lens_expanded)
            assert logits.shape[1] == k_offset

            # NOTE: do NOT pass batch_idx_list here. The prefill topk kernel
            # (topk_transform_prefill_kernel, topk.cu:301-329) indexes page_table_size_1 by SEQUENCE
            # via cu_seqlens_q (per-sequence cumulative) with a RELATIVE column (col - row_start), so
            # the per-SEQUENCE local_pt1 [B_seq, ...] is already correct. Passing batch_idx_list would
            # expand it to Q rows and trip TORCH_CHECK(src_page_table.size(0)==prefill_bs) -> crash.
            # (Verified against the real csrc/elementwise/topk.cu by the dcp-fix workflow.)
            # cp_gather: GLOBAL transform — ke_offset=None (topk_transform falls back to the global
            # get_seqlens_expanded()) and page_table_1_override=None (global attn_metadata.page_table_1
            # gives the correct GLOBAL latent slots). Both shard + non-shard take this identical path.
            raw_topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                ke_offset=None,
                page_table_1_override=None,
            )
            topk_result[:q_offset] = raw_topk_result
            return topk_result

        # Chunk path
        bytes_per_elem = 4  # float32
        bytes_per_row = k_offset * bytes_per_elem
        # Reserve 50% of free memory for logits
        max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
        max_rows = min(max_rows, q_offset)

        global_topk_offset = metadata.attn_metadata.topk_indices_offset

        assert (
            seq_lens_expanded.shape[0] == q_offset
        ), f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}"
        if global_topk_offset is not None:
            assert (
                global_topk_offset.shape[0] >= q_offset
            ), f"topk_indices_offset too short: {global_topk_offset.shape[0]} < {q_offset}"

        start = 0
        while start < q_offset:
            end = min(start + max_rows, q_offset)

            with self._with_real_sm_count():
                logits_chunk = self._ragged_mqa_logits(
                    q_fp8[start:end],
                    kv_fp8,
                    weights[start:end],
                    ks[start:end],
                    ke[start:end],
                )

            lengths_chunk = seq_lens_expanded[start:end]

            # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
            if global_topk_offset is not None:
                # RAGGED path
                topk_offset_chunk = global_topk_offset[start:end]
                cu_seqlens_q_chunk = None
                batch_idx_chunk = None
            else:
                # PAGED path: treat each token as a length-1 sequence
                topk_offset_chunk = None
                B_chunk = logits_chunk.shape[0]
                cu_seqlens_q_chunk = torch.ones(
                    B_chunk, dtype=torch.int32, device=device
                )
                batch_idx_chunk = token_to_batch_idx[start:end]

            raw_topk_chunk = metadata.topk_transform(
                logits_chunk,
                self.index_topk,
                ks=ks[start:end],
                cu_seqlens_q=cu_seqlens_q_chunk,
                ke_offset=lengths_chunk,
                batch_idx_list=batch_idx_chunk,
                topk_indices_offset_override=topk_offset_chunk,
                page_table_1_override=local_pt1_override,
            )
            topk_result[start:end] = raw_topk_chunk
            start = end

        return topk_result

    def _forward_cuda_k_only(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        enable_dual_stream: bool,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert forward_batch.forward_mode.is_extend_without_speculative()
        x_meta = x[0] if isinstance(x, tuple) else x

        # Fast path: only compute and store k cache, skip all q and weights ops
        key = self._get_k_bf16(x, positions, enable_dual_stream)

        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

        self._store_index_k_cache(
            forward_batch=forward_batch,
            layer_id=layer_id,
            key=key,
            act_quant=act_quant,
        )

        # MHA doesn't need topk_indices
        if not return_indices:
            return None

        # MLA: use dummy logits with topk kernel's fast path to generate indices
        # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=x_meta.device,
        )
        return metadata.topk_transform(dummy_logits, self.index_topk)

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: List[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        if cp_index is not None:
            # TODO Multi-batch support has accuracy issues
            for batch_idx, start_seq_position, end_seq_position in cp_index:
                pre_chunk_offset = (
                    forward_batch.seq_lens_cpu[batch_idx].item()
                    - forward_batch.extend_seq_lens_cpu[batch_idx]
                )
                start_seq_position += pre_chunk_offset
                end_seq_position += pre_chunk_offset
                if offset == 0 and batch_idx != 0:
                    offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
                k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )
                k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )

                extend_seq_len = end_seq_position - start_seq_position
                ks = torch.full(
                    (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
                )
                k_fp8_list.append(k_fp8)
                k_scale_list.append(k_scale)
                ks_list.append(ks)
                ke_offset = torch.arange(
                    start_seq_position + 1,
                    end_seq_position + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                ke_offset_list.append(ke_offset)
                actual_seq_q = torch.tensor(
                    [extend_seq_len], dtype=torch.int32, device="cuda"
                )
                actual_seq_q_list.append(actual_seq_q)
                batch_idx_list.append(batch_idx)

            k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
            k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.cat(ks_list, dim=0)
            ke_offset = torch.cat(ke_offset_list, dim=0)
            ke = ks + ke_offset
            actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
            with self._with_real_sm_count():
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8,
                    kv_fp8,
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
                batch_idx_list=batch_idx_list,
            )
        else:
            kv_len = (
                forward_batch.seq_lens_cpu[0].item()
                - forward_batch.extend_seq_lens_cpu[0]
                + kv_len
            )
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
            ke_offset = torch.arange(
                (kv_len - actual_seq_q) + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke = ks + ke_offset

            with self._with_real_sm_count():
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8,
                    kv_fp8,
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
            )

        return topk_result

    def forward_indexer(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if not _is_npu:
            from sglang.srt.layers.attention.nsa.tilelang_kernel import fp8_index

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        # logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        k_fp8_list = []
        k_scale_list = []

        topk_indices_list = []

        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided_indices = torch.arange(
            0, block_tables.shape[-1], page_size, device="cuda"
        )
        block_tables = block_tables[:, strided_indices] // page_size

        q_len_start = 0

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()
            q_len = (
                forward_batch.extend_seq_lens_cpu[i]
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            q_len_end = q_len_start + q_len

            q_fp8_partial = q_fp8[q_len_start:q_len_end]
            q_fp8_partial = q_fp8_partial.unsqueeze(0).contiguous()

            weights_partial = weights[q_len_start:q_len_end]
            weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
            k_scale = k_scale.view(torch.float32).squeeze(-1).unsqueeze(0).contiguous()

            index_score = fp8_index(
                q_fp8_partial,
                weights_partial,
                k_fp8,
                k_scale,
            )
            end_pos = seq_len
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            pad_len = ceil_align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end

        topk_indices = torch.cat(topk_indices_list, dim=0)
        return topk_indices

    def _store_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        key: torch.Tensor,
        *,
        act_quant=None,  # fallback only
    ) -> None:
        """
        Store NSA indexer K cache for current step.

        Preferred: fused_store_index_k_cache(key, cache, out_cache_loc, page_size)
        Fallback : act_quant(key) + token_to_kv_pool.set_index_k_scale_buffer(...)
        """

        # DCP Stage 2: remap the GLOBAL out_cache_loc -> rank-LOCAL index_k slots (page-ownership)
        # ONCE here, covering BOTH the fused and fallback stores. No-op unless index sharding is on.
        # IMPORTANT: do NOT mutate forward_batch.out_cache_loc (the latent set_mla_kv_buffer reads it
        # and applies its own remap); build a private remapped copy instead.
        pool = forward_batch.token_to_kv_pool
        idx_loc = pool.dcp_remap_index_loc(forward_batch.out_cache_loc)

        # Fast path: JIT fused store (CUDA, page_size=64, non-fnuz)
        if (
            _is_cuda
            and (not _is_fp8_fnuz)
            and can_use_nsa_fused_store(
                key.dtype,
                idx_loc.dtype,
                pool.page_size,
            )
        ):
            # NOTE: wrapper already normalizes shape/contiguity and asserts dtypes.
            buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
            fused_store_index_k_cache(
                key,
                buf,
                idx_loc.contiguous() if not idx_loc.is_contiguous() else idx_loc,
                pool.page_size,
            )
            return

        # Fallback: original path
        assert act_quant is not None
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        out_loc = idx_loc
        if not out_loc.is_contiguous():
            out_loc = out_loc.contiguous()

        pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=out_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if _is_hip:
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
        elif not _is_npu:
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        # When upstream uses fused FP8 RMSNorm+quant, activations may be passed as
        # a tuple like (x_fp8, x_scale[, y]). Use `x_meta` for shape/device queries.
        x_meta = x[0] if isinstance(x, tuple) else x

        metadata = forward_batch.attn_backend.get_indexer_metadata(
            layer_id, forward_batch
        )

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        # Determine if should skip topk based on sequence length
        # We can only skip the logits computation if cuda graph is not involved
        skip_logits_computation = False
        if forward_batch.forward_mode.is_extend_without_speculative():
            if forward_batch.seq_lens_cpu is not None:
                max_kv_len = forward_batch.seq_lens_cpu.max().item()
                skip_logits_computation = max_kv_len <= self.index_topk

        # Optimization: fast path when skipping topk computation
        if skip_logits_computation and (not self.nsa_enable_prefill_cp):
            return self._forward_cuda_k_only(
                x,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                enable_dual_stream,
                metadata,
                return_indices,
            )

        if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            weights = self._project_and_scale_head_gates(x)
            query, key = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )
            current_stream.wait_stream(self.alt_stream)
            weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        else:
            query, key = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )

            if enable_dual_stream:
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)

                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                with torch.cuda.stream(self.alt_stream):
                    self._store_index_k_cache(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                        key=key,
                        act_quant=act_quant,
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )

            # aiter (ROCm gfx95): the 3-tuple (fp8, scale, bf16) from
            # fused_rms_fp8_group_quant is passed directly to _get_logits_head_gate,
            # which extracts the bf16 tensor via _weights_proj_bf16_in_fp32_out,
            # completely skipping the FP8 dequantization path below.
            if (
                _use_aiter
                and _is_gfx95_supported
                and isinstance(x, tuple)
                and len(x) == 3
            ):
                x_for_gate = x
            elif isinstance(x, tuple):
                assert len(x) in (
                    2,
                    3,
                ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
                x_q, x_s = x[0], x[1]
                if (
                    x_s is not None
                    and x_q.dim() == 2
                    and x_s.dim() == 2
                    and x_q.shape[0] == x_s.shape[0]
                ):
                    m, n = x_q.shape
                    ng = x_s.shape[1]
                    if ng > 0 and n % ng == 0:
                        group = n // ng
                        x_for_gate = (
                            x_q.to(torch.float32)
                            .view(m, ng, group)
                            .mul_(x_s.to(torch.float32).unsqueeze(-1))
                            .view(m, n)
                            .to(torch.bfloat16)
                        )
                    else:
                        x_for_gate = x_q.to(torch.bfloat16)
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x

            weights = self._get_logits_head_gate(x_for_gate, q_scale)

        if _is_cuda or _is_hip:
            assert forward_batch.seq_lens_cpu is not None
            if len(forward_batch.seq_lens_cpu) == 0:
                # this seems b/c max-pad, no worries?
                # if x.shape[0] != 0:
                #     print(
                #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                #     )
                return torch.full(
                    (x_meta.shape[0], self.index_topk),
                    -1,
                    dtype=torch.int,
                    device=x_meta.device,
                )

            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                if (
                    forward_batch.nsa_cp_metadata is not None
                    and is_nsa_prefill_cp_in_seq_split()
                ):
                    kv_len_prev = forward_batch.nsa_cp_metadata.kv_len_prev
                    kv_len_next = forward_batch.nsa_cp_metadata.kv_len_next
                    actual_seq_q_prev = forward_batch.nsa_cp_metadata.actual_seq_q_prev
                    actual_seq_q_next = forward_batch.nsa_cp_metadata.actual_seq_q_next

                    # TODO support mutil-batch
                    # cp_batch_seq_index_prev = forward_batch.nsa_cp_metadata["cp_batch_seq_index_prev"]
                    # cp_batch_seq_index_next = forward_batch.nsa_cp_metadata["cp_batch_seq_index_next"]
                    # TODO prev, next, combined into a single call
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )

                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    return torch.cat([topk_result_prev, topk_result_next], dim=0)
                else:
                    topk_result = self._get_topk_ragged(
                        enable_dual_stream,
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                    )
        else:
            topk_result = self.forward_indexer(
                q_fp8.contiguous(),
                weights,
                forward_batch,
                topk=self.index_topk,
                layer_id=layer_id,
            )
        return topk_result

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        )

        bs = q_lora.shape[0]

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                    wq_b_event = self.alt_stream.record_event()
                    q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [bs, 64, 64 + 64]
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )  # [bs, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [bs, 64, 64 + 64]
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )  # [bs, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)

            if layer_id == 0:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.nsa_cp_metadata is not None
        ):
            k = cp_all_gather_rerange_output(
                k.contiguous().view(-1, self.head_dim),
                self.cp_size,
                forward_batch,
                torch.npu.current_stream(),
            )

        forward_batch.token_to_kv_pool.set_index_k_buffer(
            layer_id, forward_batch.out_cache_loc, k
        )
        if is_prefill:
            if self.nsa_enable_prefill_cp and forward_batch.nsa_cp_metadata is not None:
                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q = (
                    forward_batch.nsa_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.nsa_cp_metadata.actual_seq_q_next_tensor,
                )
                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
                    forward_batch.nsa_cp_metadata.kv_len_prev_tensor,
                    forward_batch.nsa_cp_metadata.kv_len_next_tensor,
                )
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv
                )
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                ):
                    num_draft_tokens = (
                        forward_batch.attn_backend.speculative_num_draft_tokens
                    )
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = forward_batch.attn_backend.forward_metadata.block_tables
        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.nsa_cp_metadata is not None
        ):
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
            return topk_indices
        else:
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )

            topk_indices = torch_npu.npu_lightning_indexer(
                query=q.view(-1, self.n_heads, self.head_dim),
                key=past_key_states,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                    torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
            return topk_indices[0]

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return topk_indices_prev[0], topk_indices_next[0]


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
