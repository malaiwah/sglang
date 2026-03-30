#!/usr/bin/env python3
"""
Standalone correctness test for fused PCIe allreduce + GemmaRMSNorm kernel.
Run with:  torchrun --nproc_per_node=8 test_fused_gemma_rmsnorm.py
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
import sglang.srt.distributed.device_communicators.pcie_allreduce as pcie_ops
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary


def create_shared_buffer(size_in_bytes, group, rank, world_size):
    lib = CudaRTLibrary()
    pointer = lib.cudaMalloc(size_in_bytes)
    handle = lib.cudaIpcGetMemHandle(pointer)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=group)
    pointers = []
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)
        else:
            pointers.append(lib.cudaIpcOpenMemHandle(h).value)
    return pointers


def reference_gemma_rmsnorm(x_allreduced, residual, weight, eps):
    combined = x_allreduced + residual
    new_residual = combined.clone()
    v = combined.float()
    variance = v.pow(2).mean(-1, keepdim=True)
    normed = v * torch.rsqrt(variance + eps)
    normed = normed * (1.0 + weight.float())
    return normed.to(combined.dtype), new_residual


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    max_size = 512 * 1024
    meta_size_bytes = pcie_ops.meta_size() + max_size

    meta_ptrs = create_shared_buffer(meta_size_bytes, dist.group.WORLD, rank, world_size)
    rank_data = torch.empty(max_size, dtype=torch.uint8, device=device)
    ptr = pcie_ops.init_custom_ar(meta_ptrs, rank_data, rank)

    buf0 = create_shared_buffer(max_size, dist.group.WORLD, rank, world_size)
    buf1 = create_shared_buffer(max_size, dist.group.WORLD, rank, world_size)
    pcie_ops.register_pcie_buffers(ptr, buf0, buf1)

    if rank == 0:
        print(f"PCIe allreduce initialized: world_size={world_size}")
        print()

    hidden_size = 5120
    eps = 1e-6
    passed = 0
    failed = 0

    for num_tokens in [1, 2, 4, 8, 16, 32]:
        for trial in range(3):
            seed = 1000 * num_tokens + trial

            torch.manual_seed(seed + rank)
            x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

            torch.manual_seed(seed + 9999)
            weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device) * 0.02
            residual = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

            # Reference: gather all x, sum, then GemmaRMSNorm.
            x_cpu = x.cpu()
            all_x_cpu = [torch.empty_like(x_cpu) for _ in range(world_size)]
            dist.all_gather(all_x_cpu, x_cpu)
            x_sum = torch.stack(all_x_cpu).sum(0).to(device)
            ref_out, ref_res = reference_gemma_rmsnorm(
                x_sum, residual.clone(), weight, eps
            )

            # Fused kernel.
            inp = x.clone()
            res_in = residual.clone()
            out = torch.empty_like(inp)
            res_out = torch.empty_like(inp)

            pcie_ops.allreduce_gemma_rmsnorm(
                ptr, inp, res_in, out, res_out, weight, eps, 0, 0
            )
            torch.cuda.synchronize()

            out_diff = (out.float() - ref_out.float()).abs().max().item()
            res_diff = (res_out.float() - ref_res.float()).abs().max().item()

            # bf16 sum across 8 GPUs gives ~1-2 ULP rounding vs CPU reference.
            tol = 0.1
            ok = out_diff < tol and res_diff < tol
            if ok:
                passed += 1
            else:
                failed += 1

            if rank == 0:
                tag = "PASS" if ok else "FAIL"
                print(
                    f"  tokens={num_tokens:>2} trial={trial} "
                    f"out_maxdiff={out_diff:.4e} res_maxdiff={res_diff:.4e}  {tag}"
                )
                if not ok:
                    print(f"    ref_out[:8]  = {ref_out[0, :8].tolist()}")
                    print(f"    fused_out[:8]= {out[0, :8].tolist()}")
                    print(f"    ref_res[:8]  = {ref_res[0, :8].tolist()}")
                    print(f"    fused_res[:8]= {res_out[0, :8].tolist()}")

    if rank == 0:
        print(f"\n{'='*40}")
        print(f"passed={passed}  failed={failed}")
        if failed:
            print("SOME TESTS FAILED")
        else:
            print("ALL TESTS PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
