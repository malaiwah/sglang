"""Small-scale multi-GPU test of the PCIe allreduce kernel."""

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda_ipc import CudaIPC


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    import pcie_allreduce

    ipc = CudaIPC()
    max_size = 56 * 1024
    meta_sz = pcie_allreduce.meta_size()

    meta_ptr = ipc.malloc(meta_sz + max_size)
    ipc.memset(meta_ptr, 0, meta_sz + max_size)

    meta_handle = ipc.ipc_get_handle(meta_ptr)

    all_meta_handles = [None] * world_size
    dist.all_gather_object(all_meta_handles, meta_handle)

    meta_ptrs = []
    for i in range(world_size):
        if i == rank:
            meta_ptrs.append(meta_ptr.value)
        else:
            meta_ptrs.append(ipc.ipc_open_handle(all_meta_handles[i]).value)

    buf_ptr = ipc.malloc(max_size)
    buf_handle = ipc.ipc_get_handle(buf_ptr)

    all_buf_handles = [None] * world_size
    dist.all_gather_object(all_buf_handles, buf_handle)

    buf_ptrs = []
    for i in range(world_size):
        if i == rank:
            buf_ptrs.append(buf_ptr.value)
        else:
            buf_ptrs.append(ipc.ipc_open_handle(all_buf_handles[i]).value)

    rank_data = torch.empty(max_size, dtype=torch.uint8, device=f"cuda:{rank}")
    fa = pcie_allreduce.init_custom_ar(meta_ptrs, rank_data, rank)
    pcie_allreduce.register_buffer(fa, buf_ptrs)

    # --- Test cases ---
    test_cases = [
        ("bf16 [1, 7168] (14KB hidden state)", torch.bfloat16, (1, 7168)),
        ("bf16 [1, 1024]", torch.bfloat16, (1, 1024)),
        ("fp16 [1, 7168]", torch.float16, (1, 7168)),
        ("fp32 [1, 512]", torch.float32, (1, 512)),
    ]

    all_pass = True
    for name, dtype, shape in test_cases:
        inp = torch.full(shape, float(rank + 1), dtype=dtype, device=f"cuda:{rank}")
        out = torch.zeros_like(inp)

        pcie_allreduce.all_reduce(fa, inp, out, buf_ptrs[rank], max_size)
        torch.cuda.synchronize()

        expected = world_size * (world_size + 1) / 2.0
        max_err = (out.float() - expected).abs().max().item()
        passed = max_err < 0.5
        all_pass = all_pass and passed

        if rank == 0:
            print(
                f"  {name}: out[0]={out.view(-1)[0].item():.2f} "
                f"expected={expected:.1f} max_err={max_err:.4f} "
                f"{'PASS' if passed else 'FAIL'}"
            )

    # Test multiple rounds to check barrier counter wrapping.
    if rank == 0:
        print("  Running 100 iterations for barrier stability...")
    for i in range(100):
        inp = torch.full((1, 7168), float(rank + 1), dtype=torch.bfloat16, device=f"cuda:{rank}")
        out = torch.zeros_like(inp)
        pcie_allreduce.all_reduce(fa, inp, out, buf_ptrs[rank], max_size)
    torch.cuda.synchronize()
    expected = world_size * (world_size + 1) / 2.0
    max_err = (out.float() - expected).abs().max().item()
    passed = max_err < 0.5
    all_pass = all_pass and passed
    if rank == 0:
        print(f"  100 iterations: max_err={max_err:.4f} {'PASS' if passed else 'FAIL'}")

    pcie_allreduce.dispose(fa)
    dist.barrier()

    if rank == 0:
        print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Testing PCIe allreduce with {world_size} GPUs (sys-scope barriers)\n")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
