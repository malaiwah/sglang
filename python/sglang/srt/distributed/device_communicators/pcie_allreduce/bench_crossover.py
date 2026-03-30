"""Find the PCIe custom allreduce vs NCCL crossover point.

Sweeps message sizes from 1KB to 2MB, measures median latency for both
backends, and prints a comparison table. Run with:

    python bench_crossover.py [num_gpus]
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
# Parent dir so spawned workers can `import pcie_allreduce`.
sys.path.insert(0, os.path.dirname(_this_dir))
from cuda_ipc import CudaIPC

WARMUP = 200
ITERS = 2000
HIDDEN = 7168


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29520"
    torch.cuda.set_device(rank)

    sys.path.insert(0, _this_dir)
    sys.path.insert(0, os.path.dirname(_this_dir))

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Also init gloo for the IPC buffer setup.
    gloo_group = dist.new_group(backend="gloo")

    import pcie_allreduce

    # ---- Setup custom AR (needs large enough buffers for the sweep) ----
    max_buf = 2 * 1024 * 1024  # 2MB — upper bound of sweep
    ipc = CudaIPC()
    meta_sz = pcie_allreduce.meta_size()

    meta_ptr = ipc.malloc(meta_sz + max_buf)
    ipc.memset(meta_ptr, 0, meta_sz + max_buf)
    meta_handle = ipc.ipc_get_handle(meta_ptr)

    all_meta = [None] * world_size
    dist.all_gather_object(all_meta, meta_handle, group=gloo_group)
    meta_ptrs = []
    _ipc_refs = []
    for i in range(world_size):
        if i == rank:
            meta_ptrs.append(meta_ptr.value)
        else:
            p = ipc.ipc_open_handle(all_meta[i])
            _ipc_refs.append(p)
            meta_ptrs.append(p.value)

    def alloc_shared():
        ptr = ipc.malloc(max_buf)
        handle = ipc.ipc_get_handle(ptr)
        all_h = [None] * world_size
        dist.all_gather_object(all_h, handle, group=gloo_group)
        ptrs = []
        for i in range(world_size):
            if i == rank:
                ptrs.append(ptr.value)
            else:
                p = ipc.ipc_open_handle(all_h[i])
                _ipc_refs.append(p)
                ptrs.append(p.value)
        return ptrs

    buf0 = alloc_shared()
    buf1 = alloc_shared()

    rank_data = torch.empty(max_buf, dtype=torch.uint8, device=f"cuda:{rank}")
    fa = pcie_allreduce.init_custom_ar(meta_ptrs, rank_data, rank)
    pcie_allreduce.register_pcie_buffers(fa, buf0, buf1)
    torch.cuda.synchronize()
    dist.barrier()

    dev = f"cuda:{rank}"

    # ---- Size sweep: expressed as (numel, dtype) pairs ----
    # We sweep bf16 since that's the decode allreduce dtype.
    sizes_bytes = []
    # Powers of 2 from 1KB to 2MB.
    b = 1024
    while b <= 2 * 1024 * 1024:
        sizes_bytes.append(b)
        b *= 2
    # Also add key Kimi/DeepSeek batch sizes.
    for bs in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        sz = bs * HIDDEN * 2  # bf16
        if sz <= max_buf:
            sizes_bytes.append(sz)
    sizes_bytes = sorted(set(sizes_bytes))

    def bench_custom(numel):
        inp = torch.ones(numel, dtype=torch.bfloat16, device=dev)
        out = torch.zeros_like(inp)
        for _ in range(WARMUP):
            pcie_allreduce.all_reduce(fa, inp, out, 0, 0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            pcie_allreduce.all_reduce(fa, inp, out, 0, 0)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ITERS

    def bench_nccl(numel):
        inp = torch.ones(numel, dtype=torch.bfloat16, device=dev)
        for _ in range(WARMUP):
            dist.all_reduce(inp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(inp)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / ITERS

    def bench_custom_graph(numel):
        """Capture a single custom AR in a CUDA graph and replay it."""
        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            g_inp = torch.ones(numel, dtype=torch.bfloat16, device=dev)
            g_out = torch.zeros_like(g_inp)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            pcie_allreduce.all_reduce(fa, g_inp, g_out, 0, 0)

        handle, off = pcie_allreduce.get_graph_buffer_ipc_meta(fa)
        all_graph_meta = [None] * world_size
        dist.all_gather_object(all_graph_meta, (handle, off), group=gloo_group)
        pcie_allreduce.register_graph_buffers(
            fa, [d[0] for d in all_graph_meta], [d[1] for d in all_graph_meta]
        )
        dist.barrier()

        with torch.cuda.stream(s):
            for _ in range(WARMUP):
                g.replay()
        s.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(ITERS):
                g.replay()
        s.synchronize()
        return (time.perf_counter() - t0) / ITERS

    def bench_nccl_graph(numel):
        """Capture a single NCCL all_reduce in a CUDA graph and replay it."""
        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            g_inp = torch.ones(numel, dtype=torch.bfloat16, device=dev)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            dist.all_reduce(g_inp)

        with torch.cuda.stream(s):
            for _ in range(WARMUP):
                g.replay()
        s.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(ITERS):
                g.replay()
        s.synchronize()
        return (time.perf_counter() - t0) / ITERS

    # ---- Eager benchmark ----
    if rank == 0:
        print(f"\n=== Eager (double-buffer) ===")
        print(f"{'Size':>10s}  {'bs@7168':>7s}  {'Custom':>10s}  {'NCCL':>10s}  {'Speedup':>8s}  Winner")
        print("-" * 65)

    for sz in sizes_bytes:
        numel = sz // 2  # bf16
        numel = (numel // 8) * 8
        if numel == 0:
            continue

        t_custom = bench_custom(numel)
        t_nccl = bench_nccl(numel)
        dist.barrier()

        if rank == 0:
            bs_equiv = sz / (HIDDEN * 2)
            speedup = t_nccl / t_custom
            winner = "custom" if speedup > 1.0 else "NCCL"
            flag = " <-- crossover" if 0.9 <= speedup <= 1.1 else ""
            print(
                f"{sz:>10,d}  {bs_equiv:>7.1f}  {t_custom*1e6:>8.1f}us  {t_nccl*1e6:>8.1f}us  {speedup:>7.2f}x  {winner}{flag}"
            )

    # ---- CUDA graph benchmark ----
    if rank == 0:
        print(f"\n=== CUDA Graph (replay) ===")
        print(f"{'Size':>10s}  {'bs@7168':>7s}  {'Custom':>10s}  {'NCCL':>10s}  {'Speedup':>8s}  Winner")
        print("-" * 65)

    for sz in sizes_bytes:
        numel = sz // 2
        numel = (numel // 8) * 8
        if numel == 0:
            continue

        t_custom = bench_custom_graph(numel)
        t_nccl = bench_nccl_graph(numel)
        dist.barrier()

        if rank == 0:
            bs_equiv = sz / (HIDDEN * 2)
            speedup = t_nccl / t_custom
            winner = "custom" if speedup > 1.0 else "NCCL"
            flag = " <-- crossover" if 0.9 <= speedup <= 1.1 else ""
            print(
                f"{sz:>10,d}  {bs_equiv:>7.1f}  {t_custom*1e6:>8.1f}us  {t_nccl*1e6:>8.1f}us  {speedup:>7.2f}x  {winner}{flag}"
            )

    pcie_allreduce.dispose(fa)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    ws = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"PCIe allreduce vs NCCL crossover benchmark: {ws} GPUs")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    mp.spawn(worker, args=(ws,), nprocs=ws, join=True)
