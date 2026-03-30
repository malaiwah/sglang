"""Comprehensive stress test for PCIe allreduce."""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cuda_ipc import CudaIPC


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29510"
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    import pcie_allreduce

    ipc = CudaIPC()
    max_size = 56 * 1024
    meta_sz = pcie_allreduce.meta_size()

    meta_ptr = ipc.malloc(meta_sz + max_size)
    ipc.memset(meta_ptr, 0, meta_sz + max_size)

    meta_handle = ipc.ipc_get_handle(meta_ptr)
    all_meta = [None] * world_size
    dist.all_gather_object(all_meta, meta_handle)
    meta_ptrs = []
    _ipc_refs = []
    for i in range(world_size):
        if i == rank:
            meta_ptrs.append(meta_ptr.value)
        else:
            p = ipc.ipc_open_handle(all_meta[i])
            _ipc_refs.append(p)
            meta_ptrs.append(p.value)

    def alloc_shared_buffer():
        ptr = ipc.malloc(max_size)
        handle = ipc.ipc_get_handle(ptr)
        all_handles = [None] * world_size
        dist.all_gather_object(all_handles, handle)
        ptrs = []
        for i in range(world_size):
            if i == rank:
                ptrs.append(ptr.value)
            else:
                p = ipc.ipc_open_handle(all_handles[i])
                _ipc_refs.append(p)
                ptrs.append(p.value)
        return ptrs

    buf_ptrs_0 = alloc_shared_buffer()
    buf_ptrs_1 = alloc_shared_buffer()

    rank_data = torch.empty(max_size, dtype=torch.uint8, device=f"cuda:{rank}")
    fa = pcie_allreduce.init_custom_ar(meta_ptrs, rank_data, rank)
    pcie_allreduce.register_pcie_buffers(fa, buf_ptrs_0, buf_ptrs_1)
    torch.cuda.synchronize()

    dev = f"cuda:{rank}"
    ws = world_size
    all_pass = True

    def ar(inp, out):
        pcie_allreduce.all_reduce(fa, inp, out, 0, 0)

    def run(name, fn):
        nonlocal all_pass
        dist.barrier()
        torch.cuda.synchronize()
        try:
            ok = fn()
        except RuntimeError as e:
            ok = False
            if rank == 0:
                print(f"  {name}: EXCEPTION {e}")
        all_pass = all_pass and ok

    # 1. Rapid fire 10K.
    def t1():
        N = 10000
        inp = torch.full((1, 7168), float(rank + 1), dtype=torch.bfloat16, device=dev)
        out = torch.zeros_like(inp)
        t0 = time.perf_counter()
        for _ in range(N):
            ar(inp, out)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        err = (out.float() - ws * (ws + 1) / 2.0).abs().max().item()
        ok = err < 0.5
        if rank == 0:
            print(f"  rapid_fire (10K): {el*1000:.1f}ms ({el/N*1e6:.1f}us/iter) err={err:.4f} {'PASS' if ok else 'FAIL'}")
        return ok

    # 2. Varying sizes.
    def t2():
        sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 7168, 8192, 14336]
        ok_all = True
        for n in sizes:
            inp = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device=dev)
            out = torch.zeros_like(inp)
            ar(inp, out)
            torch.cuda.synchronize()
            err = (out.float() - ws * (ws + 1) / 2.0).abs().max().item()
            if err > 0.5:
                ok_all = False
                if rank == 0:
                    print(f"    FAIL numel={n} err={err}")
        if rank == 0:
            print(f"  varying_sizes ({len(sizes)}): {'PASS' if ok_all else 'FAIL'}")
        return ok_all

    # 3. Unique per-element values (fp32).
    def t3():
        numel = 7168
        ok_all = True
        for i in range(200):
            base = (i * ws + rank + 1) * 0.01
            inp = torch.arange(numel, dtype=torch.float32, device=dev) * 0.001 + base
            out = torch.zeros_like(inp)
            ar(inp, out)
            torch.cuda.synchronize()
            exp = sum(
                torch.arange(numel, dtype=torch.float32, device=dev) * 0.001 + (i * ws + r + 1) * 0.01
                for r in range(ws)
            )
            err = (out - exp).abs().max().item()
            if err > 0.1:
                ok_all = False
                if rank == 0:
                    print(f"    FAIL iter {i} err={err}")
                break
        if rank == 0:
            print(f"  unique_values (200x fp32): {'PASS' if ok_all else 'FAIL'}")
        return ok_all

    # 4. Back-to-back 5K.
    def t4():
        N = 5000
        for i in range(N):
            v = float(rank + 1) + i * 0.001
            inp = torch.full((1, 7168), v, dtype=torch.bfloat16, device=dev)
            out = torch.zeros_like(inp)
            ar(inp, out)
        torch.cuda.synchronize()
        last = sum(float(r + 1) + (N - 1) * 0.001 for r in range(ws))
        err = (out.float() - last).abs().max().item()
        ok = err < 1.0
        if rank == 0:
            print(f"  back_to_back (5K): err={err:.4f} {'PASS' if ok else 'FAIL'}")
        return ok

    # 5. Alternating tiny/large.
    def t5():
        for i in range(1000):
            n = 16 if i % 2 == 0 else 7168
            inp = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device=dev)
            out = torch.zeros_like(inp)
            ar(inp, out)
        torch.cuda.synchronize()
        err = (out.float() - ws * (ws + 1) / 2.0).abs().max().item()
        ok = err < 0.5
        if rank == 0:
            print(f"  alternating (1K): err={err:.4f} {'PASS' if ok else 'FAIL'}")
        return ok

    # 6. All dtypes x 100.
    def t6():
        ok_all = True
        for dt, nm in [(torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32")]:
            inp = torch.full((1, 1024), float(rank + 1), dtype=dt, device=dev)
            out = torch.zeros_like(inp)
            for _ in range(100):
                ar(inp, out)
            torch.cuda.synchronize()
            err = (out.float() - ws * (ws + 1) / 2.0).abs().max().item()
            ok = err < 0.5
            ok_all = ok_all and ok
            if rank == 0 and not ok:
                print(f"    FAIL {nm} err={err}")
        if rank == 0:
            print(f"  dtypes (100x each): {'PASS' if ok_all else 'FAIL'}")
        return ok_all

    # 7. Kimi-K2.5 / DeepseekV3 decode step: 61 layers x 2 ARs = 122 per step.
    HIDDEN = 7168
    NUM_LAYERS = 61

    def t7():
        steps = 50
        expected = float(ws * (ws + 1) // 2)
        inp = torch.full((1, HIDDEN), float(rank + 1), dtype=torch.bfloat16, device=dev)
        out = torch.zeros_like(inp)
        scratch = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        ok = True
        t0 = time.perf_counter()
        for step in range(steps):
            for layer in range(NUM_LAYERS):
                inp.fill_(float(rank + 1))
                ar(inp, out)
                scratch.copy_(out)
                scratch.mul_(1.0 / expected)
                scratch.mul_(float(rank + 1))
                ar(scratch, out)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        total_ar = steps * NUM_LAYERS * 2
        err = (out.float() - expected).abs().max().item()
        ok = err < 1.0
        if rank == 0:
            us = el / total_ar * 1e6
            ms_step = el / steps * 1e3
            print(f"  decode_step (50 steps x 122 ARs): {el*1000:.0f}ms total, "
                  f"{ms_step:.1f}ms/step, {us:.1f}us/AR, err={err:.4f} {'PASS' if ok else 'FAIL'}")
        return ok

    # 8. Continuous batching: batch size varies between forward passes.
    def t8():
        ok_all = True
        batch_sizes = [bs for bs in [1, 2, 3, 4] if bs * HIDDEN * 2 <= max_size]
        for bs in batch_sizes:
            inp = torch.full((bs, HIDDEN), float(rank + 1), dtype=torch.bfloat16, device=dev)
            out = torch.zeros_like(inp)
            for _ in range(NUM_LAYERS * 2):
                ar(inp, out)
            torch.cuda.synchronize()
            err = (out.float() - ws * (ws + 1) / 2.0).abs().max().item()
            ok = err < 0.5
            ok_all = ok_all and ok
            if rank == 0:
                tag = "PASS" if ok else "FAIL"
                print(f"    bs={bs} ({bs*HIDDEN*2}B): err={err:.4f} {tag}")
        for step in range(20):
            bs = batch_sizes[step % len(batch_sizes)]
            inp = torch.full((bs, HIDDEN), float(rank + 1), dtype=torch.bfloat16, device=dev)
            out = torch.zeros_like(inp)
            for _ in range(NUM_LAYERS * 2):
                ar(inp, out)
            torch.cuda.synchronize()
            err = (out.float() - ws * (ws + 1) / 2.0).abs().max().item()
            if err > 0.5:
                ok_all = False
                if rank == 0:
                    print(f"    FAIL interleave step={step} bs={bs} err={err:.4f}")
                break
        if rank == 0:
            print(f"  continuous_batching: {'PASS' if ok_all else 'FAIL'}")
        return ok_all

    # 9. Rapid decode generation: 200 tokens, each token = 122 ARs.
    def t9():
        tokens = 200
        expected = float(ws * (ws + 1) // 2)
        inp = torch.empty(1, HIDDEN, dtype=torch.bfloat16, device=dev)
        out = torch.zeros_like(inp)
        t0 = time.perf_counter()
        for tok in range(tokens):
            for _ in range(NUM_LAYERS * 2):
                inp.fill_(float(rank + 1))
                ar(inp, out)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        total_ar = tokens * NUM_LAYERS * 2
        err = (out.float() - expected).abs().max().item()
        ok = err < 1.0
        if rank == 0:
            print(f"  sustained_decode (200 tok, {total_ar} ARs): "
                  f"{el*1000:.0f}ms, {el/tokens*1e3:.1f}ms/tok, "
                  f"{el/total_ar*1e6:.1f}us/AR, err={err:.4f} {'PASS' if ok else 'FAIL'}")
        return ok

    # 10. Real CUDA graph: capture N allreduces, replay M times.
    def t10():
        N_AR = 10
        N_REPLAY = 500

        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            graph_inp = torch.full((1, HIDDEN), float(rank + 1), dtype=torch.bfloat16, device=dev)
            graph_out = torch.zeros_like(graph_inp)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(N_AR):
                pcie_allreduce.all_reduce(fa, graph_inp, graph_out, 0, 0)

        handle, off = pcie_allreduce.get_graph_buffer_ipc_meta(fa)
        all_meta = [None] * world_size
        dist.all_gather_object(all_meta, (handle, off))
        pcie_allreduce.register_graph_buffers(
            fa, [d[0] for d in all_meta], [d[1] for d in all_meta]
        )
        dist.barrier()

        expected = float(ws * (ws + 1) // 2)
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(N_REPLAY):
                graph_inp.fill_(float(rank + 1))
                g.replay()
        s.synchronize()
        el = time.perf_counter() - t0

        err = (graph_out.float() - expected).abs().max().item()
        ok = err < 0.5
        total_ar = N_REPLAY * N_AR
        if rank == 0:
            print(f"  cuda_graph ({N_REPLAY} replays x {N_AR} ARs): "
                  f"{el*1000:.0f}ms, {el/total_ar*1e6:.1f}us/AR, err={err:.4f} "
                  f"{'PASS' if ok else 'FAIL'}")
        return ok

    # 11. CUDA graph with interleaved compute (realistic decode step).
    def t11():
        N_REPLAY = 200
        expected = float(ws * (ws + 1) // 2)

        s = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s):
            g_inp = torch.full((1, HIDDEN), float(rank + 1), dtype=torch.bfloat16, device=dev)
            g_out = torch.zeros_like(g_inp)
            g_scratch = torch.empty_like(g_inp)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(NUM_LAYERS):
                pcie_allreduce.all_reduce(fa, g_inp, g_out, 0, 0)
                g_scratch.copy_(g_out)
                g_scratch.mul_(1.0 / expected)
                g_scratch.mul_(float(rank + 1))
                pcie_allreduce.all_reduce(fa, g_scratch, g_out, 0, 0)

        handle, off = pcie_allreduce.get_graph_buffer_ipc_meta(fa)
        all_meta = [None] * world_size
        dist.all_gather_object(all_meta, (handle, off))
        pcie_allreduce.register_graph_buffers(
            fa, [d[0] for d in all_meta], [d[1] for d in all_meta]
        )
        dist.barrier()

        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(N_REPLAY):
                g_inp.fill_(float(rank + 1))
                g.replay()
        s.synchronize()
        el = time.perf_counter() - t0

        err = (g_out.float() - expected).abs().max().item()
        ok = err < 1.0
        total_ar = N_REPLAY * NUM_LAYERS * 2
        if rank == 0:
            print(f"  cuda_graph_decode ({N_REPLAY} replays x 122 ARs): "
                  f"{el*1000:.0f}ms, {el/N_REPLAY*1e3:.1f}ms/step, "
                  f"{el/total_ar*1e6:.1f}us/AR, err={err:.4f} "
                  f"{'PASS' if ok else 'FAIL'}")
        return ok

    run("rapid_fire", t1)
    run("varying_sizes", t2)
    run("unique_values", t3)
    run("back_to_back", t4)
    run("alternating", t5)
    run("dtypes", t6)
    run("decode_step", t7)
    run("continuous_batching", t8)
    run("sustained_decode", t9)
    run("cuda_graph", t10)
    run("cuda_graph_decode", t11)

    pcie_allreduce.dispose(fa)
    dist.barrier()
    if rank == 0:
        print(f"\n{'ALL STRESS TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    dist.destroy_process_group()


if __name__ == "__main__":
    ws = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"Stress test: {ws} GPUs, PCIe sys-scope barriers\n")
    mp.spawn(worker, args=(ws,), nprocs=ws, join=True)
