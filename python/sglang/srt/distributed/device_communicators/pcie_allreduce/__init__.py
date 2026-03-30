"""PCIe-only custom allreduce with system-scope barriers.

JIT-compiles a CUDA extension on first import. Exports the same op
interface as sgl_kernel.allreduce so it can be used as a drop-in for
the PCIe path.
"""

import os

import torch
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_ext = load(
    name="pcie_allreduce_ext",
    sources=[os.path.join(_dir, "pcie_allreduce.cu")],
    extra_cuda_cflags=["-O2", "--expt-relaxed-constexpr"],
    extra_ldflags=["-lcuda"],
    verbose=True,
)

init_custom_ar = _ext.init_custom_ar
all_reduce = _ext.all_reduce
dispose = _ext.dispose
meta_size = _ext.meta_size
register_buffer = _ext.register_buffer
register_pcie_buffers = _ext.register_pcie_buffers
get_graph_buffer_ipc_meta = _ext.get_graph_buffer_ipc_meta
register_graph_buffers = _ext.register_graph_buffers
allreduce_rmsnorm = _ext.allreduce_rmsnorm
allreduce_gemma_rmsnorm = _ext.allreduce_gemma_rmsnorm
