"""Minimal ctypes wrapper for CUDA IPC operations.

Self-contained — no sglang imports. Used by the pcie_allreduce tests
and can be used standalone for any CUDA IPC buffer management.
"""

import ctypes
from typing import Optional


cudaError_t = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _find_cudart() -> str:
    with open("/proc/self/maps") as f:
        for line in f:
            if "libcudart" in line:
                return line[line.index("/") :].strip()
    raise RuntimeError("libcudart not found in /proc/self/maps (is torch loaded?)")


class CudaIPC:
    """Thin ctypes interface to the cudaRT functions needed for IPC."""

    def __init__(self, so_file: Optional[str] = None):
        import torch  # noqa: F401 — ensures libcudart is loaded.

        if so_file is None:
            so_file = _find_cudart()
        self._lib = ctypes.CDLL(so_file)

        self._lib.cudaMalloc.restype = cudaError_t
        self._lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self._lib.cudaFree.restype = cudaError_t
        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaMemset.restype = cudaError_t
        self._lib.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self._lib.cudaIpcGetMemHandle.restype = cudaError_t
        self._lib.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(cudaIpcMemHandle_t),
            ctypes.c_void_p,
        ]
        self._lib.cudaIpcOpenMemHandle.restype = cudaError_t
        self._lib.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            cudaIpcMemHandle_t,
            ctypes.c_uint,
        ]
        self._lib.cudaGetErrorString.restype = ctypes.c_char_p
        self._lib.cudaGetErrorString.argtypes = [cudaError_t]

    def _check(self, err: int) -> None:
        if err != 0:
            msg = self._lib.cudaGetErrorString(err)
            raise RuntimeError(f"CUDA error: {msg.decode()}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), size))
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        self._check(self._lib.cudaFree(ptr))

    def memset(self, ptr: ctypes.c_void_p, value: int, count: int) -> None:
        self._check(self._lib.cudaMemset(ptr, value, count))

    def ipc_get_handle(self, ptr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self._check(self._lib.cudaIpcGetMemHandle(ctypes.byref(handle), ptr))
        return handle

    def ipc_open_handle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        ptr = ctypes.c_void_p()
        self._check(
            self._lib.cudaIpcOpenMemHandle(
                ctypes.byref(ptr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return ptr
