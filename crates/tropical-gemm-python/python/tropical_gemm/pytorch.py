"""
PyTorch integration for tropical matrix multiplication.

This module provides custom autograd functions that enable using tropical
matrix multiplication in PyTorch neural networks with full gradient support.

Tropical semirings are useful for:
- Shortest path problems (MinPlus)
- Longest path problems (MaxPlus)
- Dynamic programming on graphs
- Probabilistic inference (log-space operations)

Example:
    >>> import torch
    >>> from tropical_gemm.pytorch import tropical_maxplus_matmul
    >>>
    >>> a = torch.randn(100, 50, requires_grad=True)
    >>> b = torch.randn(50, 80, requires_grad=True)
    >>> c = tropical_maxplus_matmul(a, b)
    >>> loss = c.sum()
    >>> loss.backward()
"""

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is required for this module. "
        "Install it with: pip install tropical-gemm[torch]"
    )

import tropical_gemm

# Check if GPU is available
GPU_AVAILABLE = tropical_gemm.cuda_available()


class TropicalMaxPlusMatmul(torch.autograd.Function):
    """
    Custom autograd function for MaxPlus tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] + B[k,j])

    The gradient is sparse: only the argmax index contributes to each output.
    For each output C[i,j], the gradient flows back to:
    - A[i, argmax[i,j]]
    - B[argmax[i,j], j]
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute tropical MaxPlus matmul.

        Args:
            a: Input tensor of shape (M, K)
            b: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        m, k = a.shape
        n = b.shape[1]

        # Convert to contiguous numpy arrays
        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        # Ensure contiguous layout
        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        # Call the optimized Rust implementation (returns flattened arrays)
        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_with_argmax(a_np, b_np)

        # Reshape to 2D
        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        # Save argmax for backward pass
        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        """
        Backward pass: compute gradients w.r.t. A and B.

        The tropical matmul gradient is sparse because only the argmax
        index contributes to each output element.
        """
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        # Ensure contiguous numpy arrays
        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # Compute gradients using the Rust backend (returns flattened arrays)
        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        # Reshape to 2D
        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMinPlusMatmul(torch.autograd.Function):
    """
    Custom autograd function for MinPlus tropical matrix multiplication.

    Forward: C[i,j] = min_k(A[i,k] + B[k,j])

    Useful for shortest path computations in graphs.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMaxMulMatmul(torch.autograd.Function):
    """
    Custom autograd function for MaxMul tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] * B[k,j])

    The backward pass differs from MaxPlus/MinPlus because the operation
    is multiplication, not addition. The chain rule gives:
    - grad_A[i,k] = grad_C[i,j] * B[k,j] if k == argmax[i,j]
    - grad_B[k,j] = grad_C[i,j] * A[i,k] if k == argmax[i,j]

    Useful for max-probability computations (non-log space).
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        # Save inputs and argmax for backward (needed for multiplicative gradient)
        ctx.save_for_backward(
            torch.from_numpy(a_np),
            torch.from_numpy(b_np),
            torch.from_numpy(argmax_np),
        )
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        a, b, argmax = ctx.saved_tensors
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)
        a_np = a.numpy().astype(np.float32)
        b_np = b.numpy().astype(np.float32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # Use multiplicative backward rule
        grad_a_flat = tropical_gemm.maxmul_backward_a(grad_c_np, argmax_np, b_np)
        grad_b_flat = tropical_gemm.maxmul_backward_b(grad_c_np, argmax_np, a_np)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k_dim)).to(
            grad_c.device
        )
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k_dim, n)).to(
            grad_c.device
        )

        return grad_a, grad_b


# ===========================================================================
# GPU-Accelerated Autograd Functions (requires CUDA feature)
# ===========================================================================


class TropicalMaxPlusMatmulGPU(torch.autograd.Function):
    """
    GPU-accelerated MaxPlus tropical matrix multiplication.

    Uses CUDA kernels for the forward pass. The backward pass uses CPU
    since the argmax indices are already computed.

    Note: Creates a new CUDA context for each call. For repeated operations,
    consider batching computations.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "CUDA GPU support not available. Rebuild with --features cuda"
            )

        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        # Use GPU implementation
        c_flat, argmax_flat = tropical_gemm.maxplus_matmul_gpu_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # CPU backward (GPU backward not yet exposed to Python)
        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMinPlusMatmulGPU(torch.autograd.Function):
    """
    GPU-accelerated MinPlus tropical matrix multiplication.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "CUDA GPU support not available. Rebuild with --features cuda"
            )

        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        c_flat, argmax_flat = tropical_gemm.minplus_matmul_gpu_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        ctx.save_for_backward(torch.from_numpy(argmax_np))
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        (argmax,) = ctx.saved_tensors
        k = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        grad_a_flat = tropical_gemm.backward_a(grad_c_np, argmax_np, k)
        grad_b_flat = tropical_gemm.backward_b(grad_c_np, argmax_np, k)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k)).to(grad_c.device)
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k, n)).to(grad_c.device)

        return grad_a, grad_b


class TropicalMaxMulMatmulGPU(torch.autograd.Function):
    """
    GPU-accelerated MaxMul tropical matrix multiplication.

    Forward: C[i,j] = max_k(A[i,k] * B[k,j])

    Uses CUDA kernels for the forward pass. The backward pass uses CPU
    with the multiplicative gradient rule.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "CUDA GPU support not available. Rebuild with --features cuda"
            )

        m, k = a.shape
        n = b.shape[1]

        a_np = a.detach().cpu().numpy().astype(np.float32)
        b_np = b.detach().cpu().numpy().astype(np.float32)

        if not a_np.flags["C_CONTIGUOUS"]:
            a_np = np.ascontiguousarray(a_np)
        if not b_np.flags["C_CONTIGUOUS"]:
            b_np = np.ascontiguousarray(b_np)

        # Use GPU implementation
        c_flat, argmax_flat = tropical_gemm.maxmul_matmul_gpu_with_argmax(a_np, b_np)

        c_np = np.array(c_flat).reshape(m, n)
        argmax_np = np.array(argmax_flat).reshape(m, n)

        # Save inputs and argmax for backward (needed for multiplicative gradient)
        ctx.save_for_backward(
            torch.from_numpy(a_np),
            torch.from_numpy(b_np),
            torch.from_numpy(argmax_np),
        )
        ctx.k = k
        ctx.m = m
        ctx.n = n

        return torch.from_numpy(c_np).to(a.device)

    @staticmethod
    def backward(ctx, grad_c: torch.Tensor):
        a, b, argmax = ctx.saved_tensors
        k_dim = ctx.k
        m = ctx.m
        n = ctx.n

        grad_c_np = grad_c.cpu().numpy().astype(np.float32)
        argmax_np = argmax.numpy().astype(np.int32)
        a_np = a.numpy().astype(np.float32)
        b_np = b.numpy().astype(np.float32)

        if not grad_c_np.flags["C_CONTIGUOUS"]:
            grad_c_np = np.ascontiguousarray(grad_c_np)
        if not argmax_np.flags["C_CONTIGUOUS"]:
            argmax_np = np.ascontiguousarray(argmax_np)

        # Use multiplicative backward rule
        grad_a_flat = tropical_gemm.maxmul_backward_a(grad_c_np, argmax_np, b_np)
        grad_b_flat = tropical_gemm.maxmul_backward_b(grad_c_np, argmax_np, a_np)

        grad_a = torch.from_numpy(np.array(grad_a_flat).reshape(m, k_dim)).to(
            grad_c.device
        )
        grad_b = torch.from_numpy(np.array(grad_b_flat).reshape(k_dim, n)).to(
            grad_c.device
        )

        return grad_a, grad_b


# ===========================================================================
# Convenience functions
# ===========================================================================


def tropical_maxplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MaxPlus tropical matrix multiplication: C[i,j] = max_k(A[i,k] + B[k,j])

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return TropicalMaxPlusMatmul.apply(a, b)


def tropical_minplus_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MinPlus tropical matrix multiplication: C[i,j] = min_k(A[i,k] + B[k,j])

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return TropicalMinPlusMatmul.apply(a, b)


def tropical_maxmul_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    MaxMul tropical matrix multiplication: C[i,j] = max_k(A[i,k] * B[k,j])

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)
    """
    return TropicalMaxMulMatmul.apply(a, b)


def tropical_maxplus_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated MaxPlus tropical matrix multiplication.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)

    Raises:
        RuntimeError: If CUDA support is not available
    """
    return TropicalMaxPlusMatmulGPU.apply(a, b)


def tropical_minplus_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated MinPlus tropical matrix multiplication.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)

    Raises:
        RuntimeError: If CUDA support is not available
    """
    return TropicalMinPlusMatmulGPU.apply(a, b)


def tropical_maxmul_matmul_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated MaxMul tropical matrix multiplication.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)

    Returns:
        Output tensor of shape (M, N)

    Raises:
        RuntimeError: If CUDA support is not available
    """
    return TropicalMaxMulMatmulGPU.apply(a, b)


__all__ = [
    # CPU autograd functions
    "TropicalMaxPlusMatmul",
    "TropicalMinPlusMatmul",
    "TropicalMaxMulMatmul",
    # GPU autograd functions
    "TropicalMaxPlusMatmulGPU",
    "TropicalMinPlusMatmulGPU",
    "TropicalMaxMulMatmulGPU",
    # Convenience functions
    "tropical_maxplus_matmul",
    "tropical_minplus_matmul",
    "tropical_maxmul_matmul",
    "tropical_maxplus_matmul_gpu",
    "tropical_minplus_matmul_gpu",
    "tropical_maxmul_matmul_gpu",
    # GPU availability flag
    "GPU_AVAILABLE",
]
