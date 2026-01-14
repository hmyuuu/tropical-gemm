#!/usr/bin/env python3
"""
PyTorch Custom Autograd Function for Tropical Matrix Multiplication.

This example demonstrates how to integrate tropical-gemm with PyTorch's
automatic differentiation system using custom autograd functions.

Tropical semirings are useful for:
- Shortest path problems (MinPlus)
- Longest path problems (MaxPlus)
- Dynamic programming on graphs
- Probabilistic inference (log-space operations)

Features:
- CPU implementation using optimized SIMD kernels
- GPU implementation using CUDA (when available)

Usage:
    pip install torch numpy
    cd crates/tropical-gemm-python
    maturin develop --features cuda  # For GPU support
    python examples/pytorch_tropical.py
"""

import torch
import numpy as np

# Import the tropical_gemm module and PyTorch integration
import tropical_gemm
from tropical_gemm.pytorch import (
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    tropical_maxplus_matmul_gpu,
    tropical_minplus_matmul_gpu,
    GPU_AVAILABLE,
)


def verify_gradients():
    """Verify gradients using manual check."""
    print("=" * 60)
    print("Gradient Verification")
    print("=" * 60)

    print("\nManual gradient check:")

    a = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    b = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

    # Forward pass
    c = tropical_maxplus_matmul(a, b)
    loss = c.sum()
    loss.backward()

    print(f"  Input A shape: {a.shape}")
    print(f"  Input B shape: {b.shape}")
    print(f"  Output C shape: {c.shape}")
    print(f"  grad_A shape: {a.grad.shape}")
    print(f"  grad_B shape: {b.grad.shape}")
    print(f"  grad_A sum: {a.grad.sum().item():.4f}")
    print(f"  grad_B sum: {b.grad.sum().item():.4f}")

    # The sum of gradients should equal the number of elements in C
    # because each C[i,j] contributes exactly 1 to both grad_A and grad_B
    expected_sum = c.numel()
    actual_sum_a = a.grad.sum().item()
    actual_sum_b = b.grad.sum().item()

    print(f"\n  Expected grad sum (num elements in C): {expected_sum}")
    print(f"  Actual grad_A sum: {actual_sum_a:.1f}")
    print(f"  Actual grad_B sum: {actual_sum_b:.1f}")

    if abs(actual_sum_a - expected_sum) < 0.1 and abs(actual_sum_b - expected_sum) < 0.1:
        print("  Gradient sum check passed!")
    else:
        print("  Gradient sum check failed!")


def demo_shortest_path():
    """
    Demonstrate using tropical MinPlus for shortest path computation.

    In graph algorithms, the adjacency matrix A contains edge weights,
    and A^n (tropical power) gives shortest paths of length n.
    """
    print("\n" + "=" * 60)
    print("Shortest Path Example (MinPlus)")
    print("=" * 60)

    # Adjacency matrix for a simple graph (inf = no edge)
    inf = float("inf")
    adj = torch.tensor(
        [
            [0.0, 1.0, inf, 4.0],
            [inf, 0.0, 2.0, inf],
            [inf, inf, 0.0, 1.0],
            [inf, inf, inf, 0.0],
        ],
        dtype=torch.float32,
    )

    print("\nAdjacency matrix (edge weights, inf = no edge):")
    print(adj.numpy())

    # Compute 2-hop shortest paths
    two_hop = tropical_minplus_matmul(adj, adj)
    print("\n2-hop shortest paths:")
    print(two_hop.numpy())

    # Compute 3-hop shortest paths
    three_hop = tropical_minplus_matmul(two_hop, adj)
    print("\n3-hop shortest paths:")
    print(three_hop.numpy())


def demo_longest_path():
    """
    Demonstrate using tropical MaxPlus for longest path computation.

    Useful in critical path analysis (project scheduling).
    """
    print("\n" + "=" * 60)
    print("Longest Path Example (MaxPlus)")
    print("=" * 60)

    # Edge weights for a DAG (task durations)
    neg_inf = float("-inf")
    adj = torch.tensor(
        [
            [0.0, 3.0, 2.0, neg_inf],
            [neg_inf, 0.0, neg_inf, 4.0],
            [neg_inf, neg_inf, 0.0, 5.0],
            [neg_inf, neg_inf, neg_inf, 0.0],
        ],
        dtype=torch.float32,
    )

    print("\nAdjacency matrix (edge weights, -inf = no edge):")
    print(adj.numpy())

    # Compute 2-hop longest paths
    two_hop = tropical_maxplus_matmul(adj, adj)
    print("\n2-hop longest paths:")
    print(two_hop.numpy())


def demo_optimization():
    """
    Demonstrate using tropical matmul in an optimization loop.

    This shows that gradients flow correctly through the tropical operations.
    """
    print("\n" + "=" * 60)
    print("Optimization Example")
    print("=" * 60)

    torch.manual_seed(42)

    # Create learnable parameters
    a = torch.randn(4, 5, requires_grad=True)
    b = torch.randn(5, 3, requires_grad=True)

    # Target output
    target = torch.randn(4, 3)

    optimizer = torch.optim.Adam([a, b], lr=0.1)

    print("\nOptimizing to match target using tropical MaxPlus matmul:")
    for step in range(5):
        optimizer.zero_grad()

        # Forward pass through tropical matmul
        c = tropical_maxplus_matmul(a, b)

        # MSE loss
        loss = ((c - target) ** 2).mean()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        print(f"  Step {step + 1}: loss = {loss.item():.6f}")


def benchmark():
    """Performance comparison between CPU and GPU."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    import time

    sizes = [64, 128, 256, 512, 1024]

    print(f"\nGPU Available: {GPU_AVAILABLE}")
    print("\nCPU Performance:")

    for n in sizes:
        a = torch.randn(n, n, dtype=torch.float32)
        b = torch.randn(n, n, dtype=torch.float32)

        # Warm up
        _ = tropical_maxplus_matmul(a, b)

        # Benchmark
        start = time.perf_counter()
        iterations = 10
        for _ in range(iterations):
            c = tropical_maxplus_matmul(a, b)
        elapsed = (time.perf_counter() - start) / iterations * 1000

        print(f"  {n}x{n}: {elapsed:.3f} ms per matmul")

    if GPU_AVAILABLE:
        print("\nGPU Performance:")
        for n in sizes:
            a = torch.randn(n, n, dtype=torch.float32)
            b = torch.randn(n, n, dtype=torch.float32)

            # Warm up (includes CUDA context + kernel compilation)
            _ = tropical_maxplus_matmul_gpu(a, b)

            # Benchmark
            start = time.perf_counter()
            iterations = 10
            for _ in range(iterations):
                c = tropical_maxplus_matmul_gpu(a, b)
            elapsed = (time.perf_counter() - start) / iterations * 1000

            print(f"  {n}x{n}: {elapsed:.3f} ms per matmul")

        print("\nNote: GPU timings include CUDA context initialization and kernel")
        print("compilation (~7-8 seconds) for each call. In production, use batched")
        print("operations or reuse contexts for much better performance.")


if __name__ == "__main__":
    print("Tropical GEMM PyTorch Integration Demo")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")

    verify_gradients()
    demo_shortest_path()
    demo_longest_path()
    demo_optimization()
    benchmark()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
