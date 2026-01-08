# tropical-gemm

[![CI](https://github.com/TensorBFS/tropical-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorBFS/tropical-gemm/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/TensorBFS/tropical-gemm/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/tropical-gemm)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tensorbfs.github.io/tropical-gemm/tropical_gemm/)

High-performance tropical matrix multiplication in Rust with SIMD and CUDA backends.

## Features

- **Multiple Semirings**: MaxPlus, MinPlus, MaxMul, AndOr, Counting
- **SIMD Acceleration**: AVX-512, AVX2, SSE4.1, NEON auto-detection
- **CUDA Backend**: GPU-accelerated kernels via cudarc/NVRTC
- **Argmax Tracking**: For backpropagation in tropical neural networks
- **Cache-Optimized**: BLIS-style 5-loop blocking

## Installation

```toml
[dependencies]
tropical-gemm = "0.1"

# For GPU acceleration, add:
tropical-gemm-cuda = "0.1"
```

## Quick Start

### CPU (SIMD-optimized)

```rust
use tropical_gemm::{tropical_matmul, TropicalMaxPlus};

let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

// C[i,j] = max_k(A[i,k] + B[k,j])
let c = tropical_matmul::<TropicalMaxPlus<f32>>(&a, 2, 3, &b, 2);
assert_eq!(c[0].value(), 8.0); // max(1+1, 2+3, 3+5) = 8
```

### GPU (CUDA)

```rust
use tropical_gemm::TropicalMaxPlus;
use tropical_gemm_cuda::{tropical_matmul_gpu, CudaContext, GpuMatrix};

// One-shot API (handles memory automatically)
let c = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;

// Persistent context API (reuse for multiple operations)
let ctx = CudaContext::new()?;
let a_gpu = GpuMatrix::from_host_row_major(&ctx, &a, m, k)?;
let b_gpu = GpuMatrix::from_host_row_major(&ctx, &b, k, n)?;
let c_gpu = tropical_gemm_cuda::tropical_matmul_gpu_with_ctx::<TropicalMaxPlus<f32>>(&ctx, &a_gpu, &b_gpu)?;
let c = c_gpu.to_host_row_major(&ctx)?;
```

### Argmax Tracking (for Backpropagation)

```rust
use tropical_gemm::{tropical_matmul_with_argmax, TropicalMaxPlus};

let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
let b = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2

let result = tropical_matmul_with_argmax::<TropicalMaxPlus<f64>>(&a, 2, 3, &b, 2);

// Get the optimal value and which k produced it
let value = result.get(0, 0).value(); // 8.0
let k_idx = result.get_argmax(0, 0);  // 2 (k=2 gave max)
```

GPU argmax is also available:

```rust
use tropical_gemm::TropicalMaxPlus;
use tropical_gemm_cuda::tropical_matmul_gpu_with_argmax;

let (c, argmax) = tropical_matmul_gpu_with_argmax::<TropicalMaxPlus<f32>>(&a, m, k, &b, n)?;
// argmax[i*n + j] = k such that C[i,j] = A[i,k] + B[k,j]
```

## Supported Semirings

| Type | Addition (⊕) | Multiplication (⊗) | Zero | One | Use Case |
|------|--------------|-------------------|------|-----|----------|
| `TropicalMaxPlus<T>` | max | + | -∞ | 0 | Viterbi, longest path |
| `TropicalMinPlus<T>` | min | + | +∞ | 0 | Shortest path, Dijkstra |
| `TropicalMaxMul<T>` | max | × | 0 | 1 | Probability (non-log) |
| `TropicalAndOr` | OR | AND | false | true | Graph reachability |
| `CountingTropical<T,C>` | max+count | +,× | (-∞,0) | (0,1) | Path counting |

## Benchmark Results

Tested on NVIDIA RTX A4500 (Ampere, sm_86).

### GPU vs CPU Performance

| Size | CPU SIMD (ms) | GPU Kernel (ms) | Speedup |
|------|---------------|-----------------|---------|
| 256  | 4.1           | 0.032           | **128x** |
| 512  | 32.8          | 0.086           | **381x** |
| 1024 | 262.3         | 0.358           | **733x** |
| 2048 | 2091.6        | 2.510           | **833x** |

### Rust CUDA vs C Reference

Comparison with [TropicalGemm_Cuda](https://github.com/ArrogantGao/TropicalGemm_Cuda):

| Size | C Library (ms) | Rust CUDA (ms) | Ratio |
|------|----------------|----------------|-------|
| 256  | 0.028          | 0.032          | 1.14x |
| 512  | 0.074          | 0.086          | 1.16x |
| 1024 | 0.315          | 0.358          | 1.14x |
| 2048 | 2.224          | 2.509          | 1.13x |

The C library is ~13-16% faster due to pre-compiled PTX vs runtime compilation.

## Crate Structure

```
tropical-gemm/
├── tropical-gemm       # Main crate: types, CPU backend, public API
│   ├── types/          # Semiring type definitions
│   ├── core/           # BLIS-style GEMM algorithms
│   └── simd/           # SIMD kernels (AVX-512, AVX2, NEON)
│
└── tropical-gemm-cuda  # GPU backend (optional)
    └── kernels/        # CUDA kernel source
```

## Running Benchmarks

```bash
# CPU benchmark
cargo run --release --example bench_rust -p tropical-gemm

# CUDA vs CPU benchmark
cargo run --release --example bench_cuda_vs_cpu -p tropical-gemm-cuda
```

## License

MIT
