# tropical-gemm

[![CI](https://github.com/TensorBFS/tropical-gemm/actions/workflows/ci.yml/badge.svg)](https://github.com/TensorBFS/tropical-gemm/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/TensorBFS/tropical-gemm/branch/main/graph/badge.svg)](https://codecov.io/gh/TensorBFS/tropical-gemm)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://tensorbfs.github.io/tropical-gemm/)

High-performance tropical matrix multiplication in Rust with SIMD and CUDA backends. Inspired by [CuTropicalGEMM.jl](https://github.com/TensorBFS/CuTropicalGEMM.jl).

## Features

- **Multiple Semirings**: MaxPlus, MinPlus, MaxMul
- **SIMD Acceleration**: AVX-512, AVX2, SSE4.1, NEON auto-detection
- **CUDA Backend**: GPU-accelerated kernels via NVRTC
- **Argmax Tracking**: For backpropagation in tropical neural networks
- **Python Bindings**: NumPy and PyTorch integration

## Installation

```toml
[dependencies]
tropical-gemm = "0.1"
tropical-gemm-cuda = "0.1"  # Optional GPU support
```

## Quick Start

```rust
use tropical_gemm::{Mat, MaxPlus};

let a = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
let b = Mat::<MaxPlus<f32>>::from_row_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);

// C[i,j] = max_k(A[i,k] + B[k,j])
let c = a.matmul(&b);
assert_eq!(c.get_value(0, 0), 8.0); // max(1+1, 2+3, 3+5) = 8
```

### Python

```bash
# Install from PyPI
pip install tropical-gemm

# With PyTorch support (for automatic differentiation)
pip install tropical-gemm[torch]

# For development with GPU support (requires CUDA toolkit)
pip install maturin
git clone https://github.com/TensorBFS/tropical-gemm
cd tropical-gemm/crates/tropical-gemm-python
maturin develop --features cuda
```

#### Basic Usage

```python
import numpy as np
import tropical_gemm

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

# MaxPlus: C[i,j] = max_k(A[i,k] + B[k,j])
c = tropical_gemm.maxplus_matmul(a, b)

# MinPlus: C[i,j] = min_k(A[i,k] + B[k,j])
c = tropical_gemm.minplus_matmul(a, b)

# MaxMul: C[i,j] = max_k(A[i,k] * B[k,j])
c = tropical_gemm.maxmul_matmul(a, b)

# With argmax tracking (for backpropagation)
c, argmax = tropical_gemm.maxplus_matmul_with_argmax(a, b)

# GPU acceleration (if compiled with CUDA)
if tropical_gemm.cuda_available():
    c = tropical_gemm.maxplus_matmul_gpu(a, b)
```

#### PyTorch Integration

For neural network training with tropical operations:

```python
import torch
from tropical_gemm.pytorch import (
    tropical_maxplus_matmul,
    tropical_minplus_matmul,
    tropical_maxmul_matmul,
    # GPU variants (requires CUDA)
    tropical_maxplus_matmul_gpu,
    tropical_minplus_matmul_gpu,
    tropical_maxmul_matmul_gpu,
    GPU_AVAILABLE,
)

# Create tensors with gradients
a = torch.randn(100, 50, requires_grad=True)
b = torch.randn(50, 80, requires_grad=True)

# Forward pass - computes tropical matmul
c = tropical_maxplus_matmul(a, b)

# Backward pass - gradients flow through automatically
loss = c.sum()
loss.backward()

print(a.grad.shape)  # (100, 50)
print(b.grad.shape)  # (50, 80)

# GPU acceleration for larger matrices
if GPU_AVAILABLE:
    c = tropical_maxplus_matmul_gpu(a, b)
```

**Gradient Semantics:**
- **MaxPlus/MinPlus**: Additive rule - `grad_A[i,k] += grad_C[i,j]` for `k = argmax[i,j]`
- **MaxMul**: Multiplicative rule - `grad_A[i,k] += grad_C[i,j] * B[k,j]` for `k = argmax[i,j]`

## Documentation

📖 **[User Guide](https://tensorbfs.github.io/tropical-gemm/)** - Installation, tutorials, examples

📚 **[API Reference](https://tensorbfs.github.io/tropical-gemm/api/tropical_gemm/)** - Rust API documentation

## Semirings

| Type | ⊕ | ⊗ | Use Case |
|------|---|---|----------|
| `MaxPlus<T>` | max | + | Longest path, Viterbi |
| `MinPlus<T>` | min | + | Shortest path |
| `MaxMul<T>` | max | × | Max probability |

## Performance

| Size | CPU (ms) | GPU (ms) | Speedup |
|------|----------|----------|---------|
| 256 | 4.1 | 0.03 | 128x |
| 1024 | 262 | 0.36 | 728x |
| 2048 | 2092 | 2.5 | 837x |

## License

MIT
