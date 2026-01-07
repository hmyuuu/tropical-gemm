#!/usr/bin/env julia
# Comparison benchmark: Rust CUDA vs Julia native CUDA kernels
# (CuTropicalGEMM requires CUDA ≤12.3, so we use native kernels)

using Pkg

for pkg in ["CUDA", "Printf"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using CUDA
using Printf

const SIZES = [256, 512, 1024, 2048]
const SAMPLES = 5

# Optimized GPU kernel with shared memory (similar to Rust kernel)
function tropical_maxplus_tiled!(C, A, B, M, N, K)
    TILE = 32

    tx = threadIdx().x
    ty = threadIdx().y
    bx = (blockIdx().x - 1) * TILE
    by = (blockIdx().y - 1) * TILE

    As = @cuStaticSharedMem(Float32, (32, 32))
    Bs = @cuStaticSharedMem(Float32, (32, 32))

    acc = typemin(Float32)

    for t in 0:TILE:(K-1)
        # Load tiles
        row = bx + tx
        col = t + ty
        if row <= M && col <= K
            As[tx, ty] = A[row, col]
        else
            As[tx, ty] = typemin(Float32)
        end

        row = t + tx
        col = by + ty
        if row <= K && col <= N
            Bs[tx, ty] = B[row, col]
        else
            Bs[tx, ty] = typemin(Float32)
        end

        sync_threads()

        for k in 1:TILE
            val = As[tx, k] + Bs[k, ty]
            acc = max(acc, val)
        end

        sync_threads()
    end

    row = bx + tx
    col = by + ty
    if row <= M && col <= N
        C[row, col] = acc
    end

    return nothing
end

# Simple kernel for comparison
function tropical_maxplus_simple!(C, A, B, M, N, K)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= N
        acc = typemin(Float32)
        for k in 1:K
            acc = max(acc, A[i, k] + B[k, j])
        end
        C[i, j] = acc
    end
    return nothing
end

function bench_kernel(kernel!, n::Int; tiled=false)
    A = CUDA.rand(Float32, n, n)
    B = CUDA.rand(Float32, n, n)
    C = CUDA.zeros(Float32, n, n)

    if tiled
        threads = (32, 32)
        blocks = (cld(n, 32), cld(n, 32))
    else
        threads = (16, 16)
        blocks = (cld(n, 16), cld(n, 16))
    end

    # Warmup
    CUDA.@sync @cuda threads=threads blocks=blocks kernel!(C, A, B, n, n, n)

    # Benchmark
    times = Float64[]
    for _ in 1:SAMPLES
        CUDA.synchronize()
        t0 = time_ns()
        CUDA.@sync @cuda threads=threads blocks=blocks kernel!(C, A, B, n, n, n)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e6)  # ms
    end

    return minimum(times)
end

function main()
    println("=" ^ 70)
    println("Julia Native CUDA Kernels Benchmark")
    println("(For comparison with Rust CUDA backend)")
    println("=" ^ 70)

    if !CUDA.functional()
        println("ERROR: CUDA not available!")
        return
    end

    println("GPU: $(CUDA.name(CUDA.device()))")
    println()

    # Warmup GPU
    CUDA.rand(Float32, 256, 256) * CUDA.rand(Float32, 256, 256)
    CUDA.synchronize()

    println("-" ^ 70)
    println("TropicalMaxPlus<f32> - Simple Kernel (no shared memory)")
    println("-" ^ 70)
    @printf("%6s  %12s\n", "Size", "Time (ms)")

    for n in SIZES
        t = bench_kernel(tropical_maxplus_simple!, n)
        @printf("%6d  %12.3f\n", n, t)
    end
    println()

    println("-" ^ 70)
    println("TropicalMaxPlus<f32> - Tiled Kernel (with shared memory)")
    println("-" ^ 70)
    @printf("%6s  %12s\n", "Size", "Time (ms)")

    for n in SIZES
        t = bench_kernel(tropical_maxplus_tiled!, n, tiled=true)
        @printf("%6d  %12.3f\n", n, t)
    end
    println()

    println("=" ^ 70)
    println("Compare with Rust CUDA (GPU-Pers column):")
    println("  Size    Rust (ms)")
    println("   256       0.032")
    println("   512       0.086")
    println("  1024       0.357")
    println("  2048       2.510")
    println("=" ^ 70)
end

main()
