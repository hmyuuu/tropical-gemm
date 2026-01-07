#!/usr/bin/env julia
# Benchmark script for tropical GEMM on GPU using native Julia CUDA
# This is a fallback when CuTropicalGEMM is not available due to CUDA version

using Pkg

for pkg in ["CUDA", "TropicalNumbers", "Printf"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using CUDA
using TropicalNumbers
using Printf

const SIZES = [128, 256, 512, 1024, 2048, 4096]
const SAMPLES = 10

function warmup_gpu()
    println("Warming up GPU...")
    a = CUDA.rand(Float32, 256, 256)
    for _ in 1:10
        CUDA.@sync a * a
    end
    CUDA.synchronize()
    println("GPU warm-up complete")
end

# GPU kernel for tropical max-plus matmul
function tropical_maxplus_kernel!(C, A, B, M, N, K)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= N
        max_val = typemin(Float32)
        for k in 1:K
            val = A[i, k] + B[k, j]
            max_val = max(max_val, val)
        end
        C[i, j] = max_val
    end
    return nothing
end

# GPU kernel for tropical min-plus matmul
function tropical_minplus_kernel!(C, A, B, M, N, K)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= N
        min_val = typemax(Float32)
        for k in 1:K
            val = A[i, k] + B[k, j]
            min_val = min(min_val, val)
        end
        C[i, j] = min_val
    end
    return nothing
end

# GPU kernel for tropical max-mul matmul
function tropical_maxmul_kernel!(C, A, B, M, N, K)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= M && j <= N
        max_val = Float32(0)
        for k in 1:K
            val = A[i, k] * B[k, j]
            max_val = max(max_val, val)
        end
        C[i, j] = max_val
    end
    return nothing
end

function bench_gpu_kernel(kernel!, n::Int; samples=SAMPLES)
    A = CUDA.rand(Float32, n, n)
    B = CUDA.rand(Float32, n, n)
    C = CUDA.zeros(Float32, n, n)

    threads = (16, 16)
    blocks = (cld(n, threads[1]), cld(n, threads[2]))

    # Warmup
    CUDA.@sync @cuda threads=threads blocks=blocks kernel!(C, A, B, n, n, n)

    # Benchmark
    times = Float64[]
    for _ in 1:samples
        CUDA.synchronize()
        t0 = time_ns()
        CUDA.@sync @cuda threads=threads blocks=blocks kernel!(C, A, B, n, n, n)
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end

    sort!(times)
    return (median=times[div(length(times), 2) + 1], min=minimum(times))
end

# Also benchmark standard BLAS GEMM for reference
function bench_cublas_gemm(n::Int; samples=SAMPLES)
    A = CUDA.rand(Float32, n, n)
    B = CUDA.rand(Float32, n, n)

    # Warmup
    CUDA.@sync A * B

    times = Float64[]
    for _ in 1:samples
        CUDA.synchronize()
        t0 = time_ns()
        CUDA.@sync A * B
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)
    end

    sort!(times)
    return (median=times[div(length(times), 2) + 1], min=minimum(times))
end

function run_benchmarks()
    println("=" ^ 70)
    println("Tropical GEMM GPU Benchmark (Native CUDA Kernels)")
    println("=" ^ 70)
    println()

    if !CUDA.functional()
        println("ERROR: CUDA not available!")
        return
    end

    device = CUDA.device()
    println("GPU: $(CUDA.name(device))")
    println("CUDA driver: $(CUDA.driver_version())")
    println("Note: Using custom CUDA kernels (CuTropicalGEMM requires CUDA ≤12.3)")
    println()

    warmup_gpu()
    println()

    results = Dict{String, Vector{Tuple{Int, Float64, Float64}}}()

    # Benchmark cuBLAS GEMM for reference
    println("-" ^ 70)
    println("Reference: cuBLAS SGEMM (standard matmul)")
    println("-" ^ 70)
    @printf("%-8s  %12s  %12s  %12s\n", "Size", "Median (ms)", "Min (ms)", "TFLOPS")

    results["cuBLAS"] = Tuple{Int, Float64, Float64}[]
    for n in SIZES
        result = bench_cublas_gemm(n)
        tflops = 2.0 * n^3 / result.min / 1e12
        @printf("%-8d  %12.3f  %12.3f  %12.3f\n", n, result.median * 1000, result.min * 1000, tflops)
        push!(results["cuBLAS"], (n, result.median, result.min))
    end
    println()

    # Benchmark tropical MaxPlus
    println("-" ^ 70)
    println("Benchmarking: TropicalMaxPlus<f32> (GPU kernel)")
    println("-" ^ 70)
    @printf("%-8s  %12s  %12s  %12s\n", "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv")

    results["MaxPlus"] = Tuple{Int, Float64, Float64}[]
    for n in SIZES
        result = bench_gpu_kernel(tropical_maxplus_kernel!, n)
        gflops = 2.0 * n^3 / result.min / 1e9
        @printf("%-8d  %12.3f  %12.3f  %12.2f\n", n, result.median * 1000, result.min * 1000, gflops)
        push!(results["MaxPlus"], (n, result.median, result.min))
    end
    println()

    # Benchmark tropical MinPlus
    println("-" ^ 70)
    println("Benchmarking: TropicalMinPlus<f32> (GPU kernel)")
    println("-" ^ 70)
    @printf("%-8s  %12s  %12s  %12s\n", "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv")

    results["MinPlus"] = Tuple{Int, Float64, Float64}[]
    for n in SIZES
        result = bench_gpu_kernel(tropical_minplus_kernel!, n)
        gflops = 2.0 * n^3 / result.min / 1e9
        @printf("%-8d  %12.3f  %12.3f  %12.2f\n", n, result.median * 1000, result.min * 1000, gflops)
        push!(results["MinPlus"], (n, result.median, result.min))
    end
    println()

    # Benchmark tropical MaxMul
    println("-" ^ 70)
    println("Benchmarking: TropicalMaxMul<f32> (GPU kernel)")
    println("-" ^ 70)
    @printf("%-8s  %12s  %12s  %12s\n", "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv")

    results["MaxMul"] = Tuple{Int, Float64, Float64}[]
    for n in SIZES
        result = bench_gpu_kernel(tropical_maxmul_kernel!, n)
        gflops = 2.0 * n^3 / result.min / 1e9
        @printf("%-8d  %12.3f  %12.3f  %12.2f\n", n, result.median * 1000, result.min * 1000, gflops)
        push!(results["MaxMul"], (n, result.median, result.min))
    end
    println()

    # Summary table
    println("=" ^ 70)
    println("Summary (min times in ms) - GPU vs Rust CPU comparison")
    println("=" ^ 70)
    @printf("%-8s  %12s  %12s  %12s  %12s\n", "Size", "cuBLAS", "MaxPlus", "MinPlus", "MaxMul")

    for (i, n) in enumerate(SIZES)
        @printf("%-8d", n)
        for key in ["cuBLAS", "MaxPlus", "MinPlus", "MaxMul"]
            if haskey(results, key) && i <= length(results[key])
                @printf("  %12.3f", results[key][i][3] * 1000)
            else
                @printf("  %12s", "N/A")
            end
        end
        println()
    end

    println()
    println("Benchmark complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
