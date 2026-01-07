#!/usr/bin/env julia
# Benchmark script for CuTropicalGEMM.jl GPU performance
# Compares with Rust implementation timing

using Pkg

# Ensure required packages are available
for pkg in ["CUDA", "CuTropicalGEMM", "TropicalNumbers", "BenchmarkTools", "Printf"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg)
    end
end

using CUDA
using CuTropicalGEMM
using TropicalNumbers
using BenchmarkTools
using Printf

# Matrix sizes to benchmark (matching Rust benchmarks)
const SIZES = [128, 256, 512, 1024, 2048, 4096]

# Warm up GPU
function warmup_gpu()
    println("Warming up GPU...")
    a = CUDA.rand(Float32, 256, 256)
    for _ in 1:10
        CUDA.@sync a * a
    end
    CUDA.synchronize()
    println("GPU warm-up complete")
end

# Benchmark a single configuration
function bench_matmul(::Type{T}, n::Int; samples=20) where T
    # Create matrices on GPU
    data = rand(Float32, n, n)
    a = CuArray(T.(data))

    # Warmup
    CUDA.@sync a * a

    # Benchmark with proper synchronization
    times = Float64[]
    for _ in 1:samples
        CUDA.synchronize()
        t0 = time_ns()
        CUDA.@sync a * a
        t1 = time_ns()
        push!(times, (t1 - t0) / 1e9)  # Convert to seconds
    end

    median_time = sort(times)[div(length(times), 2) + 1]
    min_time = minimum(times)

    return (median=median_time, min=min_time)
end

# Run all benchmarks
function run_benchmarks()
    println("=" ^ 70)
    println("CuTropicalGEMM.jl GPU Benchmark")
    println("=" ^ 70)
    println()

    # Check CUDA availability
    if !CUDA.functional()
        println("ERROR: CUDA not available!")
        return
    end

    device = CUDA.device()
    println("GPU: $(CUDA.name(device))")
    println("CUDA driver version: $(CUDA.driver_version())")
    println()

    warmup_gpu()
    println()

    # Results storage
    results = Dict{String, Vector{Tuple{Int, Float64, Float64}}}()

    # Benchmark configurations
    configs = [
        ("TropicalMaxPlus<f32>", Tropical{Float32}),
        ("TropicalMaxPlus<f64>", Tropical{Float64}),
        ("TropicalMinPlus<f32>", TropicalMinPlus{Float32}),
        ("TropicalMaxMul<f32>", TropicalMaxMul{Float32}),
    ]

    for (name, T) in configs
        println("-" ^ 70)
        println("Benchmarking: $name")
        println("-" ^ 70)
        @printf("%-8s  %12s  %12s  %12s\n", "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv")

        results[name] = Tuple{Int, Float64, Float64}[]

        for n in SIZES
            try
                result = bench_matmul(T, n)

                # Calculate equivalent GFLOPS (2*n^3 operations for matmul)
                gflops = 2.0 * n^3 / result.min / 1e9

                @printf("%-8d  %12.3f  %12.3f  %12.2f\n",
                        n, result.median * 1000, result.min * 1000, gflops)

                push!(results[name], (n, result.median, result.min))
            catch e
                @printf("%-8d  %12s  %12s  %12s\n", n, "ERROR", "ERROR", "ERROR")
                println("  Error: $e")
            end
        end
        println()
    end

    # Print summary for easy comparison with Rust
    println("=" ^ 70)
    println("Summary (min times in milliseconds) - for comparison with Rust")
    println("=" ^ 70)
    @printf("%-8s", "Size")
    for (name, _) in configs
        @printf("  %18s", name)
    end
    println()

    for (i, n) in enumerate(SIZES)
        @printf("%-8d", n)
        for (name, _) in configs
            if haskey(results, name) && i <= length(results[name])
                @printf("  %18.3f", results[name][i][3] * 1000)
            else
                @printf("  %18s", "N/A")
            end
        end
        println()
    end

    println()
    println("Benchmark complete!")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
