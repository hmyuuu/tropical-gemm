//! Quick benchmark for comparison with Julia/CUDA
//!
//! Run with: cargo run --release --example bench_rust

use std::time::Instant;
use tropical_gemm::prelude::*;

const SIZES: &[usize] = &[128, 256, 512, 1024, 2048, 4096];
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

fn bench_maxplus_f32(n: usize) -> (f64, f64) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n * n)
        .map(|i| (((i + 500) % 1000) as f32) * 0.01)
        .collect();

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);
    }

    // Benchmark
    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul::<TropicalMaxPlus<f32>>(&a, n, n, &b, n);
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[0])
}

fn bench_maxplus_f64(n: usize) -> (f64, f64) {
    let a: Vec<f64> = (0..n * n).map(|i| ((i % 1000) as f64) * 0.01).collect();
    let b: Vec<f64> = (0..n * n)
        .map(|i| (((i + 500) % 1000) as f64) * 0.01)
        .collect();

    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul::<TropicalMaxPlus<f64>>(&a, n, n, &b, n);
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul::<TropicalMaxPlus<f64>>(&a, n, n, &b, n);
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[0])
}

fn bench_minplus_f32(n: usize) -> (f64, f64) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n * n)
        .map(|i| (((i + 500) % 1000) as f32) * 0.01)
        .collect();

    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul::<TropicalMinPlus<f32>>(&a, n, n, &b, n);
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul::<TropicalMinPlus<f32>>(&a, n, n, &b, n);
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[0])
}

fn bench_maxmul_f32(n: usize) -> (f64, f64) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 1000) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n * n)
        .map(|i| (((i + 500) % 1000) as f32) * 0.01)
        .collect();

    for _ in 0..WARMUP_ITERS {
        let _ = tropical_matmul::<TropicalMaxMul<f32>>(&a, n, n, &b, n);
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let _ = tropical_matmul::<TropicalMaxMul<f32>>(&a, n, n, &b, n);
        times.push(start.elapsed().as_secs_f64());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[times.len() / 2], times[0])
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("Tropical GEMM Rust CPU Benchmark");
    println!("{}", "=".repeat(70));
    println!();
    println!("Backend: {}", Backend::description());
    println!();

    let mut results: Vec<(&str, Vec<(usize, f64, f64)>)> = Vec::new();

    // Benchmark TropicalMaxPlus<f32>
    {
        println!("{}", "-".repeat(70));
        println!("Benchmarking: TropicalMaxPlus<f32>");
        println!("{}", "-".repeat(70));
        println!(
            "{:<8}  {:>12}  {:>12}  {:>12}",
            "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv"
        );

        let mut data = Vec::new();
        for &n in SIZES {
            let (median, min) = bench_maxplus_f32(n);
            let gflops = 2.0 * (n as f64).powi(3) / min / 1e9;
            println!(
                "{:<8}  {:>12.3}  {:>12.3}  {:>12.2}",
                n,
                median * 1000.0,
                min * 1000.0,
                gflops
            );
            data.push((n, median, min));
        }
        results.push(("TropicalMaxPlus<f32>", data));
        println!();
    }

    // Benchmark TropicalMaxPlus<f64>
    {
        println!("{}", "-".repeat(70));
        println!("Benchmarking: TropicalMaxPlus<f64>");
        println!("{}", "-".repeat(70));
        println!(
            "{:<8}  {:>12}  {:>12}  {:>12}",
            "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv"
        );

        let mut data = Vec::new();
        for &n in &SIZES[..5] {
            let (median, min) = bench_maxplus_f64(n);
            let gflops = 2.0 * (n as f64).powi(3) / min / 1e9;
            println!(
                "{:<8}  {:>12.3}  {:>12.3}  {:>12.2}",
                n,
                median * 1000.0,
                min * 1000.0,
                gflops
            );
            data.push((n, median, min));
        }
        results.push(("TropicalMaxPlus<f64>", data));
        println!();
    }

    // Benchmark TropicalMinPlus<f32>
    {
        println!("{}", "-".repeat(70));
        println!("Benchmarking: TropicalMinPlus<f32>");
        println!("{}", "-".repeat(70));
        println!(
            "{:<8}  {:>12}  {:>12}  {:>12}",
            "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv"
        );

        let mut data = Vec::new();
        for &n in SIZES {
            let (median, min) = bench_minplus_f32(n);
            let gflops = 2.0 * (n as f64).powi(3) / min / 1e9;
            println!(
                "{:<8}  {:>12.3}  {:>12.3}  {:>12.2}",
                n,
                median * 1000.0,
                min * 1000.0,
                gflops
            );
            data.push((n, median, min));
        }
        results.push(("TropicalMinPlus<f32>", data));
        println!();
    }

    // Benchmark TropicalMaxMul<f32>
    {
        println!("{}", "-".repeat(70));
        println!("Benchmarking: TropicalMaxMul<f32>");
        println!("{}", "-".repeat(70));
        println!(
            "{:<8}  {:>12}  {:>12}  {:>12}",
            "Size", "Median (ms)", "Min (ms)", "GFLOPS equiv"
        );

        let mut data = Vec::new();
        for &n in SIZES {
            let (median, min) = bench_maxmul_f32(n);
            let gflops = 2.0 * (n as f64).powi(3) / min / 1e9;
            println!(
                "{:<8}  {:>12.3}  {:>12.3}  {:>12.2}",
                n,
                median * 1000.0,
                min * 1000.0,
                gflops
            );
            data.push((n, median, min));
        }
        results.push(("TropicalMaxMul<f32>", data));
        println!();
    }

    // Print summary
    println!("{}", "=".repeat(70));
    println!("Summary (min times in milliseconds) - for comparison with Julia/CUDA");
    println!("{}", "=".repeat(70));
    print!("{:<8}", "Size");
    for (name, _) in &results {
        print!("  {:>18}", name);
    }
    println!();

    for (i, &n) in SIZES.iter().enumerate() {
        print!("{:<8}", n);
        for (_, data) in &results {
            if i < data.len() {
                print!("  {:>18.3}", data[i].2 * 1000.0);
            } else {
                print!("  {:>18}", "N/A");
            }
        }
        println!();
    }

    println!();
    println!("Benchmark complete!");
}
