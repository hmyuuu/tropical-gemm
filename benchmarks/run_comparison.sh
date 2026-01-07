#!/bin/bash
# Benchmark comparison script: Rust CPU vs Julia GPU (CuTropicalGEMM)
#
# Usage: ./run_comparison.sh [options]
#   --rust-only    Run only Rust benchmarks
#   --julia-only   Run only Julia benchmarks
#   --quick        Run quick benchmarks (smaller sizes)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
RUN_RUST=true
RUN_JULIA=true
QUICK_MODE=false

for arg in "$@"; do
    case $arg in
        --rust-only)
            RUN_JULIA=false
            ;;
        --julia-only)
            RUN_RUST=false
            ;;
        --quick)
            QUICK_MODE=true
            ;;
    esac
done

echo "========================================================================"
echo "Tropical GEMM Benchmark Comparison"
echo "========================================================================"
echo ""
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo ""

# Create results directory
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run Rust benchmarks
if [ "$RUN_RUST" = true ]; then
    echo "========================================================================"
    echo "Running Rust CPU Benchmarks"
    echo "========================================================================"
    echo ""

    cd "$PROJECT_DIR"

    # Build release
    echo "Building release binary..."
    cargo build --release --example bench_rust

    # Run benchmark
    echo ""
    echo "Running benchmark..."
    RUST_OUTPUT="$RESULTS_DIR/rust_${TIMESTAMP}.txt"
    cargo run --release --example bench_rust 2>&1 | tee "$RUST_OUTPUT"

    echo ""
    echo "Rust results saved to: $RUST_OUTPUT"
fi

# Run Julia benchmarks
if [ "$RUN_JULIA" = true ]; then
    echo ""
    echo "========================================================================"
    echo "Running Julia GPU Benchmarks (CuTropicalGEMM)"
    echo "========================================================================"
    echo ""

    JULIA_SCRIPT="$SCRIPT_DIR/bench_julia.jl"

    if [ ! -f "$JULIA_SCRIPT" ]; then
        echo "ERROR: Julia benchmark script not found at $JULIA_SCRIPT"
        exit 1
    fi

    # Check if Julia is available
    if ! command -v julia &> /dev/null; then
        echo "ERROR: Julia not found in PATH"
        echo "Please install Julia: https://julialang.org/downloads/"
        exit 1
    fi

    echo "Julia version: $(julia --version)"
    echo ""

    # Run Julia benchmark
    JULIA_OUTPUT="$RESULTS_DIR/julia_${TIMESTAMP}.txt"
    julia "$JULIA_SCRIPT" 2>&1 | tee "$JULIA_OUTPUT"

    echo ""
    echo "Julia results saved to: $JULIA_OUTPUT"
fi

# Print summary
echo ""
echo "========================================================================"
echo "Benchmark Complete"
echo "========================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""

if [ "$RUN_RUST" = true ] && [ "$RUN_JULIA" = true ]; then
    echo "To compare results, examine:"
    echo "  Rust: $RESULTS_DIR/rust_${TIMESTAMP}.txt"
    echo "  Julia: $RESULTS_DIR/julia_${TIMESTAMP}.txt"
fi
