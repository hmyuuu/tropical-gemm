// Benchmark for TropicalGemm_Cuda C library
// Compile: nvcc -O3 -arch=sm_86 -o bench_c_cuda bench_c_cuda.cu -L/home/leo/pycode/TropicalGemm_Cuda/lib -l_TropicalMaxPlus_FP32 -l_TropicalMinPlus_FP32 -l_TropicalMaxMul_FP32 -lcudart
// Run: LD_LIBRARY_PATH=/home/leo/pycode/TropicalGemm_Cuda/lib:$LD_LIBRARY_PATH ./bench_c_cuda

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <vector>

// Function signatures from TropicalGemm_Cuda
extern "C" {
    void FLOAT_maxplus(int m, int n, int k, float* A, float* B, float* C,
                       float alpha, float beta, char TA, char TB, cudaStream_t stream);
    void FLOAT_minplus(int m, int n, int k, float* A, float* B, float* C,
                       float alpha, float beta, char TA, char TB, cudaStream_t stream);
    void FLOAT_maxmul(int m, int n, int k, float* A, float* B, float* C,
                      float alpha, float beta, char TA, char TB, cudaStream_t stream);
}

const int SIZES[] = {256, 512, 1024, 2048};
const int NUM_SIZES = 4;
const int WARMUP = 2;
const int SAMPLES = 5;

double bench_kernel(void (*kernel)(int, int, int, float*, float*, float*, float, float, char, char, cudaStream_t),
                    int n) {
    float *d_A, *d_B, *d_C;
    size_t bytes = n * n * sizeof(float);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Initialize with random data
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < n * n; i++) {
        h_data[i] = (float)(rand() % 1000) * 0.01f;
    }
    cudaMemcpy(d_A, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_data, bytes, cudaMemcpyHostToDevice);
    free(h_data);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        kernel(n, n, n, d_A, d_B, d_C, 1.0f, 0.0f, 'N', 'N', stream);
        cudaStreamSynchronize(stream);
    }

    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < SAMPLES; i++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        kernel(n, n, n, d_A, d_B, d_C, 1.0f, 0.0f, 'N', 'N', stream);
        cudaStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());

    cudaStreamDestroy(stream);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return times[0];  // Return min time
}

int main() {
    printf("====================================================================\n");
    printf("TropicalGemm_Cuda C Library Benchmark\n");
    printf("====================================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    printf("--------------------------------------------------------------------\n");
    printf("TropicalMaxPlus<f32>\n");
    printf("--------------------------------------------------------------------\n");
    printf("%6s  %12s\n", "Size", "Time (ms)");

    for (int i = 0; i < NUM_SIZES; i++) {
        int n = SIZES[i];
        double t = bench_kernel(FLOAT_maxplus, n);
        printf("%6d  %12.3f\n", n, t);
    }
    printf("\n");

    printf("--------------------------------------------------------------------\n");
    printf("TropicalMinPlus<f32>\n");
    printf("--------------------------------------------------------------------\n");
    printf("%6s  %12s\n", "Size", "Time (ms)");

    for (int i = 0; i < NUM_SIZES; i++) {
        int n = SIZES[i];
        double t = bench_kernel(FLOAT_minplus, n);
        printf("%6d  %12.3f\n", n, t);
    }
    printf("\n");

    printf("--------------------------------------------------------------------\n");
    printf("TropicalMaxMul<f32>\n");
    printf("--------------------------------------------------------------------\n");
    printf("%6s  %12s\n", "Size", "Time (ms)");

    for (int i = 0; i < NUM_SIZES; i++) {
        int n = SIZES[i];
        double t = bench_kernel(FLOAT_maxmul, n);
        printf("%6d  %12.3f\n", n, t);
    }
    printf("\n");

    printf("====================================================================\n");
    printf("Compare with Rust CUDA (GPU-Pers column from Rust benchmark)\n");
    printf("====================================================================\n");

    return 0;
}
