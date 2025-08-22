#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 1024, K = 1024, N = 1024;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *a_cpu = (float*)malloc(bytesA);
    float *b_cpu = (float*)malloc(bytesB);
    float *c_cpu = (float*)malloc(bytesC);

    srand(42); // 固定随机种子
    for (int i = 0; i < M * K; ++i) a_cpu[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) b_cpu[i] = (float)rand() / RAND_MAX;

    float *a_gpu, *b_gpu, *c_gpu;
    cudaMalloc(&a_gpu, bytesA);
    cudaMalloc(&b_gpu, bytesB);
    cudaMalloc(&c_gpu, bytesC);

    cudaMemcpy(a_gpu, a_cpu, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, bytesB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -------- 预热 3 次 --------
    for (int i = 0; i < 3; ++i) {
        matmul_naive<<<grid, block>>>(a_gpu, b_gpu, c_gpu, M, K, N);
    }
    cudaDeviceSynchronize();

    // -------- 正式计时 5 次 --------
    float total_ms = 0.0f;
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(start);
        matmul_naive<<<grid, block>>>(a_gpu, b_gpu, c_gpu, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Run %d: %.3f ms\n", i + 1, ms);
        total_ms += ms;
    }

    float avg_ms = total_ms / 5.0f;
    printf("Average time over 5 runs: %.3f ms\n", avg_ms);

    // -------- 简单检查结果 --------
    cudaMemcpy(c_cpu, c_gpu, bytesC, cudaMemcpyDeviceToHost);
    printf("C[0,0] = %.3f\n", c_cpu[0]);
    printf("C[M/2,N/2] = %.3f\n", c_cpu[(M/2)*N + (N/2)]);
    printf("C[M-1,N-1] = %.3f\n", c_cpu[(M-1)*N + (N-1)]);

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    return 0;
}
