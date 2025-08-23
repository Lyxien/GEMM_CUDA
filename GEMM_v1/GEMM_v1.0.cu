#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#ifndef BLOCK_Y
#define BLOCK_Y 128   // 每个 block 负责的行数（1 列 × BLOCK_Y 行）
#endif

#define A(i, j) A[(size_t)(i) * K + (j)]
#define B(i, j) B[(size_t)(i) * N + (j)]
#define C(i, j) C[(size_t)(i) * N + (j)]


#ifndef BLOCK_Y
#define BLOCK_Y 128
#endif

// 访问器宏（row-major）
#define A(i, j) A[(size_t)(i) * K + (j)]
#define B(i, j) B[(size_t)(i) * N + (j)]
#define C(i, j) C[(size_t)(i) * N + (j)]

// 计算：C = A(MxK) * B(KxN)
// 仅缓存 B 的一整列到 shared memory
__global__ void matmul_shared_Bcol(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N) {
    extern __shared__ float sBcol[];   // 动态共享内存：大小 = K * sizeof(float)

    int col = blockIdx.x;                               // 本 block 处理的列
    int row = blockIdx.y * BLOCK_Y + threadIdx.y;       // 本线程处理的行

    // 将 B 的第 col 列搬到 shared memory，y 维线程分段加载
    for (int kk = threadIdx.y; kk < K; kk += BLOCK_Y) {
        if (col < N) sBcol[kk] = B(kk, col);
    }
    __syncthreads();

    // 累加求和
    if (row < M && col < N) {
        float acc = 0.0f;
        #pragma unroll 1
        for (int k = 0; k < K; ++k) {
            acc += A(row, k) * sBcol[k];
        }
        C(row, col) = acc;
    }
}


int main() {
    // 尺寸
    int M = 1024, K = 1024, N = 1024;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    // 主机内存
    float *a_cpu = (float*)malloc(bytesA);
    float *b_cpu = (float*)malloc(bytesB);
    float *c_cpu = (float*)malloc(bytesC);

    // 固定随机种子并初始化
    srand(42);
    for (int i = 0; i < M * K; ++i) a_cpu[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) b_cpu[i] = (float)rand() / RAND_MAX;

    // 设备内存
    float *a_gpu, *b_gpu, *c_gpu;
    cudaMalloc(&a_gpu, bytesA);
    cudaMalloc(&b_gpu, bytesB);
    cudaMalloc(&c_gpu, bytesC);

    cudaMemcpy(a_gpu, a_cpu, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b_cpu, bytesB, cudaMemcpyHostToDevice);

    // 启动配置：grid.x = N（每列一个 block），grid.y 覆盖所有行的条带
    dim3 block(1, BLOCK_Y); // 只用 y 维做行并行
    dim3 grid(N, (M + BLOCK_Y - 1) / BLOCK_Y);

    // 动态共享内存大小：缓存一整列 B 的 K 个元素
    size_t shm_bytes = (size_t)K * sizeof(float);

    // 事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热 3 次
    for (int i = 0; i < 3; ++i) {
        matmul_shared_Bcol<<<grid, block, shm_bytes>>>(a_gpu, b_gpu, c_gpu, M, K, N);
    }
    cudaDeviceSynchronize();

    // 正式计时 5 次
    const int RUNS = 5;
    float total_ms = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_shared_Bcol<<<grid, block, shm_bytes>>>(a_gpu, b_gpu, c_gpu, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Run %d: %.3f ms\n", i + 1, ms);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;

    // 结果与吞吐
    cudaMemcpy(c_cpu, c_gpu, bytesC, cudaMemcpyDeviceToHost);
    printf("Average time over %d runs: %.3f ms\n", RUNS, avg_ms);
    double ops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = (ops / (avg_ms / 1000.0)) / 1e9;
    printf("Avg Throughput: %.2f GFLOPS\n", gflops);

    // 简单检查几个元素
    printf("C[0,0]=%.3f  C[M/2,N/2]=%.3f  C[M-1,N-1]=%.3f\n",
           c_cpu[0],
           c_cpu[(M/2)*N + (N/2)],
           c_cpu[(M-1)*N + (N-1)]);

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
