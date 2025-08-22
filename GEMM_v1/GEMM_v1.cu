#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#ifndef TILE
#define TILE 32   // 也可改为 16，看硬件/占用情况
#endif

// C = A(MxK) * B(KxN)
__global__ void matmul_tiled_shared(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y; // C 的行
    int col = blockIdx.x * TILE + threadIdx.x; // C 的列

    float acc = 0.0f;

    // 沿着 K 维分块（每次处理 K 上的 TILE 列/行）
    for (int tk = 0; tk < (K + TILE - 1) / TILE; ++tk) {
        // 从全局内存加载 A 的一个 tile 到 shared
        int a_row = row;
        int a_col = tk * TILE + threadIdx.x;
        if (a_row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // 加载 B 的一个 tile 到 shared
        int b_row = tk * TILE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算当前 tile 对应的部分乘加
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE; ++k_inner) {
            acc += As[threadIdx.y][k_inner] * Bs[k_inner][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

int main() {
    // 矩阵尺寸
    int M = 1024, K = 1024, N = 1024;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    // 主机内存
    float *a_cpu = (float*)malloc(bytesA);
    float *b_cpu = (float*)malloc(bytesB);
    float *c_cpu = (float*)malloc(bytesC);

    // 固定种子，生成 [0,1) 随机数
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

    // 启动配置
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // 事件计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热 3 次
    for (int i = 0; i < 3; ++i) {
        matmul_tiled_shared<<<grid, block>>>(a_gpu, b_gpu, c_gpu, M, K, N);
    }
    cudaDeviceSynchronize();

    // 正式计时 5 次
    const int RUNS = 5;
    float total_ms = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_tiled_shared<<<grid, block>>>(a_gpu, b_gpu, c_gpu, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Run %d: %.3f ms\n", i + 1, ms);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;

    // 拷回并简单检查数值
    cudaMemcpy(c_cpu, c_gpu, bytesC, cudaMemcpyDeviceToHost);
    printf("Average time over %d runs: %.3f ms\n", RUNS, avg_ms);

    // 计算平均 GFLOPS（2*M*K*N 次浮点运算）
    double ops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = (ops / (avg_ms / 1000.0)) / 1e9;
    printf("Avg Throughput: %.2f GFLOPS\n", gflops);

    // 打印几个元素看看
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
