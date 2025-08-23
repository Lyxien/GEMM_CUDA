#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

// 行主序访问宏
#define A_ELEM(p,i,j,K) (p[(size_t)(i)*(K) + (j)])
#define B_ELEM(p,i,j,N) (p[(size_t)(i)*(N) + (j)])
#define C_ELEM(p,i,j,N) (p[(size_t)(i)*(N) + (j)])

__global__ void matmul_naive_float4(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // C 的行
    int col = blockIdx.x * blockDim.x + threadIdx.x; // C 的列
    if (row >= M || col >= N) return;

    float acc = 0.0f;

    // --- 主循环：K 维按 4 步长处理，A 使用 float4 向量化加载 ---
    int k4_end = (K / 4) * 4;  // 向下对齐到 4 的倍数
    for (int k = 0; k < k4_end; k += 4) {
        // A[row, k..k+3] 连续，适合 float4
        const float4* a4ptr = reinterpret_cast<const float4*>(&A_ELEM(A, row, k, K));
        float4 a4 = *a4ptr;

        // B[k..k+3, col] 跨行，不连续，保持标量加载
        float b0 = B_ELEM(B, k + 0, col, N);
        float b1 = B_ELEM(B, k + 1, col, N);
        float b2 = B_ELEM(B, k + 2, col, N);
        float b3 = B_ELEM(B, k + 3, col, N);

        // FMA 累加
        acc += a4.x * b0;
        acc += a4.y * b1;
        acc += a4.z * b2;
        acc += a4.w * b3;
    }

    // --- 尾部（K 不是 4 的倍数时） ---
    for (int k = k4_end; k < K; ++k) {
        acc += A_ELEM(A, row, k, K) * B_ELEM(B, k, col, N);
    }

    C_ELEM(C, row, col, N) = acc;
}

static void fill_rand(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX;
}

int main() {
    // 尺寸（可改）
    int M = 1024, K = 1024, N = 1024;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);

    srand(42);
    fill_rand(hA, (size_t)M*K);
    fill_rand(hB, (size_t)K*N);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);
    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);

    // 启动配置（常见的 16x16）
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // 预热 3 次
    for (int i = 0; i < 3; ++i) {
        matmul_naive_float4<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaDeviceSynchronize();

    // 计时 5 次
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(start);
        matmul_naive_float4<<<grid, block>>>(dA, dB, dC, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Run %d: %.3f ms\n", i + 1, ms);
        total_ms += ms;
    }
    float avg_ms = total_ms / 5.0f;

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // 统计
    double ops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = (ops / (avg_ms / 1000.0)) / 1e9;
    printf("Average time: %.3f ms,  Throughput: %.2f GFLOPS\n", avg_ms, gflops);
    printf("C[0,0]=%.3f  C[M/2,N/2]=%.3f  C[M-1,N-1]=%.3f\n",
           hC[0], hC[(M/2)*N + (N/2)], hC[(M-1)*N + (N-1)]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
