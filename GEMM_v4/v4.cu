#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define A_ELEM(p,i,j,K) (p[(size_t)(i)*(K) + (j)])
#define B_ELEM(p,i,j,N) (p[(size_t)(i)*(N) + (j)])
#define C_ELEM(p,i,j,N) (p[(size_t)(i)*(N) + (j)])

// 每线程计算同一行的 4 个相邻列（1x4），用 float4 读取 B；
// A[row,k] 由每个 warp 的 lane0 读取一次，然后 __shfl_sync 广播给同 warp。
__global__ void matmul_vec1x4_shfl(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N)
{
    // 线程块配置建议 block.x==32，这样每个 warp 在 x 方向排列，便于广播
    const int lane = threadIdx.x & 31;
    const unsigned full_mask = 0xFFFFFFFFu;

    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col0 = blockIdx.x * (blockDim.x * 4) + (threadIdx.x * 4); // 此线程负责的起始列
    if (row >= M || col0 >= N) return;

    float acc0=0.f, acc1=0.f, acc2=0.f, acc3=0.f;

    // 主循环：按标量 k 前进；A 用广播减少重复读；B 用 float4 连续读（行主序）
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        // 让每个 warp 的 lane0 读取 A[row,k]，再广播给同 warp
        float aik = 0.f;
        if (lane == 0) {
            aik = A_ELEM(A, row, k, K);
        }
        aik = __shfl_sync(full_mask, aik, 0); // 从 lane0 广播

        // 读取 B[k, col0..col0+3]
        int base = k * N + col0;
        if (((base & 3) == 0) && (col0 + 3 < N)) {
            // 16B 对齐且无越界 → 用 float4
            float4 b4 = *reinterpret_cast<const float4*>(&B[base]);
            acc0 += aik * b4.x;
            acc1 += aik * b4.y;
            acc2 += aik * b4.z;
            acc3 += aik * b4.w;
        } else {
            // 尾部或不对齐 → 标量回退
            if (col0 + 0 < N) acc0 += aik * B[base + 0];
            if (col0 + 1 < N) acc1 += aik * B[base + 1];
            if (col0 + 2 < N) acc2 += aik * B[base + 2];
            if (col0 + 3 < N) acc3 += aik * B[base + 3];
        }
    }

    // 写回（尽量用 float4，对齐不满足则标量回退）
    size_t out_idx = (size_t)row * N + col0;
    if (((out_idx & 3) == 0) && (col0 + 3 < N)) {
        *reinterpret_cast<float4*>(&C[out_idx]) = make_float4(acc0, acc1, acc2, acc3);
    } else {
        if (col0 + 0 < N) C[out_idx + 0] = acc0;
        if (col0 + 1 < N) C[out_idx + 1] = acc1;
        if (col0 + 2 < N) C[out_idx + 2] = acc2;
        if (col0 + 3 < N) C[out_idx + 3] = acc3;
    }
}

static void fill_rand(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX;
}

int main() {
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

    // 关键：block.x = 32（一个 warp），block.y 可取 8/16；grid.x 根据“每线程4列”计算
    dim3 block(32, 8); // 32×8 = 256 线程/块
    dim3 grid((N + (block.x*4) - 1) / (block.x*4),
              (M + block.y - 1) / block.y);

    // 预热
    for (int i = 0; i < 3; ++i) {
        matmul_vec1x4_shfl<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_ms = 0.f;
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(start);
        matmul_vec1x4_shfl<<<grid, block>>>(dA, dB, dC, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms=0.f; cudaEventElapsedTime(&ms, start, stop);
        printf("Run %d: %.3f ms\n", i+1, ms);
        total_ms += ms;
    }
    float avg_ms = total_ms / 5.f;
    double ops = 2.0 * (double)M * (double)K * (double)N;
    printf("Average time: %.3f ms,  Throughput: %.2f GFLOPS\n", avg_ms, (ops/(avg_ms/1000.0))/1e9);

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);
    printf("C[0,0]=%.3f  C[M/2,N/2]=%.3f  C[M-1,N-1]=%.3f\n",
           hC[0], hC[(M/2)*N + (N/2)], hC[(M-1)*N + (N-1)]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
