// gemm_v4_0.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef TILE
#define TILE 32            // 方形 tile 尺寸（16/32 常用）
#endif
#ifndef SMEM_PAD
#define SMEM_PAD 1         // shared 第二维 padding，缓解 bank 冲突
#endif

// v4.0：Tiled GEMM + 双缓冲（As[2], Bs[2]）
__global__ void gemm_v4_0_tiled_double_buffer(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int K, int N)
{
    __shared__ float As[2][TILE][TILE + SMEM_PAD];
    __shared__ float Bs[2][TILE][TILE + SMEM_PAD];

    const int row = blockIdx.y * TILE + threadIdx.y; // C 的行
    const int col = blockIdx.x * TILE + threadIdx.x; // C 的列
    const int num_tiles = (K + TILE - 1) / TILE;

    float acc = 0.0f;

    // 预取第 0 段到 buf=0
    int buf = 0;
    {
        int a_row = row;
        int a_col = 0 * TILE + threadIdx.x;
        if (a_row < M && a_col < K)
            As[buf][threadIdx.y][threadIdx.x] = A[(size_t)a_row * K + a_col];
        else
            As[buf][threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = 0 * TILE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N)
            Bs[buf][threadIdx.y][threadIdx.x] = B[(size_t)b_row * N + b_col];
        else
            Bs[buf][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // 主循环：当前 buf 计算，同时 next 预取
    for (int tk = 0; tk < num_tiles; ++tk) {
        // 预取下一段到 next 缓冲
        int next = buf ^ 1;
        if (tk + 1 < num_tiles) {
            int a_row = row;
            int a_col = (tk + 1) * TILE + threadIdx.x;
            if (a_row < M && a_col < K)
                As[next][threadIdx.y][threadIdx.x] = A[(size_t)a_row * K + a_col];
            else
                As[next][threadIdx.y][threadIdx.x] = 0.0f;

            int b_row = (tk + 1) * TILE + threadIdx.y;
            int b_col = col;
            if (b_row < K && b_col < N)
                Bs[next][threadIdx.y][threadIdx.x] = B[(size_t)b_row * N + b_col];
            else
                Bs[next][threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 用当前 buf 计算 TILE×TILE 的部分内积
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE; ++k_inner) {
            acc += As[buf][threadIdx.y][k_inner] * Bs[buf][k_inner][threadIdx.x];
        }

        __syncthreads();   // 确保 next 预取完成再切换
        buf = next;
    }

    if (row < M && col < N) {
        C[(size_t)row * N + col] = acc;
    }
}

// ----------------- 简单驱动与基准 -----------------
static void fill_rand(float* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX;
}

template <typename KERN>
float bench(const char* tag, KERN k, dim3 grid, dim3 block,
            const float* dA, const float* dB, float* dC,
            int M, int K, int N, int warm=3, int runs=5)
{
    for (int i=0;i<warm;++i) k<<<grid,block>>>(dA,dB,dC,M,K,N);
    cudaDeviceSynchronize();

    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    float total=0.f;
    for (int i=0;i<runs;++i) {
        cudaEventRecord(s);
        k<<<grid,block>>>(dA,dB,dC,M,K,N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);
        printf("%s  Run %d: %.3f ms\n", tag, i+1, ms);
        total += ms;
    }
    float avg = total / runs;
    double ops = 2.0 * (double)M * (double)K * (double)N;
    printf("%s  avg: %.3f ms,  %.2f GFLOPS\n\n", tag, avg, (ops/(avg/1000.0))/1e9);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return avg;
}

int main() {
    int M=1024, K=1024, N=1024;

    size_t bytesA=(size_t)M*K*sizeof(float);
    size_t bytesB=(size_t)K*N*sizeof(float);
    size_t bytesC=(size_t)M*N*sizeof(float);

    float *hA=(float*)malloc(bytesA);
    float *hB=(float*)malloc(bytesB);
    float *hC=(float*)malloc(bytesC);

    srand(42);
    fill_rand(hA,(size_t)M*K);
    fill_rand(hB,(size_t)K*N);

    float *dA,*dB,*dC;
    cudaMalloc(&dA,bytesA);
    cudaMalloc(&dB,bytesB);
    cudaMalloc(&dC,bytesC);
    cudaMemcpy(dA,hA,bytesA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,bytesB,cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    bench("v4.0  Tiled+DoubleBuffer",
          gemm_v4_0_tiled_double_buffer, grid, block,
          dA, dB, dC, M, K, N);

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);
    printf("C[0,0]=%.6f  C[M/2,N/2]=%.6f  C[M-1,N-1]=%.6f\n",
           hC[0], hC[(M/2)*N + (N/2)], hC[(M-1)*N + (N-1)]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
