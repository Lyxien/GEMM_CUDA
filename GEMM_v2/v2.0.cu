#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// ===== 可调参数 =====
#ifndef REG_TILE_M
#define REG_TILE_M 4     // 每线程在 M 方向计算的元素数
#endif
#ifndef REG_TILE_N
#define REG_TILE_N 4     // 每线程在 N 方向计算的元素数
#endif
#ifndef THREADS_X
#define THREADS_X 16     // block 内线程布局 X
#endif
#ifndef THREADS_Y
#define THREADS_Y 16     // block 内线程布局 Y
#endif

// 由上面决定每个 block 覆盖的 C 子块大小
#define TILE_M (THREADS_Y * REG_TILE_M)
#define TILE_N (THREADS_X * REG_TILE_N)

// 行主序访问宏
#define A_ELEM(p,i,j,K) (p[(size_t)(i)*(K) + (j)])
#define B_ELEM(p,i,j,N) (p[(size_t)(i)*(N) + (j)])
#define C_ELEM(p,i,j,N) (p[(size_t)(i)*(N) + (j)])

// C = A(MxK) * B(KxN)
// 无 shared memory，寄存器分块：每线程计算 REG_TILE_M×REG_TILE_N（默认 4×4）
__global__ void matmul_regblock_nosmem(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // 本 block 覆盖的 C 子块左上角
    const int block_row = by * TILE_M;
    const int block_col = bx * TILE_N;

    // 本线程负责的 4×4 输出起点（相对子块）
    const int tmr = ty * REG_TILE_M;
    const int tnc = tx * REG_TILE_N;

    // 对应到全局 C 的坐标
    const int row_base = block_row + tmr;
    const int col_base = block_col + tnc;

    // 4×4 寄存器累加器
    float acc[REG_TILE_M][REG_TILE_N] = {0};

    // 沿 K 累加
    for (int k = 0; k < K; ++k) {
        // 预取 A 的 4 个行元素到寄存器
        float a_reg[REG_TILE_M];
        #pragma unroll
        for (int rm = 0; rm < REG_TILE_M; ++rm) {
            int r = row_base + rm;
            a_reg[rm] = (r < M) ? A_ELEM(A, r, k, K) : 0.0f;
        }
        // 预取 B 的 4 个列元素到寄存器
        float b_reg[REG_TILE_N];
        #pragma unroll
        for (int cn = 0; cn < REG_TILE_N; ++cn) {
            int c = col_base + cn;
            b_reg[cn] = (c < N) ? B_ELEM(B, k, c, N) : 0.0f;
        }

        // 4×4 FMA
        #pragma unroll
        for (int rm = 0; rm < REG_TILE_M; ++rm) {
            #pragma unroll
            for (int cn = 0; cn < REG_TILE_N; ++cn) {
                acc[rm][cn] += a_reg[rm] * b_reg[cn];
            }
        }
    }

    // 写回 C（带边界判断）
    #pragma unroll
    for (int rm = 0; rm < REG_TILE_M; ++rm) {
        int r = row_base + rm;
        if (r >= M) continue;
        #pragma unroll
        for (int cn = 0; cn < REG_TILE_N; ++cn) {
            int c = col_base + cn;
            if (c < N) C_ELEM(C, r, c, N) = acc[rm][cn];
        }
    }
}

// ===== 一个最简主程序：随机输入 + 预热3次 + 跑5次平均 =====
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

    dim3 block(THREADS_X, THREADS_Y); // 默认 16×16
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    // 预热
    for (int i = 0; i < 3; ++i) {
        matmul_regblock_nosmem<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaDeviceSynchronize();

    // 计时 5 次
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float total_ms = 0.f;
    for (int i = 0; i < 5; ++i) {
        cudaEventRecord(start);
        matmul_regblock_nosmem<<<grid, block>>>(dA, dB, dC, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms=0.f; cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
        printf("Run %d: %.3f ms\n", i+1, ms);
    }
    float avg = total_ms / 5.f;

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    double ops = 2.0 * (double)M * (double)K * (double)N;
    printf("Average: %.3f ms,  %.2f GFLOPS\n", avg, (ops / (avg / 1000.0)) / 1e9);
    printf("C[0,0]=%.3f  C[M/2,N/2]=%.3f  C[M-1,N-1]=%.3f\n",
           hC[0], hC[(M/2)*N + (N/2)], hC[(M-1)*N + (N-1)]);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
