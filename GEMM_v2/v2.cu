#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#ifndef TILE_M
#define TILE_M 64     // 每个block覆盖的C子块行数
#endif
#ifndef TILE_N
#define TILE_N 64     // 每个block覆盖的C子块列数
#endif
#ifndef KSTEP
#define KSTEP 16      // K方向分块步长（每次搬入共享内存的K深度）
#endif
#ifndef THREADS_X
#define THREADS_X 16  // block内线程布局 X
#endif
#ifndef THREADS_Y
#define THREADS_Y 16  // block内线程布局 Y
#endif
#ifndef REG_TILE_M
#define REG_TILE_M 4  // 每线程在M方向计算的元素数
#endif
#ifndef REG_TILE_N
#define REG_TILE_N 4  // 每线程在N方向计算的元素数
#endif

// 访问器（row-major）
#define A_ELEM(ptr,i,j,K) (ptr[(size_t)(i)*(K) + (j)])
#define B_ELEM(ptr,i,j,N) (ptr[(size_t)(i)*(N) + (j)])
#define C_ELEM(ptr,i,j,N) (ptr[(size_t)(i)*(N) + (j)])

// C = A(MxK) * B(KxN) with register blocking
__global__ void matmul_regblock_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K, int N) {
    // 共享内存：A的 64xKSTEP，B的 KSTEPx64
    __shared__ float As[TILE_M][KSTEP];
    __shared__ float Bs[KSTEP][TILE_N];

    // 本block负责的C子块起点
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;

    // 线程在子块中的“基”行/列（每线程负责 REG_TILE_M×REG_TILE_N 个输出）
    const int tidx = threadIdx.x; // [0, THREADS_X)
    const int tidy = threadIdx.y; // [0, THREADS_Y)

    const int thread_row_base = tidy * REG_TILE_M; // 相对子块行
    const int thread_col_base = tidx * REG_TILE_N; // 相对子块列

    // 全局C里，这个线程负责的输出起始坐标
    const int row_base = block_row + thread_row_base; // 真实行
    const int col_base = block_col + thread_col_base; // 真实列

    // acc寄存器块
    float acc[REG_TILE_M][REG_TILE_N] = {0};

    // 沿K方向分块
    const int num_k_tiles = (K + KSTEP - 1) / KSTEP;
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        const int k_base = tk * KSTEP;

        // --- 将A子块搬到shared：形状 [TILE_M x KSTEP] ---
        // 让每个线程搬运 REG_TILE_M 行、KSTEP/THREADS_X 列（简单写法：每线程搬运 REG_TILE_M×REG_TILE_N 元素，也可按需调整）
        for (int rm = 0; rm < REG_TILE_M; ++rm) {
            int a_r = thread_row_base + rm;     // 子块内行
            int g_r = block_row + a_r;          // 全局行
            int a_c = tidx;                     // 子块内列的“基”由 x 线程选择
            int g_c = k_base + a_c;             // 全局K列
            if (a_r < TILE_M && g_r < M && g_c < K) {
                As[a_r][a_c] = A_ELEM(A, g_r, g_c, K);
            } else if (a_r < TILE_M && a_c < KSTEP) {
                As[a_r][a_c] = 0.0f;
            }
        }

        // --- 将B子块搬到shared：形状 [KSTEP x TILE_N] ---
        for (int cn = 0; cn < REG_TILE_N; ++cn) {
            int b_c = thread_col_base + cn;     // 子块内列
            int g_c = block_col + b_c;          // 全局列
            int b_r = tidy;                     // 子块内行“基”由 y 线程选择
            int g_r = k_base + b_r;             // 全局K行
            if (b_r < KSTEP && g_r < K && g_c < N) {
                Bs[b_r][b_c] = B_ELEM(B, g_r, g_c, N);
            } else if (b_r < KSTEP && b_c < TILE_N) {
                Bs[b_r][b_c] = 0.0f;
            }
        }

        __syncthreads();

        // --- 计算：acc += A( 4行 ) × B( 4列 ) over KSTEP ---
        #pragma unroll
        for (int kk = 0; kk < KSTEP; ++kk) {
            // 预取A 4行到寄存器
            float a_reg[REG_TILE_M];
            #pragma unroll
            for (int rm = 0; rm < REG_TILE_M; ++rm) {
                int a_r = thread_row_base + rm;
                a_reg[rm] = (a_r < TILE_M) ? As[a_r][kk] : 0.0f;
            }

            // 预取B 4列到寄存器
            float b_reg[REG_TILE_N];
            #pragma unroll
            for (int cn = 0; cn < REG_TILE_N; ++cn) {
                int b_c = thread_col_base + cn;
                b_reg[cn] = (b_c < TILE_N) ? Bs[kk][b_c] : 0.0f;
            }

            // FMA 累加
            #pragma unroll
            for (int rm = 0; rm < REG_TILE_M; ++rm) {
                #pragma unroll
                for (int cn = 0; cn < REG_TILE_N; ++cn) {
                    acc[rm][cn] += a_reg[rm] * b_reg[cn];
                }
            }
        }
        __syncthreads();
    }

    // --- 写回C（带边界判断） ---
    #pragma unroll
    for (int rm = 0; rm < REG_TILE_M; ++rm) {
        int r = row_base + rm;
        if (r >= M) continue;
        #pragma unroll
        for (int cn = 0; cn < REG_TILE_N; ++cn) {
            int c = col_base + cn;
            if (c < N) {
                C_ELEM(C, r, c, N) = acc[rm][cn];
            }
        }
    }
}

// ----------------- 测试主程序（预热3次、跑5次、平均） -----------------
int main() {
    int M = 1024, K = 1024, N = 1024;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);

    srand(42);
    for (int i = 0; i < M*K; ++i) hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K*N; ++i) hB[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);

    dim3 block(THREADS_X, THREADS_Y);
    dim3 grid((N + TILE_N - 1) / TILE_N,
              (M + TILE_M - 1) / TILE_M);

    // 预热 3 次
    for (int i = 0; i < 3; ++i) {
        matmul_regblock_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaDeviceSynchronize();

    // 计时 5 次
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 5;
    float total_ms = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_regblock_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Run %d: %.3f ms\n", i+1, ms);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // 结果与吞吐
    printf("Average time: %.3f ms\n", avg_ms);
    double ops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = (ops / (avg_ms / 1000.0)) / 1e9;
    printf("Avg Throughput: %.2f GFLOPS\n", gflops);

    // 简单检查
    printf("C[0,0]=%.3f  C[M/2,N/2]=%.3f  C[M-1,N-1]=%.3f\n",
           hC[0],
           hC[(M/2)*N + (N/2)],
           hC[(M-1)*N + (N-1)]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
