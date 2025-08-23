#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

// ---------------- 参数可调 ----------------
#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef KSTEP
#define KSTEP 32      // K 方向分块，必须是 4 的倍数
#endif
#ifndef THREADS_X
#define THREADS_X 16
#endif
#ifndef THREADS_Y
#define THREADS_Y 16
#endif
#ifndef REG_TILE_M
#define REG_TILE_M 4  // 每线程在 M 方向计算 4 个输出
#endif
#ifndef REG_TILE_N
#define REG_TILE_N 4  // 每线程在 N 方向计算 4 个输出
#endif

// ---------------- 访问器 ----------------
#define A_ELEM(ptr,i,j,K) (ptr[(size_t)(i)*(K) + (j)])
#define B_ELEM(ptr,i,j,N) (ptr[(size_t)(i)*(N) + (j)])
#define C_ELEM(ptr,i,j,N) (ptr[(size_t)(i)*(N) + (j)])

// C = A(MxK) * B(KxN)
// float4 向量化装载 A、B 到 shared；寄存器累加 4x4 输出
__global__ void matmul_regblock_f4_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int K, int N) {
    // shared：A 子块 [TILE_M x KSTEP]，B 子块 [KSTEP x TILE_N]
    __shared__ float As[TILE_M][KSTEP];
    __shared__ float Bs[KSTEP][TILE_N];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    // 本 block 覆盖的 C 子块左上角
    const int block_row = by * TILE_M;
    const int block_col = bx * TILE_N;

    // 本线程负责的 4x4 子片输出起点（相对子块）
    const int thread_row_base = ty * REG_TILE_M;
    const int thread_col_base = tx * REG_TILE_N;

    // 对应到全局 C 的坐标
    const int row_base = block_row + thread_row_base;
    const int col_base = block_col + thread_col_base;

    // 寄存器累加器
    float acc[REG_TILE_M][REG_TILE_N] = {0};

    // K 方向 tile 数
    const int ktiles = (K + KSTEP - 1) / KSTEP;

    for (int tk = 0; tk < ktiles; ++tk) {
        const int k_base = tk * KSTEP;

        // -------- 向量化装载 A 到 shared (沿 K 连续) --------
        // As 的每行长度为 KSTEP，按 float4 视图装载：KSTEP/4 个 float4
        const int A_vecs = KSTEP / 4;
        for (int a_r = ty; a_r < TILE_M; a_r += THREADS_Y) {
            int g_r = block_row + a_r;
            if (g_r < M) {
                // 循环覆盖这一行的所有 float4 段（以 THREADS_X 为步长分配给线程）
                for (int a_c4 = tx; a_c4 < A_vecs; a_c4 += THREADS_X) {
                    int g_k4 = (k_base >> 2) + a_c4;   // /4
                    int g_k   = g_k4 << 2;            // *4
                    // 边界判断（这里要求 K % 4 == 0 且 g_k+3 < K）
                    if (g_k + 3 < K) {
                        const float4* src = reinterpret_cast<const float4*>(
                            &A_ELEM(A, g_r, g_k, K));
                        float4 v = *src;
                        // 把 4 个标量写入 shared（也可把 As 行视作 float4* 写一次）
                        float* dst_row = &As[a_r][0];
                        reinterpret_cast<float4*>(dst_row)[a_c4] = v;
                    } else {
                        // 若需通用化，这里可做尾部标量装载；本示例假定 K % 4 == 0
                    }
                }
            } else {
                // 超界行置零
                for (int a_c4 = tx; a_c4 < A_vecs; a_c4 += THREADS_X) {
                    float* dst_row = &As[a_r][0];
                    reinterpret_cast<float4*>(dst_row)[a_c4] = make_float4(0,0,0,0);
                }
            }
        }

        // -------- 向量化装载 B 到 shared (沿 N 连续) --------
        // Bs 的每行长度为 TILE_N，按 float4 视图装载：TILE_N/4 个 float4
        const int B_vecs = TILE_N / 4;
        for (int b_r = ty; b_r < KSTEP; b_r += THREADS_Y) {
            int g_r = k_base + b_r;
            if (g_r < K) {
                for (int b_c4 = tx; b_c4 < B_vecs; b_c4 += THREADS_X) {
                    int g_c4 = ((block_col) >> 2) + b_c4; // /4
                    int g_c  = g_c4 << 2;                 // *4
                    if (g_c + 3 < N) {
                        const float4* src = reinterpret_cast<const float4*>(
                            &B_ELEM(B, g_r, g_c, N));
                        float4 v = *src;
                        float* dst_row = &Bs[b_r][0];
                        reinterpret_cast<float4*>(dst_row)[b_c4] = v;
                    } else {
                        // 同上，可加尾部处理；本示例假定 N % 4 == 0
                    }
                }
            } else {
                for (int b_c4 = tx; b_c4 < B_vecs; b_c4 += THREADS_X) {
                    float* dst_row = &Bs[b_r][0];
                    reinterpret_cast<float4*>(dst_row)[b_c4] = make_float4(0,0,0,0);
                }
            }
        }

        __syncthreads();

        // -------- 计算：acc += A(4行) × B(4列) over kk=0..KSTEP-1 --------
        #pragma unroll
        for (int kk = 0; kk < KSTEP; ++kk) {
            // 预取 A 的 4 行
            float a_reg[REG_TILE_M];
            #pragma unroll
            for (int rm = 0; rm < REG_TILE_M; ++rm) {
                int ar = thread_row_base + rm;
                a_reg[rm] = (ar < TILE_M) ? As[ar][kk] : 0.0f;
            }
            // 预取 B 的 4 列
            float b_reg[REG_TILE_N];
            #pragma unroll
            for (int cn = 0; cn < REG_TILE_N; ++cn) {
                int bc = thread_col_base + cn;
                b_reg[cn] = (bc < TILE_N) ? Bs[kk][bc] : 0.0f;
            }
            // FMA
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

    // -------- 写回 C --------
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

// ---------------- 测试主程序：预热3次，跑5次取平均 ----------------
int main() {
    int M = 1024, K = 1024, N = 1024;   // 确保 K%4==0 且 N%4==0

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

    dim3 block(THREADS_X, THREADS_Y);                 // 16x16
    dim3 grid((N + TILE_N - 1) / TILE_N,              // 64 列子块
              (M + TILE_M - 1) / TILE_M);             // 64 行子块

    // 预热
    for (int i = 0; i < 3; ++i) {
        matmul_regblock_f4_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    }
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 5;
    float total_ms = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_regblock_f4_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Run %d: %.3f ms\n", i+1, ms);
        total_ms += ms;
    }
    float avg_ms = total_ms / RUNS;

    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    printf("Average time: %.3f ms\n", avg_ms);
    double ops = 2.0 * (double)M * (double)K * (double)N;
    printf("Avg Throughput: %.2f GFLOPS\n", (ops / (avg_ms / 1000.0)) / 1e9);

    printf("C[0,0]=%.3f  C[M/2,N/2]=%.3f  C[M-1,N-1]=%.3f\n",
           hC[0], hC[(M/2)*N + (N/2)], hC[(M-1)*N + (N-1)]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
