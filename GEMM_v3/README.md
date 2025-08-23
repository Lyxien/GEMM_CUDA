# v4 实现

```c++
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
```

```c++
# 一次性加载 4 个数据
const float4* a4ptr = reinterpret_cast<const float4*>(&A_ELEM(A, row, k, K));
```