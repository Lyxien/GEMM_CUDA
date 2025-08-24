#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace nvcuda;

// 定义矩阵的维度
const int M = 1024;
const int N = 1024;
const int K = 1024;

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half *a, half *b, float *c, 
                             int M, int N, int K, 
                             float alpha, float beta) 
{

    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;
    
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    // Declare the fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // set o in accumulator fragment
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
        
        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

void matrixMultiply(half* A, half* B, float* C, int M, int N, int K, float alpha, float beta) {
    // Set the grid and block size for the kernel launch
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);


    // Launch the kernel
    wmma_example<<<grid, block>>>(A, B, C, M, N, K, alpha, beta);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Error checking after kernel execution
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

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
    // Allocate memory on the host
    half *h_A, *h_B;
    float *h_C;
    cudaMallocHost(&h_A, M * K * sizeof(half));
    cudaMallocHost(&h_B, K * N * sizeof(half));
    cudaMallocHost(&h_C, M * N * sizeof(float));

    float *h_A_check, *h_B_check, *h_C_check;
    cudaMallocHost(&h_A_check, M * K * sizeof(float));
    cudaMallocHost(&h_B_check, K * N * sizeof(float));
    cudaMallocHost(&h_C_check, M * N * sizeof(float));

    srand(42); // 固定随机种子
    // Initialize matrices A and B with some values
    for (int i = 0; i < M * K; i++){
        float ta = (float)rand() / RAND_MAX;
        h_A[i] = __float2half(ta);
        h_A_check[i] = ta;
    }
    for (int i = 0; i < K * N; i++){ 
        float tb = (float)rand() / RAND_MAX;
        h_B[i] = __float2half(tb);
        h_B_check[i] = tb;
    }
    for (int i = 0; i < M * N; i++){
        h_C[i] = 0.0f;
        h_C_check[i] = 0.0f;
    }

    // Allocate memory on the device
    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    float *a_gpu, *b_gpu, *c_gpu;
    cudaMalloc(&a_gpu, M * K * sizeof(float));
    cudaMalloc(&b_gpu, K * N * sizeof(float));
    cudaMalloc(&c_gpu, M * N * sizeof(float));
    cudaMemcpy(a_gpu, h_A_check, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, h_B_check, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication
    float alpha = 1.0f;
    float beta = 1.0f;
    matrixMultiply(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    printf("C[0,0] = %.3f\n", h_C[0]);
    printf("C[M/2,N/2] = %.3f\n", h_C[(M/2)*N + (N/2)]);
    printf("C[M-1,N-1] = %.3f\n", h_C[(M-1)*N + (N-1)]);

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(a_gpu, b_gpu, c_gpu, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_check, c_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("C[0,0] = %.3f\n", h_C_check[0]);
    printf("C[M/2,N/2] = %.3f\n", h_C_check[(M/2)*N + (N/2)]);
    printf("C[M-1,N-1] = %.3f\n", h_C_check[(M-1)*N + (N-1)]);

    // Clean up memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
