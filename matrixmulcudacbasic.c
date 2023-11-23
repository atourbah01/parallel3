#include <stdio.h>
#include <cuda.h>

#define BlockSize 16
#define M 1024
#define N 1024
#define K 1024

// CUDA kernel for matrix multiplication
__global__ void matrixMulBasic(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int sum = 0;
        for (int i = 0; i < K; i++) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}
//Function to perform serial matrix multiplication
void matrixMultiplySerial(int *a, int *b, int *c, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
            for (int x = 0; x < k; x++) {
                sum += a[i * k + x] * b[x * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main() {
    float *h_a, *h_b, *h_c; // host matrices
    float *d_a, *d_b, *d_c; // device matrices

    float size_a = M * K * sizeof(float);
    float size_b = K * N * sizeof(float);
    float size_c = M * N * sizeof(float);

    // Allocate memory for host matrices
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = rand() % 10;
    }

    for (int i = 0; i < K * N; ++i) {
        h_b[i] = rand() % 10;
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size_a);
    cudaMalloc((void **)&d_b, size_b);
    cudaMalloc((void **)&d_c, size_c);

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop, start1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Define block and grid dimensions
    dim3 dimBlock(BlockSize, BlockSize); 
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    matrixMulBasic<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float paralleltime = 0;
    cudaEventElapsedTime(&paralleltime, start, stop);
    printf("Parallel Execution Time: %.2f milliseconds\n",paralleltime);
    
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    matrixMultiplySerial(h_a,h_b,h_c,M,N,K);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float sequentialtime = 0;
    cudaEventElapsedTime(&sequentialtime, start1, stop1);
    printf("Serial Execution Time: %.2f milliseconds\n", sequentialtime);
    
    // Calculate Speedup factor, efficiency and scalability
    double speedup= (double) sequentialtime/paralleltime;
    double efficiency= speedup / (dimGrid.x*dimGrid.y);
    double scalability= speedup / (dimGrid.x*dimGrid.y*dimBlock.x*dimBlock.y);
    
    printf("Speedup: %.2f\n", speedup);
    printf("Efficiency: %.2f\n", efficiency);
    printf("Scalability: %.2f\n", scalability);
    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // Print the result (optional)
    printf("Result Matrix:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d\t", h_c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Release CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    return 0;
}
