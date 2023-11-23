#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define BlockSize 16
#define M 1024
#define N 1024
#define K 1024

// CUDA kernel for matrix multiplication
__global__ void matrixMulBasic(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

//Function to perform serial matrix multiplication
void matrixMultiplySerial(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int x = 0; x < K; x++) {
                sum += A[i * K + x] * b[x * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    // Allocate memory host matrices
    float *h_a = (float *)malloc(M * K * sizeof(float));
    float *h_b = (float *)malloc(K * N * sizeof(float));
    float *h_c = (float *)malloc(M * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = rand() % 100;
    }

    for (int i = 0; i < K * N; ++i) {
        h_b[i] = rand() % 100;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, M * K * sizeof(float));
    cudaMalloc((void **)&d_b, K * N * sizeof(float));
    cudaMalloc((void **)&d_c, M * N * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);
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
    matrixMultiplySerial(h_a, h_b, h_c, M, N, K);
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
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result (optional)
    printf("Result Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", h_c[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    // Release CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    
    return 0;
}
