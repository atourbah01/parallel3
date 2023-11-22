#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define M 1024
#define N 1024
#define K 1024

// CUDA kernel for matrix multiplication
__global__ void matrixMulBasic(int *a, int *b, int *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int sum = 0;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
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
long long currentMillis(){
   struct timeval te;
   gettimeofday(&te, NULL);
   return te.tv_sec * 1000LL + te.tv_usec / 1000;
}

int main() {
    int *h_a, *h_b, *h_c; // host matrices
    int *d_a, *d_b, *d_c; // device matrices

    int size_a = M * K * sizeof(int);
    int size_b = K * N * sizeof(int);
    int size_c = M * N * sizeof(int);

    // Allocate memory for host matrices
    h_a = (int *)malloc(size_a);
    h_b = (int *)malloc(size_b);
    h_c = (int *)malloc(size_c);

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

    // Define block and grid dimensions
    dim3 dimBlock(16, 16, 1); // You can experiment with different block sizes
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    
    long long startParallel = currentMillis();
    // Launch the kernel
    matrixMulBasic<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);
    long long endParallel = currentMillis();
    printf("Parallel Execution Time: %lld milliseconds\n",endParallel-startParallel);
    
    long long startSerial = currentMillis();
    matrixMultiplySerial(h_a,h_b,h_c,M,N,K);
    long long endSerial = currentMillis();
    printf("Serial Execution Time: %lld milliseconds\n",endSerial-startSerial);
    
    // Calculate Speedup factor, efficiency and scalability
    double speedup= (double) (endSerial-startSerial)/(endParallel-startParallel);
    double efficiency= speedup / (dimGrid.x*dimGrid.y*dimBlock.x*dimBlock.y);
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

    return 0;
}
