#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cuda.h>

#define M 1024
#define N 1024
#define K 1024
#define TILE_SIZE 16

// CUDA kernel for matrix multiplication with tiling
__global__ void matrixMulTiled(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int result = 0;

    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < m && t * TILE_SIZE + threadIdx.x < k)
            tileA[threadIdx.y][threadIdx.x] = a[row * k + t * TILE_SIZE + threadIdx.x];
         else
            tileA[threadIdx.y][threadIdx.x] = 0;
        
        if (col < n && t * TILE_SIZE + threadIdx.y < k)
            tileB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];
            
         else
            tileB[threadIdx.y][threadIdx.x] = 0;
        
        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();

        // Compute partial result for the tile
        for (int i = 0; i < TILE_SIZE; i++) {
            result += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        // Synchronize to ensure all threads have finished using the tiles
        __syncthreads();
    }

    // Store the final result
    if (row < m && col < n) {
        c[row * n + col] = result;
    }
}


// Function to perform matrix multiplication on the CPU
void matrixMulCPU(float *a, float *b, float *c, int m, int n, int k) {
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
    float *h_a, *h_b, *h_c, *h_c_cpu; 
    float *d_a, *d_b, *d_c; // device matrices

    // Allocate memory for host matrices
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    h_c_cpu = (float *)malloc(size_c);

    // Initialize matrices
    for (int i = 0; i < M * K; ++i) {
        h_a[i] = rand() % 10; 
    }

    for (int i = 0; i < K * N; ++i) {
        h_b[i] = rand() % 10; 
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, M * K * sizeof(float));
    cudaMalloc((void **)&d_b, K * N * sizeof(float));
    cudaMalloc((void **)&d_c, M * N * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Record start time for parallel execution
    cudaEvent_t start, stop, start1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    // Launch the tiled kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);

    // Record end time for parallel execution
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float parallelTime;
    cudaEventElapsedTime(&parallelTime, start, stop);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Record start time for sequential execution (on CPU)
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, 0);
    
    // Perform matrix multiplication on the CPU
    matrixMulCPU(h_a, h_b, h_c_cpu, M, N, K);

    
    // Record end time for sequential execution (on CPU)
    cudaEventRecord(stop1, 0);
    cudaEventSynchronize(stop1);
    float sequentialTime;
    cudaEventElapsedTime(&sequentialTime, start1, stop1);

    // Calculate speedup, efficiency, and scalability
    double speedup = (double) sequentialTime / parallelTime;
    double efficiency = speedup / (dimGrid.x * dimGrid.y);
    double scalability = speedup / (dimGrid.x * dimGrid.y * dimBlock.x * dimBlock.y);

    // Print performance metrics
    printf("Parallel Execution Time: %.2f ms\n", parallelTime);
    printf("Sequential Execution Time: %.2f s\n", sequentialTime);
    printf("Speedup: %.2f\n", speedup);
    printf("Efficiency: %.2f\n", efficiency);
    printf("Scalability: %.2f\n", scalability);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
    // Release CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    return 0;
}
