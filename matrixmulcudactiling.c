#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define M 1024
#define N 1024
#define K 1024
#define TILE_SIZE 16

// CUDA kernel for matrix multiplication with tiling
__global__ void matrixMulTiled(int *a, int *b, int *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int result = 0;

    // Loop over tiles
    for (int t = 0; t < k / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        tileA[threadIdx.y][threadIdx.x] = a[row * k + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = b[(t * TILE_SIZE + threadIdx.y) * n + col];

        // Synchronize to ensure all threads have loaded their tiles
        __syncthreads();

        // Compute partial result for the tile
        for (int i = 0; i < TILE_SIZE; ++i) {
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
void matrixMulCPU(int *a, int *b, int *c, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int x = 0; x < k; ++x) {
                sum += a[i * k + x] * b[x * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// Function to check the correctness of the result
void checkResult(int *h_c, int *h_c_cpu, int size) {
    for (int i = 0; i < size; ++i) {
        if (h_c[i] != h_c_cpu[i]) {
            printf("Result verification failed at element %d!\n", i);
            return;
        }
    }
    printf("Result verification passed!\n");
}

int main() {
    int *h_a, *h_b, *h_c, *h_c_cpu; // host matrices
    int *d_a, *d_b, *d_c; // device matrices

    int size_a = M * K * sizeof(int);
    int size_b = K * N * sizeof(int);
    int size_c = M * N * sizeof(int);

    // Allocate memory for host matrices
    h_a = (int *)malloc(size_a);
    h_b = (int *)malloc(size_b);
    h_c = (int *)malloc(size_c);
    h_c_cpu = (int *)malloc(size_c);

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
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    // Record start time for parallel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch the tiled kernel
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N, K);

    // Record end time for parallel execution
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float parallelTime;
    cudaEventElapsedTime(&parallelTime, start, stop);

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // Record start time for sequential execution (on CPU)
    struct timeval cpu_start, cpu_stop;
    gettimeofday(&cpu_start, NULL);

    // Perform matrix multiplication on the CPU
    matrixMulCPU(h_a, h_b, h_c_cpu, M, N, K);

    // Record end time for sequential execution (on CPU)
    gettimeofday(&cpu_stop, NULL);
    double cpuTime = (cpu_stop.tv_sec - cpu_start.tv_sec) + (cpu_stop.tv_usec - cpu_start.tv_usec) / 1e6;

    // Check the correctness of the result
    checkResult(h_c, h_c_cpu, M * N);

    // Calculate speedup, efficiency, and scalability
    float speedup = cpuTime / (parallelTime / 1000.0); // Convert parallel time to seconds
    float efficiency = speedup / dimGrid.x / dimGrid.y;
    float scalability = speedup / dimGrid.x / dimGrid.y / (dimBlock.x * dimBlock.y);

    // Print performance metrics
    printf("Parallel Execution Time: %f ms\n", parallelTime);
    printf("Sequential Execution Time: %f s\n", cpuTime);
    printf("Speedup: %f\n", speedup);
    printf("Efficiency: %f\n", efficiency);
    printf("Scalability: %f\n", scalability);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
