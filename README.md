# jjxb

%%writefile vector_matrix_cuda.cu
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;
using namespace std::chrono;

__global__ void vectorAdd(int* A, int* B, int* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__global__ void matMul(int* A, int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

void vectorAdd_seq(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; i++) C[i] = A[i] + B[i];
}

void matMul_seq(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

void printVector(int* arr, int n, int limit = 10) {
    int show = min(n, limit);
    cout << "  [";
    for (int i = 0; i < show; i++) {
        cout << arr[i];
        if (i < show - 1) cout << ", ";
    }
    if (n > limit) cout << " ... (" << n - limit << " more)";
    cout << "]\n";
}

void printMatrix(int* mat, int N, int limit = 4) {
    int show = min(N, limit);
    for (int i = 0; i < show; i++) {
        cout << "  [";
        for (int j = 0; j < show; j++) {
            printf("%6d", mat[i * N + j]);
            if (j < show - 1) cout << ", ";
        }
        if (N > limit) cout << " ...";
        cout << "]\n";
    }
    if (N > limit) cout << "  ... (" << N - limit << " more rows)\n";
}

bool verifyVector(int* A, int* B, int n) {
    for (int i = 0; i < n; i++) if (A[i] != B[i]) return false;
    return true;
}

bool verifyMatrix(int* A, int* B, int N) {
    for (int i = 0; i < N * N; i++) if (A[i] != B[i]) return false;
    return true;
}

void testVectorAddition(int testNum, int n) {
    cout << "\n########################################\n";
    cout << "  VECTOR ADDITION - TEST CASE " << testNum << "\n";
    cout << "########################################\n";
    cout << "Vector size: " << n << "\n";
    int size = n * sizeof(int);
    int* h_A = new int[n]; int* h_B = new int[n];
    int* h_C_seq = new int[n]; int* h_C_par = new int[n];
    srand(testNum * 42);
    for (int i = 0; i < n; i++) { h_A[i] = rand() % 1000; h_B[i] = rand() % 1000; }
    cout << "\nVector A: \n"; printVector(h_A, n);
    cout << "Vector B: \n"; printVector(h_B, n);
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();
    vectorAdd_seq(h_A, h_B, h_C_seq, n);
    t2 = high_resolution_clock::now();
    double seq_time = duration_cast<duration<double>>(t2 - t1).count();
    cout << "\n----- Sequential -----\n";
    cout << "Result: \n"; printVector(h_C_seq, n);
    printf("  Time: %.6f seconds\n", seq_time);
    int* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int blockSize = 256; int numBlocks = (n + blockSize - 1) / blockSize;
    t1 = high_resolution_clock::now();
    vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    double par_time = duration_cast<duration<double>>(t2 - t1).count();
    cudaMemcpy(h_C_par, d_C, size, cudaMemcpyDeviceToHost);
    cout << "\n----- Parallel (CUDA) -----\n";
    cout << "  Grid: " << numBlocks << " blocks, Block: " << blockSize << " threads\n";
    cout << "Result: \n"; printVector(h_C_par, n);
    printf("  Time: %.6f seconds\n", par_time);
    cout << "\n  Verification: " << (verifyVector(h_C_seq, h_C_par, n) ? "PASS" : "FAIL") << "\n";
    double speedup = (par_time > 0) ? seq_time / par_time : 0;
    printf("  Speedup: %.2fx\n", speedup);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C_seq; delete[] h_C_par;
}

void testMatrixMultiplication(int testNum, int N) {
    cout << "\n########################################\n";
    cout << "  MATRIX MULTIPLICATION - TEST CASE " << testNum << "\n";
    cout << "########################################\n";
    cout << "Matrix size: " << N << " x " << N << "\n";
    int size = N * N * sizeof(int);
    int* h_A = new int[N*N]; int* h_B = new int[N*N];
    int* h_C_seq = new int[N*N]; int* h_C_par = new int[N*N];
    srand(testNum * 99);
    for (int i = 0; i < N*N; i++) { h_A[i] = rand() % 10; h_B[i] = rand() % 10; }
    cout << "\nMatrix A:\n"; printMatrix(h_A, N);
    cout << "Matrix B:\n"; printMatrix(h_B, N);
    high_resolution_clock::time_point t1, t2;
    t1 = high_resolution_clock::now();
    matMul_seq(h_A, h_B, h_C_seq, N);
    t2 = high_resolution_clock::now();
    double seq_time = duration_cast<duration<double>>(t2 - t1).count();
    cout << "\n----- Sequential -----\n";
    cout << "Result C = A x B:\n"; printMatrix(h_C_seq, N);
    printf("  Time: %.6f seconds\n", seq_time);
    int* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size); cudaMalloc(&d_B, size); cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int BS = 16;
    dim3 threads(BS, BS);
    dim3 blocks((N+BS-1)/BS, (N+BS-1)/BS);
    t1 = high_resolution_clock::now();
    matMul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    double par_time = duration_cast<duration<double>>(t2 - t1).count();
    cudaMemcpy(h_C_par, d_C, size, cudaMemcpyDeviceToHost);
    cout << "\n----- Parallel (CUDA) -----\n";
    cout << "  Grid: " << blocks.x << "x" << blocks.y << " blocks, Block: " << BS << "x" << BS << " threads\n";
    cout << "Result C = A x B:\n"; printMatrix(h_C_par, N);
    printf("  Time: %.6f seconds\n", par_time);
    cout << "\n  Verification: " << (verifyMatrix(h_C_seq, h_C_par, N) ? "PASS" : "FAIL") << "\n";
    double speedup = (par_time > 0) ? seq_time / par_time : 0;
    printf("  Speedup: %.2fx\n", speedup);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C_seq; delete[] h_C_par;
}

int main() {
    cout << "=== CUDA: Vector Addition & Matrix Multiplication ===\n";
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cout << "\nGPU: " << prop.name << "\n";
        cout << "SMs: " << prop.multiProcessorCount;
        cout << ", Max Threads/Block: " << prop.maxThreadsPerBlock;
        cout << ", Memory: " << prop.totalGlobalMem/(1024*1024) << " MB\n";
    }
    cout << "\n======= PART 1: VECTOR ADDITION =======\n";
    testVectorAddition(1, 10000);
    testVectorAddition(2, 100000);
    testVectorAddition(3, 1000000);
    cout << "\n======= PART 2: MATRIX MULTIPLICATION =======\n";
    testMatrixMultiplication(1, 64);
    testMatrixMultiplication(2, 256);
    testMatrixMultiplication(3, 512);
    cout << "\n======= ALL TESTS COMPLETE =======\n";
    return 0;
}
