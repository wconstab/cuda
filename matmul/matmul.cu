#include <functional>
#include <stdio.h>

#include "matrix.h"
#include "kernel_matmul.hpp"
// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
__global__ void BlockMatMulKernel(const Matrix, const Matrix, Matrix, int sh_A_offs, int sh_B_offs);


void BlockMatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error! cuMalloc\n" );

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    int shared = 2 * A.width * dimBlock.y * sizeof(float);
    int sh_A_offs = 0;
    int sh_B_offs = A.width * dimBlock.y;
    BlockMatMulKernel<<<dimGrid, dimBlock, shared>>>(d_A, d_B, d_C, sh_A_offs, sh_B_offs);
    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error! Kernel\n" );

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error! cudaMemcpy DtoH\n" );
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

/**
    Cooperate to load relevant slice of A, B into shmem before multiplying
*/
__global__ void BlockMatMulKernel(Matrix A, Matrix B, Matrix C, int sh_A_offs, int sh_B_offs)
{
    extern __shared__ float s[];
    float* shared_A = &s[sh_A_offs];
    float* shared_B = &s[sh_B_offs];

    int iters = 2;
    int b_idx;
    int a_idx;
    int sh_idx;
    for(int iter = 0; iter < iters; iter++)
    {
        sh_idx = (threadIdx.y * A.width) + (iter * blockDim.x) + threadIdx.x;
        a_idx = (blockIdx.y * blockDim.y * A.width) + sh_idx;
        shared_A[sh_idx] = A.elements[a_idx];

        b_idx = (iter * blockDim.y * B.width)
              + (threadIdx.y * B.width)
              + (blockIdx.x * blockDim.x)
              + threadIdx.x;
        sh_idx = ((blockDim.x - 1 - threadIdx.x) * B.height)
               + (iter * blockDim.y)
               + threadIdx.y;
        shared_B[sh_idx] = B.elements[b_idx];
    }
    __syncthreads();

    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += shared_A[threadIdx.y * A.width + e]
                * shared_B[((blockDim.x - 1 - threadIdx.x) * A.width) + e];
    C.elements[row * C.width + col] = Cvalue;
}
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[threadIdx.y * A.width + e]
                * B.elements[threadIdx.y * A.width + e];
    C.elements[row * C.width + col] = Cvalue;
}
