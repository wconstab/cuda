#include <functional>
#include <stdio.h>

#include "matrix.h"
#include "kernel_matmul.hpp"

#define BLOCK_SIZE 16

/**
    Load only square blocks of A, B and accumultate into shared block of C.

*/

int ceil_div(int a, int b) {
    if (a % b == 0) {
        return a / b;
    } else {
        return (a / b) + 1;
    }
}

__device__ int dev_ceil_div(int a, int b) {
    if (a % b == 0) {
        return a / b;
    } else {
        return (a / b) + 1;
    }
}

__device__ float* block_offset(float* base, int width, int y_block, int x_block) {
    return base + (y_block * width * BLOCK_SIZE) + (x_block * BLOCK_SIZE);
}

__global__ void BlockMatMulKernel(Matrix A, Matrix B, Matrix C, int sh_A_offs, int sh_B_offs, int sh_C_offs)
{
    extern __shared__ float s[];
    float* sh_A = &s[sh_A_offs];
    float* sh_B = &s[sh_B_offs];
    float* sh_C = &s[sh_C_offs];
    int sh_idx = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    float *A_ptr, *B_ptr, *C_ptr;
    C_ptr = block_offset(C.elements, C.width, blockIdx.y, blockIdx.x);

    for(int i = 0; i < dev_ceil_div(A.width, BLOCK_SIZE); i++){
        // compute offset into A, B for this iteration
        A_ptr = block_offset(A.elements, A.width, blockIdx.y, i);
        B_ptr = block_offset(B.elements, B.width, i, blockIdx.x);

        // Load my element of A, B blocks
        sh_A[sh_idx] = A_ptr[threadIdx.y * A.width + threadIdx.x];
        sh_B[sh_idx] = B_ptr[threadIdx.y * B.width + threadIdx.x];
        __syncthreads();

        // Accumulate into my C block
        for(int k = 0; k < BLOCK_SIZE; k++) {
            sh_C[sh_idx] += sh_A[threadIdx.y * BLOCK_SIZE + k] * sh_B[k * BLOCK_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    // write my C block back
    C_ptr[threadIdx.y * C.width + threadIdx.x] = sh_C[sh_idx];

}

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
    int grid_x = ceil_div(B.width, BLOCK_SIZE);
    int grid_y = ceil_div(A.height, BLOCK_SIZE);
    dim3 dimGrid(grid_x, grid_y);
    int shared = 3 * dimBlock.x * dimBlock.y * sizeof(float);
    int sh_A_offs = 0;
    int sh_B_offs = dimBlock.x * dimBlock.y;
    int sh_C_offs = 2 * dimBlock.x * dimBlock.y;
    BlockMatMulKernel<<<dimGrid, dimBlock, shared>>>(d_A, d_B, d_C, sh_A_offs, sh_B_offs, sh_C_offs);
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
