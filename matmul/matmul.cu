#include <functional>
#include <stdio.h>
#include "mkl.h"

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
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

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

auto init_A = [](int i, int j)->float { return i == j ? 1 : 0; };
auto init_B = [](int i, int j)->float { return j - i > 0 ? j - i : 0; };
auto init_zeros = [](int i, int j)->float { return 0; };
void init_mat(Matrix mat, std::function<float (int, int)> initializer)
{
	for(int i = 0; i < mat.height; i++)
	{
		for(int j = 0; j < mat.width; j++)
		{
			mat.elements[j + i * mat.width] = initializer(i, j);
		}
	}
}
void print_mat(Matrix M) {
	for(int i = 0; i < M.height; i++)
	{
		for(int j = 0; j < M.width; j++)
		{
			printf("%d ", (int)M.elements[j + i * M.width]);
		}
		printf("\n");
	}
}

void ref_dgemm(Matrix A, Matrix B, Matrix C)
{
    int m = A.height;
    int n = B.width;
    int k = A.width;
    float alpha = 1.0;
    float beta = 0.0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A.elements, k, B.elements, n, beta, C.elements, n);
}

float compare_mat(Matrix A, Matrix Ref)
{
    float error = 0.0;
    int idx;
    for(int i = 0; i < Ref.height; i++)
	{
		for(int j = 0; j < Ref.width; j++)
		{
            idx = j + i * Ref.width;
			error += Ref.elements[idx] - A.elements[idx];
		}
	}
    return error;
}

int main(int argc, char ** argv)
{
    Matrix A, B, C;
	A.height = 32;
	A.width = 32;
	B.width = 32;
	B.height = A.width;
	C.height = A.height;
	C.width = B.width;
	A.elements = (float *) mkl_malloc(A.width * A.height * sizeof(float), 64);
	B.elements = (float *) mkl_malloc(B.width * B.height * sizeof(float), 64);
	C.elements = (float *) mkl_malloc(C.width * C.height * sizeof(float), 64);
	init_mat(A, init_A);
	init_mat(B, init_B);
	init_mat(C, init_zeros);
    MatMul(A, B, C);

    Matrix ref_C;
    ref_C.height = C.height;
    ref_C.width = C.width;
	ref_C.elements = (float *) mkl_malloc(C.width * C.height * sizeof(float), 64);
	init_mat(ref_C, init_zeros);
    ref_dgemm(A, B, ref_C);

    print_mat(C);

    float tolerance = 0.0;
    float error = compare_mat(C, ref_C);
    if(error > tolerance)
    {
        printf("FAIL: error = %f\n", error);
    }

    // print_mat(ref_C);

	mkl_free(A.elements);
	mkl_free(B.elements);
	mkl_free(C.elements);
	mkl_free(ref_C.elements);
}
