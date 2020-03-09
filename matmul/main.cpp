#include <functional>
#include <stdio.h>
#include "mkl.h"

#include "matrix.h"
#include "kernel_matmul.hpp"

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
    int mat_dim = 512;
	A.height = mat_dim;
	A.width = mat_dim;
	B.width = mat_dim;
	B.height = A.width;
	C.height = A.height;
	C.width = B.width;
	A.elements = (float *) mkl_malloc(A.width * A.height * sizeof(float), 64);
	B.elements = (float *) mkl_malloc(B.width * B.height * sizeof(float), 64);
	C.elements = (float *) mkl_malloc(C.width * C.height * sizeof(float), 64);
	init_mat(A, init_A);
	init_mat(B, init_B);
	init_mat(C, init_zeros);
    BlockMatMul(A, B, C);

    Matrix ref_C;
    ref_C.height = C.height;
    ref_C.width = C.width;
	ref_C.elements = (float *) mkl_malloc(C.width * C.height * sizeof(float), 64);
	init_mat(ref_C, init_zeros);
    ref_dgemm(A, B, ref_C);

    //print_mat(C);

    float tolerance = 0.0;
    float error = compare_mat(C, ref_C);
    if(error > tolerance)
    {
        printf("FAIL: error = %f\n", error);
    }

    //print_mat(ref_C);

	mkl_free(A.elements);
	mkl_free(B.elements);
	mkl_free(C.elements);
	mkl_free(ref_C.elements);
}
