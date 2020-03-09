#ifndef KERNEL_MATMUL_H
#define KERNEL_MATMUL_H

void MatMul(const Matrix A, const Matrix B, Matrix C);
void BlockMatMul(const Matrix A, const Matrix B, Matrix C);

#endif
