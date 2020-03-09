#ifndef KERNEL_MATMUL_H
#define KERNEL_MATMUL_H

// Thread block size
#define BLOCK_SIZE 16
void MatMul(const Matrix A, const Matrix B, Matrix C);
void BlockMatMul(const Matrix A, const Matrix B, Matrix C);

#endif
