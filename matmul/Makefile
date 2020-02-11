run: matmul
	LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH ./matmul
memcheck: matmul
	LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH  cuda-memcheck ./matmul
dbg: matmul
	LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH  cuda-gdb ./matmul
matmul: matmul.cu
	nvcc -L /opt/intel/mkl/lib/intel64 -lmkl_core -lmkl_rt -g -std=c++11 -I /opt/intel/mkl/include/ matmul.cu -o matmul
