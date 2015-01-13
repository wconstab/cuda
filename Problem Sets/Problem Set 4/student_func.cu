//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#define BLOCKSIZE 512
__global__ void gpu_binary_predicate(
    unsigned int* const d_inputVals,
    unsigned int* const d_predicate,
    unsigned int bit,
    const size_t numElems
)
{
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;
  if(gTid < numElems){
//    d_predicate[gTid] = (d_inputVals[gTid] & (1 << bit)) == 0 ? 1 : 0;
    d_predicate[gTid] = gTid > 0;
  }
}

__global__ void gpu_exclusive_sum_scan(
    unsigned int* const d_inputVals,
    unsigned int* d_partialSums,
    const size_t numElems
)
{
  __shared__ unsigned int s_temp[BLOCKSIZE];

  int bTid = threadIdx.x;
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;

  if(gTid < numElems){
    s_temp[bTid] = d_inputVals[gTid];
    __syncthreads();

    int s;
    for(s = 1; s < BLOCKSIZE; s*=2){
      if(bTid + s < BLOCKSIZE){
        atomicAdd(&s_temp[bTid+s], s_temp[bTid]);
      }
      __syncthreads();
    }
    d_inputVals[gTid] = s_temp[bTid];

    if(bTid == 0) d_partialSums[blockIdx.x] = s_temp[BLOCKSIZE-1];
  }
}

__global__ void gpu_exclusive_sum_scan_2(
    unsigned int* d_partialSums,
    unsigned int numSums
)
{
  extern __shared__ unsigned int s_temp[];
  int bTid = threadIdx.x;

  if(bTid < numSums){
    s_temp[bTid] = d_partialSums[bTid];
    __syncthreads();
    int s;
    for(s = 1; s < numSums; s*=2){
      if(bTid + s < numSums){
        s_temp[bTid+s] += s_temp[bTid];
      }
      __syncthreads();
    }
    d_partialSums[bTid] = s_temp[bTid];
  }
}

__global__ void gpu_exclusive_sum_scan_3(
    unsigned int* const d_inputVals,
    unsigned int* d_partialSums,
    const size_t numElems
)
{

  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;

  if(blockIdx.x > 0 && gTid < numElems){
    d_inputVals[gTid] += d_partialSums[blockIdx.x - 1];
  }
}

__global__ void gpu_scatter(
    unsigned int* const d_inputVals,
    unsigned int* const d_predicate,
    unsigned int* const d_addr,
    unsigned int* const d_outputVals,
    const size_t numElems
)
{
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;
  if(gTid < numElems){
    if(d_predicate[gTid]){
      unsigned int addr = d_addr[gTid];
      if(addr < numElems){
        d_outputVals[addr] = d_inputVals[gTid];
      }//else printf("addr %u out of range\n", addr);
    }
  }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
                size_t numElems)
{
  const dim3 blockSize(BLOCKSIZE, 1, 1);
  const dim3 gridSize( (numElems + blockSize.x - 1) / blockSize.x, 1, 1);
  unsigned int *d_predicate;
  checkCudaErrors(cudaMalloc(&d_predicate, (size_t)(numElems * sizeof(unsigned int))));
  gpu_binary_predicate<<<gridSize, blockSize>>>(d_inputVals, d_predicate, 0, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(d_inputPos, d_predicate, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));


  unsigned int* h_inputVals = (unsigned int*)malloc(numElems * sizeof(unsigned int));
  unsigned int* h_predicate = (unsigned int*)malloc(numElems * sizeof(unsigned int));
  unsigned int* h_inputPos = (unsigned int*)malloc(numElems * sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_predicate, d_predicate, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));


  size_t printLen = 4096;
  printf("\nIDX  \t");
  for(int i = 0; i < printLen; i++){
    printf("[%u]\t", i);
  }
  printf("\nVAL  \t");
  for(int i = 0; i < printLen; i++){
    printf("%2x\t", h_inputVals[i] & 0xff);
  }
  printf("\nPRED \t");
  for(int i = 0; i < printLen; i++){
    printf("%u\t", h_predicate[i]);
  }
  unsigned int* d_partialSums;
  checkCudaErrors(cudaMalloc(&d_partialSums, gridSize.x*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_outputPos, 0, numElems*sizeof(unsigned int)));
  gpu_exclusive_sum_scan<<<gridSize, blockSize>>>(d_inputPos, d_partialSums, numElems);
  gpu_exclusive_sum_scan_2<<<1, gridSize, gridSize.x*sizeof(unsigned int)>>>(d_partialSums, gridSize.x);
  gpu_exclusive_sum_scan_3<<<gridSize, blockSize>>>(d_inputPos, d_partialSums, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_inputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  printf("\nADDR \t");
  for(int i = 0; i < printLen; i++){
    printf("%d\t", h_inputPos[i]);
  }

  gpu_scatter<<<gridSize, blockSize>>>(d_inputVals, d_predicate, d_inputPos, d_outputVals, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  unsigned int* h_outputVals = (unsigned int*)malloc(numElems * sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_outputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  printf("\nOUT  \t");
  for(int i = 0; i < printLen; i++){
    printf("%2x\t", h_outputVals[i] & 0xff);
  }

  unsigned int* h_partialSums = (unsigned int*)malloc(gridSize.x*sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_partialSums, d_partialSums, gridSize.x*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  printf("\nPSUM \t");
  for(int i = 0; i < gridSize.x; i++){
    printf("%u\t", h_partialSums[i] );
  }
}
