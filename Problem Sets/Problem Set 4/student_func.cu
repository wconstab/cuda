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

#define BLOCKSIZE 4

void printArrayIndices(unsigned int num){
  printf("IDX  \t");
  for(int i = 0; i < num; i++){
    printf("[%u]\t\t", i);
  }
  printf("\n");
}
void printCudaUnsignedIntArr(const char* name, unsigned int* const d_Vals, unsigned int numVals){
  unsigned int* h_vals = (unsigned int*)malloc(numVals*sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_vals, d_Vals, numVals*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  printf("%s\t", name);
  for(int i = 0; i < numVals; i++){
    if(h_vals[i] > 99999999)
      printf("%u\t", h_vals[i] );
    else
      printf("%u\t\t", h_vals[i] );
  }
  printf("\n");
}


__global__ void gpu_binary_predicate(
    unsigned int* const d_inputVals,
    unsigned int* const d_predicate,
    unsigned int bit,
    unsigned int desiredValue,
    const size_t numElems
)
{
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;
  if(gTid < numElems){
    d_predicate[gTid] = ((d_inputVals[gTid] >> bit) & 1) == desiredValue ? 1 : 0;
//    d_predicate[gTid] = gTid > 0;
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
    unsigned int scatterStart,
    const size_t numElems
)
{
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;
  if(gTid < numElems){
    if(d_predicate[gTid]){
      unsigned int addr = scatterStart + d_addr[gTid];
      if(addr < numElems){
        d_outputVals[addr] = d_inputVals[gTid];
      }//else printf("addr %u out of range\n", addr);
    }
  }
}

unsigned int sort_helper(unsigned int bit,
                 unsigned int predicateVal,
                 unsigned int* const d_inputVals,
                 unsigned int* const d_inputPos,
                 unsigned int* const d_outputVals,
                 unsigned int        scatterStart,
                 size_t              numElems)
{
  printf("Sort bit %d, pred %d, scatterStart %d\n", bit, predicateVal, scatterStart);
  unsigned int printLen = numElems;

  const dim3 blockSize(BLOCKSIZE, 1, 1);
  const dim3 gridSize( (numElems + blockSize.x - 1) / blockSize.x, 1, 1);

  unsigned int *d_predicate, *d_partialSums;
  checkCudaErrors(cudaMalloc(&d_predicate, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_partialSums, gridSize.x*sizeof(unsigned int)));

  // generate the predicate for specified bit, value 0, store in inputPos
  gpu_binary_predicate<<<gridSize, blockSize>>>(d_inputVals, d_predicate, bit, predicateVal, numElems);
  printCudaUnsignedIntArr("PRED ", d_predicate, printLen);

  checkCudaErrors(cudaMemcpy(d_inputPos + 1, d_predicate, (numElems - 1 )* sizeof(unsigned int), cudaMemcpyDeviceToDevice));

  // scan-sum the predicate values and generate scatter addresses
  gpu_exclusive_sum_scan<<<gridSize, blockSize>>>(d_inputPos, d_partialSums, numElems);
  gpu_exclusive_sum_scan_2<<<1, gridSize, gridSize.x*sizeof(unsigned int)>>>(d_partialSums, gridSize.x);
  gpu_exclusive_sum_scan_3<<<gridSize, blockSize>>>(d_inputPos, d_partialSums, numElems);
  printCudaUnsignedIntArr("ADDRS", d_inputPos, printLen);

  unsigned int highestScatterAddr = 0;
  checkCudaErrors(cudaMemcpy(&highestScatterAddr, d_inputPos + (numElems - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // scatter the input values to the correct places in the output values array
  gpu_scatter<<<gridSize, blockSize>>>(d_inputVals, d_predicate, d_inputPos, d_outputVals, scatterStart, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  printCudaUnsignedIntArr("OUT  ", d_outputVals, printLen);

  return highestScatterAddr;
}

void sort_wrapper(unsigned int* const d_inputVals,
                  unsigned int* const d_inputPos,
                  unsigned int* const d_outputVals,
                  unsigned int* const d_outputPos,
                  size_t              numElems)
{

  printArrayIndices(numElems);
  printCudaUnsignedIntArr("VAL  ", d_inputVals, numElems);

  for(unsigned int bit = 0; bit < sizeof(unsigned int) * 8; bit++)
  {
    if(bit & 1){
      // odd numbered bits copy intermediate results backward into the input buffer
      unsigned int highestZero = sort_helper(bit, 0, d_outputVals, d_outputPos, d_inputVals, 0, numElems);
      sort_helper(bit, 1, d_outputVals, d_outputPos, d_inputVals, highestZero+1, numElems);
    }else{
      unsigned int highestZero = sort_helper(bit, 0, d_inputVals, d_inputPos, d_outputVals, 0, numElems);
      sort_helper(bit, 1, d_inputVals, d_inputPos, d_outputVals, highestZero+1, numElems);
    }
  }

  // one final copy since we end on an odd numbered bit
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}


void debug2(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
                size_t numElems)
{

  unsigned int printLen = numElems;



  unsigned int bit = 0;
  unsigned int highestZero = sort_helper(bit, 0, d_inputVals, d_inputPos, d_outputVals, 0, numElems);
  printf("\nhighestZero %d\n", highestZero);
  sort_helper(bit, 1, d_inputVals, d_inputPos, d_outputVals, highestZero+1, numElems);
  printCudaUnsignedIntArr("SORT1", d_outputVals, printLen);

  // outputVals has correct results at this point

  bit = 1;
  highestZero = sort_helper(bit, 0, d_outputVals, d_outputPos, d_inputVals, 0, numElems);
  printCudaUnsignedIntArr("SORT0", d_inputVals, printLen);
  // inputVals has correct results for first 8 positions, and garbage for second 8 positions

  // TODO double check that outputVals isn't being modified when it's on the input side
  // and ... ?
  printf("\nhighestZero %d\n", highestZero);
  sort_helper(bit, 1, d_outputVals, d_outputPos, d_inputVals, highestZero+1, numElems);
  printCudaUnsignedIntArr("SORT1", d_inputVals, printLen);


//  sort_helper(bit, 1, d_inputVals, d_inputPos, d_outputVals, highestZero, numElems);
//  printCudaUnsignedIntArr("SORT1", d_outputVals, printLen);

}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
                size_t numElems)
{

  numElems = 16;
  unsigned int h_debugVals[numElems];
  for(int i = 0; i < numElems; i++) h_debugVals[i] = numElems-i-1;
  checkCudaErrors(cudaMemcpy(d_inputVals, h_debugVals, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));

//  debug2(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);

  sort_wrapper(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
}
