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

#define BLOCKSIZE 1024
#define DEBUG true
#define NUMELEMS 3000

// blocksize 1024, numElems 3000
//idx 2575 sum 2788;
//idx 2576 sum 2796

// bit 0 pred 0
//PSums 511   512   476
//Sums  511   1023    1499
// bit 0 pred 1
//PSums 512   512   484
//Sums  512   1024    1508
// why 2575?

// psum for pred 1 wrong.
//block 2, bTid 951, produced psum 484
//block 2, bTid 951, produced psum 476


void printArrayIndices(unsigned int num){
  if(DEBUG){
    printf("IDX  \t");
    for(int i = 0; i < num; i++){
      printf("[%u]\t\t", i);
    }
    printf("\n");
  }
}
void printCudaUnsignedIntArr(const char* name, unsigned int* const d_Vals, unsigned int numVals){
  if(DEBUG){
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
}

void checkSorted(unsigned int* const d_inputVals, size_t numElems){
  unsigned int* h_vals = (unsigned int*)malloc(numElems*sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_vals, d_inputVals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));

  unsigned int last = h_vals[0];
  for(unsigned int i = 1; i < numElems; i++){
    if(h_vals[i] < last){
      printf("sort error: [%u]=%u, [%u]=%u\n", i-1, last, i, h_vals[i]);
      return;
    }
  }
  printf("sort check PASS\n");
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


//__global__ void gpu_exclusive_sum_scan(
//    unsigned int* const d_predicate,
//    unsigned int* const d_inputVals,
//    unsigned int* d_partialSums,
//    const size_t numElems
//)
//{
//  __shared__ unsigned int s_temp[BLOCKSIZE];
//
//  int bTid = threadIdx.x;
//  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;
//  int thisBlockNumElem = BLOCKSIZE;
//  if(blockIdx.x == numElems/BLOCKSIZE && numElems%BLOCKSIZE != 0){
//    thisBlockNumElem = numElems%BLOCKSIZE;
//  }
//
//  // take the predicate values and sum-scan them
//  // - but it doesn't get finished in one function, because of synchronization between threads.
//  // need to divide into sectors.
//
//
//}

// whawt is special about 2976?
// cant seem to find any reason why the problem starts happening there. seems like addition is working fine for the first 2976 elements.
// then there is some corruption for the remaining ones.


__global__ void gpu_exclusive_sum_scan(
    unsigned int* const d_predicate,
    unsigned int* const d_inputVals,
    unsigned int* d_partialSums,
    const size_t numElems
)
{
  __shared__ unsigned int s_temp[BLOCKSIZE];

  int bTid = threadIdx.x;
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;
  int thisBlockNumElem = BLOCKSIZE;
  if(blockIdx.x == numElems/BLOCKSIZE && numElems%BLOCKSIZE != 0){
    thisBlockNumElem = numElems%BLOCKSIZE;
  }

  if(bTid == 0) printf("block %u numElem %u\n", blockIdx.x, thisBlockNumElem);
  s_temp[bTid] = 0;
  if(gTid < numElems){
    s_temp[bTid] = d_inputVals[gTid];
    __syncthreads();

    int s;
    for(s = 1; s < BLOCKSIZE; s*=2){
      if(bTid + s < thisBlockNumElem){
        atomicAdd(&s_temp[bTid+s], s_temp[bTid]);
      }
      __syncthreads();
    }
    d_inputVals[gTid] = s_temp[bTid];
    __syncthreads();

    if(bTid == thisBlockNumElem-1){
      d_partialSums[blockIdx.x] = s_temp[thisBlockNumElem-1];
      printf("block %u, bTid %u, produced psum %u\n", blockIdx.x, bTid, d_partialSums[blockIdx.x]);
    }
  }
}

__global__ void gpu_exclusive_sum_scan_2(
    unsigned int* d_partialSums,
    unsigned int numSums
)
{
  extern __shared__ unsigned int s_temp[];
  int bTid = threadIdx.x;

  if(bTid < numSums)
  {
    s_temp[bTid] = d_partialSums[bTid];
    __syncthreads();
    int s;
    for(s = 1; s < numSums; s*=2){
      if(bTid + s < numSums){
         atomicAdd(&s_temp[bTid+s], s_temp[bTid]);
      }
      __syncthreads();
    }
    d_partialSums[bTid] = s_temp[bTid];
//    if(d_partialSums[bTid] != ((BLOCKSIZE/2) * (bTid+1))-1) printf("bTid %d produced Sum %d\n", bTid, d_partialSums[bTid]);
  }
}

__global__ void gpu_exclusive_sum_scan_3(
    unsigned int* const d_inputVals,
    unsigned int* d_partialSums,
    unsigned int scatterStart,
    const size_t numElems
)
{

  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;

  d_inputVals[gTid] += scatterStart;
  if(blockIdx.x > 0 && gTid < numElems){
    d_inputVals[gTid] += d_partialSums[blockIdx.x - 1];
//    if(d_inputVals[gTid] != gTid) printf("gTid %d just produced addr %u, using pSum %u, sstart %u\n", gTid, d_inputVals[gTid], d_partialSums[blockIdx.x -1], scatterStart);
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
      }else printf("addr %u out of range\n", addr);
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
  if(DEBUG) printf("Sort bit %d, pred %d, scatterStart %d\n", bit, predicateVal, scatterStart);
  printArrayIndices(numElems);

  unsigned int printLen = numElems;

  const dim3 blockSize(BLOCKSIZE, 1, 1);
  const dim3 gridSize( (numElems + blockSize.x - 1) / blockSize.x, 1, 1);

  unsigned int *d_predicate, *d_partialSums;
  checkCudaErrors(cudaMalloc(&d_predicate, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_partialSums, gridSize.x*sizeof(unsigned int)));

  // generate the predicate for specified bit
  gpu_binary_predicate<<<gridSize, blockSize>>>(d_inputVals, d_predicate, bit, predicateVal, numElems);
  printCudaUnsignedIntArr("PRED ", d_predicate, printLen);

  checkCudaErrors(cudaMemcpy(d_inputPos+1, d_predicate, (numElems-1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  unsigned int zero = 0;
  checkCudaErrors(cudaMemcpy(d_inputPos , &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

  // scan-sum the predicate values and generate scatter addresses
  gpu_exclusive_sum_scan<<<gridSize, blockSize>>>(d_predicate, d_inputPos, d_partialSums, numElems);
  printCudaUnsignedIntArr("iPsTm", d_inputPos, numElems);
  printCudaUnsignedIntArr("PSums", d_partialSums, gridSize.x);


  gpu_exclusive_sum_scan_2<<<1, gridSize, gridSize.x*sizeof(unsigned int)>>>(d_partialSums, gridSize.x);
  printCudaUnsignedIntArr("Sums", d_partialSums, gridSize.x);

  gpu_exclusive_sum_scan_3<<<gridSize, blockSize>>>(d_inputPos, d_partialSums, scatterStart, numElems);
  printCudaUnsignedIntArr("ADDRS", d_inputPos, printLen);

  unsigned int highestScatterAddr = 0;
  checkCudaErrors(cudaMemcpy(&highestScatterAddr, d_inputPos + (numElems - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // scatter the input values to the correct places in the output values array
  gpu_scatter<<<gridSize, blockSize>>>(d_inputVals, d_predicate, d_inputPos, d_outputVals, numElems);
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

//  unsigned int bit = 0;
//  for(unsigned int bit = 0; bit < sizeof(unsigned int) * 8; bit++)
  for(unsigned int bit = 0; bit < 1; bit++)

  {
    if(bit & 1){
      // odd numbered bits copy intermediate results backward into the input buffer
      unsigned int highestZero = sort_helper(bit, 0, d_outputVals, d_outputPos, d_inputVals, 0, numElems);
      if(highestZero < numElems - 1)
        sort_helper(bit, 1, d_outputVals, d_outputPos, d_inputVals, highestZero+1, numElems);
    }else{
      unsigned int highestZero = sort_helper(bit, 0, d_inputVals, d_inputPos, d_outputVals, 0, numElems);
      if(highestZero < numElems - 1)
        sort_helper(bit, 1, d_inputVals, d_inputPos, d_outputVals, highestZero+1, numElems);
    }
  }

  // one final copy since we end on an odd numbered bit
  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
                size_t numElems)
{

  numElems = NUMELEMS;
  unsigned int h_debugVals[numElems];
  for(int i = 0; i < numElems; i++) h_debugVals[i] = numElems-i-1;
  checkCudaErrors(cudaMemcpy(d_inputVals, h_debugVals, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));


  sort_wrapper(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);
  checkSorted(d_outputVals, numElems);
}
