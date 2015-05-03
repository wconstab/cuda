/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#define N 1024

__global__
void simpleHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < numVals){
    atomicAdd(&histo[vals[tid]], 1);
  }
}

__global__
void smemHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  __shared__ unsigned int sHisto[N];
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < numVals){
    unsigned int bin = vals[tid];
//    printf("tid %d bin %d\n", tid, bin);

    atomicAdd(&sHisto[bin], 1);
  }
//  __syncthreads();
//  atomicAdd(&histo[threadIdx.x], sHisto[threadIdx.x]);
}

/**
 * Shared Memory method 1
 *   since N = 1024 and numBins also equals 1024, could use shared memory to compute entire local histograms
 *   could then write them back to memory in different blocks, after which a second kernel could sum them up
 *
 *   - tried this with atomics to write back instead of writing back like described above, got average 5ms slower than naive simple soln
 */

/**
 * Sorting method:
 *  Each block computes a histogram of a subset of bin ids.  since N = 1024, numVals = 10000*1024, numBlocks = 10000
 *  Each block could work on 0.1 of a real bin id...?
 *
 *    OR
 *  Only have enough threads/blocks to cover the bins.  Perhaps 1024 blocks, one block per bin?  Then, threads per block don't evenly distribute over vals?
 *
 */

/** REFERENCE
 *   //zero out bins
  for (size_t i = 0; i < numBins; ++i)
    histo[i] = 0;

  //go through vals and increment appropriate bin
  for (size_t i = 0; i < numElems; ++i)
    histo[vals[i]]++;
 */
void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  simpleHisto<<<(numElems+N)/N, N>>>(d_vals, d_histo, numElems);
//  smemHisto<<<(numElems+N)/N, N>>>(d_vals, d_histo, numElems);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
