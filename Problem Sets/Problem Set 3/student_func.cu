/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#define MIN(X,Y) (X < Y ? X : Y)
#define MAX(X,Y) (X > Y ? X : Y)

#define BLOCKSIZE 512
__global__ void min_max_luminance(
    const float* d_logLuminance,
    float* d_temp,
    float* d_min_lum,
    float* d_max_lum,
    int    numRows,
    int    numCols)
{
  int bTid = threadIdx.x;
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;
  __shared__ float s_max[BLOCKSIZE];
  __shared__ float s_min[BLOCKSIZE];

  // copy global value into shared memory
  if(bTid < BLOCKSIZE) s_max[bTid] = d_logLuminance[gTid];
  if(bTid < BLOCKSIZE) s_min[bTid] = d_logLuminance[gTid];
  __syncthreads();

  for(int s = BLOCKSIZE/2; s > 0; s/=2){
    if(bTid < s){
      s_max[bTid] = MAX(s_max[bTid], s_max[bTid + s]);
      s_min[bTid] = MIN(s_min[bTid], s_min[bTid + s]);
    }
    __syncthreads();
  }

  if(bTid == 0) d_temp[blockIdx.x] = s_max[bTid];
  if(bTid == 0) d_temp[blockDim.x + blockIdx.x] = s_min[bTid];
}

__global__ void min_max_luminance_2(
    const float* d_logLuminance,
    float* d_temp,
    float* d_min_lum,
    float* d_max_lum,
    int    numRows,
    int    numCols)
{
  int bTid = threadIdx.x;

  // copy the partial results from global to shared memory
  __shared__ float s_max[BLOCKSIZE];
  __shared__ float s_min[BLOCKSIZE];
  s_max[bTid] = d_temp[bTid];
  s_min[bTid] = d_temp[blockDim.x + bTid];
  __syncthreads();

  for(int s = BLOCKSIZE/2; s > 0; s/=2){
    if(bTid < s){
      s_max[bTid] = MAX(s_max[bTid], s_max[bTid + s]);
      s_min[bTid] = MIN(s_min[bTid], s_min[bTid + s]);
    }
    __syncthreads();
  }

  if(bTid == 0){
    *d_min_lum = s_min[bTid];
    *d_max_lum = s_max[bTid];
//    printf("gTid %d global min %f max %f\n", bTid, *d_min_lum, *d_max_lum);
  }

}

__global__ void bins(
    const float* d_logLuminance,
    unsigned int* d_bins,
    float d_min_lum,
    float d_range,
    unsigned int    numBins)
{
//  int bTid = threadIdx.x;
  int gTid = (blockDim.x *blockIdx.x) + threadIdx.x;

  extern __shared__ unsigned int s_bins[];

  if(gTid == 0) printf("d_min_lum %f, d_range %f, numBins %d\n", d_min_lum, d_range, numBins);
  unsigned int bin = (d_logLuminance[gTid] - d_min_lum) / d_range * numBins;
  if(bin >= numBins) bin = numBins - 1;
//  unsigned int bin = gTid / numBins;
  atomicAdd(&d_bins[bin], 1);

}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // 1 min/max luminance
  float * d_min_lum, *d_max_lum, *d_temp;
  checkCudaErrors(cudaMalloc(&d_min_lum,    (size_t)sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max_lum,    (size_t)sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_temp,    (size_t)(sizeof(float)*numRows*numCols)));

  const dim3 blockSize(512, 1, 1);
  const dim3 gridSize( (numCols*numRows + blockSize.x - 1) / blockSize.x, 1, 1);
  min_max_luminance<<<gridSize, blockSize, blockSize.x>>>(d_logLuminance, d_temp, d_min_lum, d_max_lum,
                                             numRows, numCols);
  min_max_luminance_2<<<1, blockSize, blockSize.x>>>(d_logLuminance, d_temp, d_min_lum, d_max_lum,
                                             numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//  float h_min_lum=0, h_max_lum=0;
  checkCudaErrors(cudaMemcpy(&min_logLum, d_min_lum, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max_lum, sizeof(float), cudaMemcpyDeviceToHost));

  // 2 range
  float h_range = max_logLum - min_logLum;
  printf("max %f, min %f, range %f\n", max_logLum, min_logLum, h_range);

  // 3 histogram
  float* d_range;
  unsigned int* d_bins;
  checkCudaErrors(cudaMalloc(&d_range, (size_t)sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_range, &h_range, sizeof(float), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_bins, (size_t)numBins*sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_bins, 0, (size_t)numBins*sizeof(unsigned int)));

  bins<<<gridSize, blockSize, numBins*sizeof(unsigned int)>>>(d_logLuminance, d_bins, min_logLum, h_range, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int channelSize = numRows * numCols;
  float* h_logLuminance = (float*)malloc(channelSize * sizeof(float));
  checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, channelSize*sizeof(float), cudaMemcpyDeviceToHost));
  unsigned int* h_bins = (unsigned int*)malloc(numBins * sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_bins, d_bins, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));


//  printf("\nIDX  \t");
//  for(int i = 0; i < numBins; i++){
//    printf("[%u]\t", i);
//  }
//  printf("\nHIST \t");
//  for(int i = 0; i < numBins; i++){
//    printf("%u\t", h_bins[i]);
//  }

  // 4
  unsigned int* h_cdf = (unsigned int*)malloc(numBins * sizeof(unsigned int));
  h_cdf[0] = 0;
  for(int i = 1; i < numBins; i++){
    h_cdf[i] = h_cdf[i-1] + h_bins[i-1];
  }

//  printf("\nCDF  \t");
//  for(int i = 0; i < numBins; i++){
//    printf("%d\t", h_cdf[i]);
//  }
  checkCudaErrors(cudaMemcpy(d_cdf,   h_cdf,   numBins*sizeof(unsigned int), cudaMemcpyHostToDevice));

  free(h_bins);
  free(h_cdf);
  printf("\nDone.\n");
}
