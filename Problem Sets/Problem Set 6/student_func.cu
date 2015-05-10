//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */
#define NBUF 2
#define NCHAN 3
#define ch_R 0
#define ch_G 1
#define ch_B 2

#define NON_WHITE x
#define INTERIOR y
#define BORDER z

#include "utils.h"
#include <thrust/host_vector.h>

__global__ void mask_kern(uchar4* sourceImg, uchar4* masks, size_t numRowsSource, size_t numColsSource)
{

  int  gTid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
  size_t numElem = numRowsSource * numColsSource;

  if(gTid < numElem){
    masks[gTid].NON_WHITE = sourceImg[gTid].x != 255 || sourceImg[gTid].y != 255 || sourceImg[gTid].z != 255;
  }
}
__global__ void mask_kern2(uchar4* sourceImg, uchar4* masks, int numRowsSource, int numColsSource)
{

  int  gTid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
  size_t numElem = numRowsSource * numColsSource;

  if(gTid < numElem){
    masks[gTid].INTERIOR = masks[gTid].NON_WHITE;
    if(gTid - 1 >= 0 && masks[gTid-1].NON_WHITE == 0)  masks[gTid].INTERIOR = 0;
    if(gTid + 1 < numElem && masks[gTid+1].NON_WHITE == 0)  masks[gTid].INTERIOR = 0;
    if(gTid - numColsSource >= 0 && masks[gTid-numColsSource].NON_WHITE == 0)  masks[gTid].INTERIOR = 0;
    if(gTid + numColsSource < numElem && masks[gTid+numColsSource].NON_WHITE == 0)  masks[gTid].INTERIOR = 0;

    masks[gTid].BORDER = masks[gTid].NON_WHITE && !masks[gTid].INTERIOR;

    // debug, visually
//    masks[gTid].INTERIOR *= 255;
//    masks[gTid].NON_WHITE *= 255;
//    masks[gTid].BORDER *= 255;
  }
}

__global__ void extract_rgb_kern(uchar4* sourceImg, float* r, float* g, float* b, int numRowsSource, int numColsSource)
{

  int  gTid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
  size_t numElem = numRowsSource * numColsSource;

  if(gTid < numElem){
    r[gTid] = (float)sourceImg[gTid].x;
    g[gTid] = (float)sourceImg[gTid].y;
    b[gTid] = (float)sourceImg[gTid].z;
  }
}
void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement

     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

        */
  size_t numElem = numRowsSource * numColsSource;
  uchar4* d_sourceImg;
  checkCudaErrors(cudaMalloc(&d_sourceImg,  sizeof(uchar4) * numElem));
  cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * numElem, cudaMemcpyHostToDevice);

  uchar4* d_masks;
  checkCudaErrors(cudaMalloc(&d_masks,  sizeof(uchar4) * numElem));
  checkCudaErrors(cudaMemset(d_masks, 0, sizeof(uchar4) * numElem));

  const dim3 blockSize(256, 1, 1);
  const dim3 gridSize( (numElem + 1) / blockSize.x, 1, 1);
  mask_kern<<<gridSize, blockSize>>>(d_sourceImg, d_masks, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  mask_kern2<<<gridSize, blockSize>>>(d_sourceImg, d_masks, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


  // debug
//  cudaMemcpy(h_blendedImg, d_masks, sizeof(uchar4) * numElem, cudaMemcpyDeviceToHost);

  /*
        > alloc 3 (r,g,b) device float bufs x2 for guesses
        > device extract r,g,b into (both?) bufs
        > iterate jacobi func 800 times
        > replace dest img interior pixels with rsult of jacobi
          -- cast fp vals to uchar since they have been clamped already (in jacobi?)


     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

*/

  float* d_jacBuf[NBUF][NCHAN];
  for(int n = 0; n < NBUF; n++){
    for(int c = 0; c < NCHAN; c++){
      checkCudaErrors(cudaMalloc(&d_jacBuf[n][c],  sizeof(float) * numElem));
    }
  }

  extract_rgb_kern<<<gridSize, blockSize>>>(d_sourceImg, d_jacBuf[0][ch_R], d_jacBuf[0][ch_G], d_jacBuf[0][ch_B], numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  extract_rgb_kern<<<gridSize, blockSize>>>(d_sourceImg, d_jacBuf[1][ch_R], d_jacBuf[1][ch_G], d_jacBuf[1][ch_B], numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /*
     5) For each color channel perform the Jacobi iteration described
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
}
