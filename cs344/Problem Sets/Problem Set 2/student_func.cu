// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "utils.h"
#include <stdio.h>

#define BLOCK_SIZE 512
#define DEBUG_PIX 4235
__device__
int clamp(int imageIdx, int imageDim, int filterIdx, int filterDim){
  int filterOffset = filterIdx - filterDim/2;
  int clampedR = imageIdx + filterOffset;
  clampedR = (clampedR >= imageDim ? imageDim - 1 : clampedR);
  clampedR = (clampedR < 0 ? 0 : clampedR);
  return clampedR;
}

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  // compute thread index, range check
  const int imageIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(imageIdx >= numRows * numCols) return;
  const int imageR = imageIdx / numCols;
  const int imageC = imageIdx % numCols;

  // read neighboring pixels and apply filter value
  float accum = 0.0;
  for(int fR = 0; fR < filterWidth; fR++){
    for(int fC = 0; fC < filterWidth; fC++){
      float filterVal = filter[fR * filterWidth + fC];
      int neighborR = clamp(imageR, numRows, fR, filterWidth);
      int neighborC = clamp(imageC, numCols, fC, filterWidth);
      float thisVal = static_cast<float>(inputChannel[(neighborR * numCols) + neighborC]);
      accum += filterVal * thisVal;
      if(imageIdx == DEBUG_PIX){
        printf("fR %d fC %d filterVal %llf neighborR %d neighborC %d thisVal %f accum %f\n",
            fR, fC, filterVal, neighborR, neighborC, thisVal, accum);
      }
    }
  }

  // write result to output channel
  outputChannel[imageIdx] = accum > 255 ? 255 : (unsigned char)accum;
//  if(imageIdx == DEBUG_PIX){
//    printf("output %d\n",
//        outputChannel[imageIdx]);
//  }
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  // compute thread index, range check
  const int imageIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(imageIdx >= numRows * numCols) return;

  // seperate color channels
  uchar4 rgbaPixel = inputImageRGBA[imageIdx];
  redChannel[imageIdx] = rgbaPixel.x;
  greenChannel[imageIdx] = rgbaPixel.y;
  blueChannel[imageIdx] = rgbaPixel.z;
  if(imageIdx == DEBUG_PIX){
    printf("extracted RGB %d %d %d ", redChannel[imageIdx], greenChannel[imageIdx], blueChannel[imageIdx]);
  }
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int imageIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if(imageIdx >= numRows * numCols) return;


  unsigned char red   = redChannel[imageIdx];
  unsigned char green = greenChannel[imageIdx];
  unsigned char blue  = blueChannel[imageIdx];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[imageIdx] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter,  sizeof(float) * filterWidth * filterWidth));

  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  const dim3 blockSize(BLOCK_SIZE, 1, 1);

  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const size_t numPixels = numRows * numCols;
  const dim3 gridSize((numPixels / blockSize.x) + 1 , 1, 1);
  printf("Using blockSize %d, gridSize %d\n", blockSize.x, gridSize.x);
  printf(" nRows = %d, /cols %d\n", numRows, numCols);

  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
										    numRows,
											numCols,
											d_redBlurred,
											d_greenBlurred,
											d_blueBlurred);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  gaussian_blur<<<gridSize, blockSize>>>(d_redBlurred,
		                                 d_redBlurred,
                                         numRows, numCols,
                                         d_filter, filterWidth);

//  gaussian_blur<<<gridSize, blockSize>>>(d_greenBlurred,
//		                                 d_greenBlurred,
//                                         numRows, numCols,
//                                         d_filter, filterWidth);
//
//  gaussian_blur<<<gridSize, blockSize>>>(d_blueBlurred,
//		                                 d_blueBlurred,
//                                         numRows, numCols,
//                                         d_filter, filterWidth);


  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  //
  // NOTE: This kernel launch depends on the gridSize and blockSize variables,
  // which you must set yourself.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
