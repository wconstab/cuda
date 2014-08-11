// for uchar4 struct
#include <cuda_runtime.h>
#include <stdio.h>
void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols)
{
  for (size_t r = 0; r < numRows; ++r) {
    for (size_t c = 0; c < numCols; ++c) {

      int index = r * numCols + c;


      uchar4 rgba = rgbaImage[index];

      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[index] = channelSum;

      if(index == 1514){
    	  printf("ref: %d %d %d, %f %d\n", rgba.x, rgba.y, rgba.z, channelSum, greyImage[index]);
      }
    }
  }
}
