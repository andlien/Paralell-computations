#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include "lodepng.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

// Kernel definition
__global__ void invertPicture(unsigned char *d_image, unsigned int size, unsigned int offset)
{
  unsigned int tid = threadIdx.x;
  unsigned int gtid = tid + blockDim.x*blockIdx.x;

  for (int i = gtid*offset; i < MIN(gtid*offset+offset, size); i++)
  {
    d_image[i] = ~d_image[i];
  }
}

unsigned int round_div(unsigned int dividend, unsigned int divisor)
{
  return (dividend + (divisor / 2)) / divisor;
}

int main(int argc, char ** argv)
{
  size_t pngsize;
  unsigned char *png;
  const char * filename = "lenna512x512_inv.png";

  /* Read in the image */
  lodepng_load_file(&png, &pngsize, filename);

  unsigned char *h_image, *d_image;
  unsigned int width, height;

  /* Decode it into a RGB 8-bit per channel vector */
  unsigned int error = lodepng_decode24(&h_image, &width, &height, png, pngsize);

  /* Check if read and decode of .png went well */
  if (error != 0)
  {
    std::cout << "error " << error << ": " << lodepng_error_text(error) << std::endl;
  }

  //float *h_x, *d_x; // h=host, d=device
  unsigned int n_blocks = 1<<5,
      n_threads_per_block = 1<<9, // 2^3 = 8
      n_size = n_blocks * n_threads_per_block,
      size = width*height*3,
      offset = round_div(size, n_size);

  cudaMalloc((void **) &d_image, size*sizeof(unsigned char));

  cudaMemcpy(d_image, h_image, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

  // Kernel invocation
  invertPicture<<<n_blocks, n_threads_per_block>>>(d_image, size, offset);
  cudaThreadSynchronize();  // Wait for invertPicture to finish on CUDA

  cudaMemcpy(h_image, d_image, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  /* Save the result to a new .png file */
  lodepng_encode24_file("lenna512x512_orig.png", h_image, width, height);

  // Cleanup
  cudaFree(d_image);
  free(h_image);

  return 0;
}
