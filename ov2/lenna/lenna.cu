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
  clock_t start, end, start_to, start_from, end_to, end_from;
  double total_rt, to_device_rt, from_device_rt;

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

  start = clock();

  //float *h_x, *d_x; // h=host, d=device
  unsigned int n_blocks = 1<<5,
      n_threads_per_block = 1<<9, // 2^3 = 8
      n_size = n_blocks * n_threads_per_block,
      size = width*height*3,
      offset = round_div(size, n_size);

  cudaMalloc((void **) &d_image, size*sizeof(unsigned char));

  start_to = clock();
  
  cudaMemcpy(d_image, h_image, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

  end_to = clock();

  // Kernel invocation
  invertPicture<<<n_blocks, n_threads_per_block>>>(d_image, size, offset);
  //cudaThreadSynchronize();  // Wait for invertPicture to finish on CUDA

  start_from = clock();

  cudaMemcpy(h_image, d_image, size*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  end_from = clock();
  end = clock();

  to_device_rt = (end_to-start_to)/(double)CLOCKS_PER_SEC;
  from_device_rt = (end_from-start_from)/(double)CLOCKS_PER_SEC;
  total_rt = (end-start)/(double)CLOCKS_PER_SEC;

  printf( "Transfer to device: %f\n",  to_device_rt);
  printf( "Transfer from device: %f\n", from_device_rt);
  printf( "Total program run-time: %f\n", total_rt);

  printf( "Percentage, to device: %f\n", to_device_rt/total_rt);
  printf( "Percentage, from device: %f\n", from_device_rt/total_rt);
  printf( "Percentage, total: %f\n", (to_device_rt + from_device_rt)/total_rt);

  /* Save the result to a new .png file */
  lodepng_encode24_file("lenna512x512_orig.png", h_image, width, height);

  // Cleanup
  cudaFree(d_image);
  free(h_image);

  return 0;
}
