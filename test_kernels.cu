
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

//Init cuda here

// __device__ void VecTest(float * A, float* B, size_t n){
//
//   size_t tid = threadIdx.x;
//   if (tid < n){
//     B[N]  = A[N];
//   }
//
//   __syncthreads();
//   return;
// }


__global__ void copy_kernel(double * to_copy, double* items, size_t n) {
  size_t tid = threadIdx.x;



  if (tid < n) {
    to_copy[tid] = items[tid];
  }
}

void copy_wrapper(double * to_copy, double* items, size_t n){

  copy_kernel<<<1,n>>>(to_copy, items, n);

}
