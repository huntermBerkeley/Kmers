#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>
#include  <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <random>

#include "counter.hpp"

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

using namespace std;
//This is a test of integrating cuda into upcxx funtions
int main(int argc, char** argv) {
    upcxx::init();

    srand(upcxx::rank_me());


    if (upcxx::rank_me() == upcxx::rank_n()-1){

      std::cout << "Starting program with "  << upcxx::rank_n() << " processes." <<  std::endl;

    }

    //let each thread produce a 1024 mat
    std::size_t segsize = 4*1024*1024;

    //rank 0 can 'own' the gpu
    upcxx::cuda_device::id_type device_id = 0;
    upcxx::cuda_device gpu_device(device_id);

    upcxx::device_allocator<upcxx::cuda_device> gpu_alloc(gpu_device, segsize);
    upcxx::global_ptr<double, upcxx::memory_kind::cuda_device> gpu_array = gpu_alloc.allocate<double>(1024);

    upcxx::global_ptr<double> host_array1 = upcxx::new_array<double>(1024);
    upcxx::global_ptr<double> host_array2 = upcxx::new_array<double>(1024);
    double *h1 = host_array1.local();
    double *h2 = host_array2.local();

    if (upcxx::rank_me() == upcxx::rank_n()-1){

      std::cout << "device allocated." <<  std::endl;

    }
    upcxx::barrier();

    //and broadcast
    //global_ptr<double, upcxx::memory_kind::cuda_device> gpu_arr = upcxx::broadcast(gpu_array, 0).wait();
    //upcxx::global_ptr<double, upcxx::memory_kind::cuda_device> gpu_adjusted = gpu_array + upcxx::rank_me();

    //init h1
    for (int i =0; i < 1024; i++){
      h1[i] = i;
    }

    upcxx::copy(host_array1, gpu_array, 1024).wait();
    upcxx::copy(gpu_array, host_array2, 1024).wait();

    //generate data
    int nerrs = 0;
    for (int i = 0; i < 1024; i++){
      if (h1[i] != h2[i]){
        nerrs++;
      }
    }
    if (nerrs){
      cout << "Failure: " << nerrs << "detected" << endl;
    } else {
      cout <<"Success" << endl;
    }

    upcxx::delete_array(host_array1);
    upcxx::delete_array(host_array2);
    gpu_alloc.deallocate(gpu_array);
    gpu_device.destroy();


    upcxx::finalize();
    return 0;
}
