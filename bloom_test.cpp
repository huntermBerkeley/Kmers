#include "bloom.hpp"
#include  <stdlib.h>
#include  <vector>
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {

    if (argc != 4) {

      throw std::runtime_error("Incorrect usage. Run with: \n ./bloom_test filter_size num_hashes num_samples.");

    }

    BloomFilter<uint64_t> filter = BloomFilter<uint64_t>( (uint64_t) atol (argv[1]), (uint8_t) atoi(argv[2]));

    //will this work?
    std::vector<uint64_t> vecs;

    auto started =std::chrono::high_resolution_clock::now();
    uint64_t num_samples = (uint64_t)  atoi(argv[3]);
    vecs.resize(num_samples);

    //load vector with random uint64_t

    for (size_t i =0; i < num_samples; i++){

      //fill each sample with a random number
      vecs[i] = (uint64_t) rand();

    }
    auto done =  std::chrono::high_resolution_clock::now();

    std::cout << "Vector built, Time elapsed: "  <<  std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count()
    << " milliseconds." << std::endl;

    //assert correctness of empty
    started = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i<num_samples; i++){
      if (filter.possiblyContains(&vecs[i])){
        throw std::runtime_error("Filter did not find added vec");
      }
    }

    done =  std::chrono::high_resolution_clock::now();

    std::cout << "Initial checks correct, Time elapsed: "  <<  std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count()
    << " milliseconds." << std::endl;

    //now  insert
    started = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i<num_samples; i++){
      filter.add(&vecs[i]);
    }

    done =  std::chrono::high_resolution_clock::now();

    std::cout << vecs.size() << " numbers added to filter, Time elapsed: "  <<  std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count()
    << " milliseconds." << std::endl;

    //and assert correctness
    started = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i<num_samples; i++){
      if (!filter.possiblyContains(&vecs[i])){
        throw std::runtime_error("Filter did not find added vec");
      }
    }

    done =  std::chrono::high_resolution_clock::now();

    std::cout << " Output correct, Time elapsed: "  <<  std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count()
    << " milliseconds." << std::endl;

    return 0;
}
