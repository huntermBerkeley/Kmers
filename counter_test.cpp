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

#include "counter.hpp"


int main(int argc, char** argv) {
    upcxx::init();

    srand(upcxx::rank_me());

    // TODO: Dear Students,
    // Please remove this if statement, when you start writing your parallel implementation.
    // if (upcxx::rank_n() > 1) {
    //     throw std::runtime_error("Error: parallel implementation not started yet!"
    //                              " (remove this when you start working.)");
    // }

    if (upcxx::rank_me() == upcxx::rank_n()-1){

      std::cout << "Starting program"  <<  std::endl;

    }



    //fill counter with rand inserts
    //Counter  counter = Counter( );
    upcxx::dist_object<uint64_t> counter = (uint64_t) rand() %  10;

    // p false positive rate

    //involve all processors in hashmap
    //DistrMap hashmap();
    count_off(counter);

    upcxx::barrier();

    ring_sum(counter,0,0);

    upcxx::barrier();

    if (upcxx::rank_me() == upcxx::rank_n()-1){

      std::cout << "After:"  <<  std::endl;

    }


    count_off(counter);

    upcxx::barrier();

    if (upcxx::rank_me() == upcxx::rank_n()-1){

      std::cout << "Finir:"  <<  std::endl;

    }

    upcxx::barrier();


    upcxx::finalize();
    return 0;
}
