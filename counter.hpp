#include <upcxx/upcxx.hpp>
#include <iostream>


static void count_off(upcxx::dist_object<uint64_t> & counts){

    //first, print myself off

    for (int i = 0; i  < upcxx::rank_n(); i++){

      if (upcxx::rank_me()  == i){

        std::cout << "P" <<  upcxx::rank_me()  <<  ":  " << * counts << std::endl;

      }
      upcxx::barrier();
    }

  }


static void ring_sum(upcxx::dist_object<uint64_t> & counts, uint64_t sum, size_t me){

    if (me==upcxx::rank_me()){

      //save temp
      uint64_t temp = sum + * counts;

      * counts = sum;

      //std::cout  <<  "Processor" << me <<  " sees previous sum " << sum << std::endl;

      if  (me != upcxx::rank_n()-1){
        upcxx::rpc_ff(me+1,ring_sum, counts, temp, me+1);
      }

    }


  }
