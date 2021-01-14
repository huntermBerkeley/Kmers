#include <upcxx/upcxx.hpp>
#include <algorithm>
#include "pkmer_t.hpp"
#include <iostream>
#include  "bloom.hpp"

class BloomDist
{


private:

    //For the first pass, the system will have two bloom filters
    using shared_filter = upcxx::dist_object<BloomFilter<uint64_t>>;
    using count_hashmap = upcxx::dist_object<std::unordered_map<uint64_t, uint64_t>>;

    shared_filter bloomFirst;
    shared_filter bloomSecond;
    count_hashmap count_map;

public:

  //construct both filters and hashmap
  BloomDist(uint64_t size, uint8_t numHashes): bloomFirst(BloomFilter<uint64_t>(size, numHashes)), bloomSecond(BloomFilter<uint64_t>(size, numHashes)), count_map({}) {}


  //For the first pass, we check if exists in filter 1
  //  if yes, add to filter 2
  // if no, add to filter 1
  //this call does not need an rpc return as we are only building the filters
  void first_pass(const pkmer_t &  key){

    upcxx::rpc_ff(key.hash() % upcxx::rank_n(),

    //lambda to locate correct filter - we want these to go to correct hashmap
    [](shared_filter & bloomFirst, shared_filter & bloomSecond, uint64_t key){

      if (bloomFirst->possiblyContains(&key)){

        //std::cout << "Processor " <<  upcxx::rank_me() << " / " <<  upcxx::rank_n()  <<  " inserted!" << std::endl;
        bloomSecond->add(&key);

      } else {

        bloomFirst->add(&key);

      }
    },

    bloomFirst, bloomSecond, key.hash());

    return;

  }

  //on the second pass, use the second filter to selectively add to hashmap
  //and increment it's counter if not already set

  void second_pass(const pkmer_t & key){

    //like first pass, don't need response
    upcxx::rpc_ff(key.hash() % upcxx::rank_n(),

    [](shared_filter & filter,  count_hashmap & map,  uint64_t hash){

      if (filter->possiblyContains(&hash)){

        //find old count
        int count = 1;
        if (map->find(hash) != map->end()){

          count =  2;
          map->erase(hash);
        }

        map->insert({hash, count});


      }

    }, bloomSecond, count_map, key.hash());


  }

  size_t map_size(){

    return count_map->size();
  }

  size_t cleanup(){

    //each processor should clean up its local copy of the map
    size_t count  = 0;
    for (auto i = count_map->begin(), last =  count_map->end(); i != last;){
      if ((*i).second < 2){
        i =  count_map->erase(i);
        count +=1;
      } else {
        ++i;
      }
    }

    return count;

  }

  //grab my local first filter and resize it to 1 (to prevent div by 0 errors on accidental calls)
  void reduceFirst(){
    bloomFirst->resize(1);
  }

  //reduce the size of the second filters after hashmaps are finished
  void reduceSecond(){
    bloomSecond->resize(1);
  }

  void add_starts(const pkmer_t & key){

    //like first pass, don't need response
    upcxx::rpc_ff(key.hash() % upcxx::rank_n(),

    []( count_hashmap & map,  uint64_t hash){


        //find old count
        int count = 1;
        if (map->find(hash) != map->end()){

          map->erase(hash);
        }

        map->insert({hash, count});



    }, count_map, key.hash());

  }

  //now that each segment has a unique
  void assign_nums(uint64_t start){

    size_t count  = 0;
    for (auto i = count_map->begin(), last =  count_map->end(); i != last;){
        i->second = start;
        start +=1;
        ++i;
      }

  }

  //return the combined future to store in a list
  upcxx::future<bool, uint64_t, bool, uint64_t> map_to_nums(const pkmer_t & kmerFirst, const pkmer_t & kmerSecond){

      //to  do this, we are gonna launch 2 futures
      //and use a .then  to capture and return
      upcxx::future<bool, uint64_t> first = findNum(kmerFirst);
      upcxx::future<bool, uint64_t> second =  findNum(kmerSecond);

      upcxx::future<bool, uint64_t, bool, uint64_t> result = upcxx::when_all(first,second);

      return result;
  }


  upcxx::future<bool, uint64_t> findNum(const pkmer_t & key){

    return upcxx::rpc(key.hash() % upcxx::rank_n(),

      []( count_hashmap & map,  uint64_t hash)-> upcxx::future<bool, uint64_t>
      {

        auto search = map->find(hash);
        if (search != map->end()){
          //found, return
          upcxx::future<bool, uint64_t> result = upcxx::when_all(true, search->second);
          return result;
        }
        upcxx::future<bool, uint64_t> result = upcxx::when_all(false, (uint64_t) 0);
        return result;
      }, count_map, key.hash());


  }

  





};
