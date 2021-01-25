#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <numeric>
#include <set>
#include <upcxx/upcxx.hpp>
#include <vector>

#include "bloom_dist.hpp"
#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "counter.hpp"

#include "butil.hpp"

int main(int argc, char** argv) {
    upcxx::init();

    auto total_start = std::chrono::high_resolution_clock::now();


    if (argc < 2) {
        BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test [prefix]]\n");
        upcxx::finalize();
        exit(1);
    }

    std::string kmer_fname = std::string(argv[1]);
    std::string run_type = "";

    if (argc >= 3) {
        run_type = std::string(argv[2]);
    }

    std::string test_prefix = "test";
    if (run_type == "test" && argc >= 4) {
        test_prefix = std::string(argv[3]);
    }

    int ks = kmer_size(kmer_fname);

    if (ks != KMER_LEN) {
        throw std::runtime_error("Error: " + kmer_fname + " contains " + std::to_string(ks) +
                                 "-mers, while this binary is compiled for " +
                                 std::to_string(KMER_LEN) +
                                 "-mers.  Modify packing.hpp and recompile.");
    }

    size_t n_kmers = line_count(kmer_fname);

    // p false positive rate
    double p = .01;
    uint64_t m = n_kmers / upcxx::rank_n() * std::log2(p) / std::log2(.6185);
    uint8_t k = std::log2(m/n_kmers);

    //involve all processors in hashmap
    //DistrMap hashmap();
    BloomDist filters = BloomDist(m, k);

    if (run_type == "verbose") {
        BUtil::print("Initializing Bloomfilters of %d bits for %d kmers -  Approx %d kmers per node.\n", m,
                     n_kmers, n_kmers/upcxx::rank_n());
    }

    std::vector<kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

    if (run_type == "verbose") {
        BUtil::print("Finished reading kmers.\n");
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<kmer_pair> start_nodes;

    for (auto& kmer : kmers) {
        filters.first_pass(kmer.kmer);

        if (kmer.backwardExt() == 'F') {
            start_nodes.push_back(kmer);
        }
    }

    upcxx::barrier();
    auto end_insert = std::chrono::high_resolution_clock::now();

    double insert_time = std::chrono::duration<double>(end_insert - start).count();
    if (run_type != "test") {
        BUtil::print("Finished first pass in %lf\n", insert_time);
    }
    upcxx::barrier();

    start = std::chrono::high_resolution_clock::now();
    //resize first to save memory for hashmaps
    filters.reduceFirst();

    for (auto& kmer: kmers){
      filters.second_pass(kmer.kmer);
    }

    upcxx::barrier();
    end_insert = std::chrono::high_resolution_clock::now();

    insert_time = std::chrono::duration<double>(end_insert - start).count();

    if (run_type == "verbose") {
        BUtil::print("Finished second pass in %lf\n", insert_time);
    }

    filters.reduceSecond();

    if (run_type=="verbose"){
      BUtil::print("P0 has %d items\n", filters.map_size());
    }

    //have each filter run cleanup to remove kmers with only one
    start = std::chrono::high_resolution_clock::now();
    size_t count = filters.cleanup();

    //barrier not needed
    //upcxx::barrier();
    end_insert = std::chrono::high_resolution_clock::now();

    insert_time = std::chrono::duration<double>(end_insert - start).count();

    if (run_type == "verbose") {
        BUtil::print("cleaned up %d bad kmers in %lf\n", count, insert_time);
        BUtil::print("P0 has %d items\n", filters.map_size());
    }

    upcxx::barrier();

    //add back in the start kmers - they must always be present
    for (auto& kmer: start_nodes){
      filters.add_starts(kmer.kmer);
    }

    //unload the scan sizes into counters
    upcxx::dist_object<uint64_t> counter = filters.map_size();

    //perform a slow_ish sum scan
    ring_sum(counter, 0,0);

    upcxx::barrier();

    filters.assign_nums( * counter);

    //now start the assignment process
    //first, we create a storage to hold the futures

    start = std::chrono::high_resolution_clock::now();
    std::vector<upcxx::future<bool, uint64_t, bool, uint64_t>> futures;

    for (auto& kmer: kmers){

      upcxx::future<bool, uint64_t, bool, uint64_t> next = filters.map_to_nums(kmer.kmer, kmer.next_kmer());
      futures.push_back(next);
    }

    //barrier to ensure all calls  sent
    //then discharge to clear rpcs
    //then barrier to ensure all finished
    //this may not be necessary and may be slower than just running
    //to the next loop
    upcxx::barrier();
    upcxx::discharge();
    upcxx::barrier();
    //run through futures
    count = 0;
    for (auto& next_future: futures){

      //these should all be ready, but this guarantees consistency
      auto result = next_future.wait_tuple();

      if (std::get<0>(result) && std::get<2>(result)){
        count += 1;
      }

    }

    upcxx::barrier();
    end_insert = std::chrono::high_resolution_clock::now();
    insert_time = std ::chrono::duration<double>(end_insert - start).count();

    if (run_type == "verbose") {
        BUtil::print("converted to uint64_t in %lf, found  %d matches.\n", insert_time, count);
    }


    auto total_end = std::chrono::high_resolution_clock::now();
    insert_time = std::chrono::duration<double>(total_end - total_start).count();

    if (run_type == "verbose") {
        BUtil::print("Fully finished in %lf.\n", insert_time);
    }


    upcxx::finalize();
    return 0;
}
