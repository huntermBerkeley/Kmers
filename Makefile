ifeq (, $(shell which CC))
	CXX = g++
else
	CXX = CC
endif

all: bloom_test bloom_dist counter


bloom_test: bloom_test.cpp bloom.hpp MurmurHash3.cpp
	$(CXX) bloom_test.cpp MurmurHash3.cpp -o bloom_test -std=c++17 -O3 -lstdc++

bloom_dist: bloom_dist_test.cpp bloom.hpp MurmurHash3.cpp bloom_dist.hpp butil.hpp  kmer_t.hpp read_kmers.hpp pkmer_t.hpp packing.hpp
	upcxx bloom_dist_test.cpp MurmurHash3.cpp -o bloom_dist_test -std=c++17 -O3 -lstdc++ -march=knl

counter: counter_test.cpp counter.hpp
	upcxx counter_test.cpp -o counter_test -std=c++17  -O3 -lstdc++ -march=knl

clean:
	@rm -fv bloom_test
	@rm -fv bloom_dist_test
