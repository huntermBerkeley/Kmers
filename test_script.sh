#!/bin/sh

for i in {8,16}
do
	for j in {4,8,16}
	do
    for k in {64,128,256}
    do
		    echo "I:$i J:$j K:$k"
        CC src/test_matmul.cpp -o test_matmul -std=c++17 -O3 -lstdc++ -DI_SIZE=$i -DJ_SIZE=$j -DK_SIZE=$k
		    srun -N 1 -n 1 ./test_matmul
    done
	done
done
