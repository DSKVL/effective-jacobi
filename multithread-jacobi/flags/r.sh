cores=4
main='c'$cores'g'
time='time'$main
g++ -std=c++23 -O3 -g -march=native -DCPU_CORES=$cores main.cpp -o $main
for i in 1 2 3 4 5 
do
  ./$main 10000 10000 120 2>> $time;
done
perf stat -r 20 -e instructions -e cycles -e branches -e branch-misses -ocyclesandmisses ./$main 10000 10000 120
perf stat -r 20 -e L1-dcache-loads -e L1-dcache-load-misses -e LLC-loads -e LLC-load-misses -ocachestats ./$main 10000 10000 120