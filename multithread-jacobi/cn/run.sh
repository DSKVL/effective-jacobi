cores=3
main='c'$cores'g'
time='time'$main
g++ -std=c++23 -O3 -g -march=native -DCPU_CORES=$cores main.cpp -o $main
for i in 1 2 3 4 5 
do
  ./$main 10000 10000 120 2>> $time;
done
