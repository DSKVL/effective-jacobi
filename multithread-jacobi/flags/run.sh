for cpu_cores in 3
do
  g++ -O3 -march=native -std=c++23 -DCPU_CORES=$cpu_cores -o $cpu_cores'it' main.cpp
  for i in 1 2 3 4 5 
  do
    ./$cpu_cores'it' 10000 10000 120 >> $cpu_cores'time'
  done
done