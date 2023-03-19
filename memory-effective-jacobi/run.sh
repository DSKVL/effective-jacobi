for program in 15 20 24
do
  g++ -O3 -march=native -DITERS_PER_RUN=$program -DNO_DELTA -o $program'itnodelta' templateIters.cpp
  for i in 1 2 3 4 5 6 7 8 9 10 
  do
    ./$program'itnodelta' 10000 10000 120 >> $program'time'
  done
  
  perf stat -r 20 -e instructions -e cycles -e branches -e branch-misses -ocyclesandmisses$program ./$program'itnodelta' 10000 10000 120
  perf stat -r 20 -e L1-dcache-loads -e L1-dcache-load-misses -e LLC-loads -e LLC-load-misses -ocachestats$program ./$program'itnodelta' 10000 10000 120
done