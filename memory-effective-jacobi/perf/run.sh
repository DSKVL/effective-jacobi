for i in 1 2 3 4 5 6 7 8 9 10 
do
  ./12it 10000 10000 120 >> ./time
done

perf record --freq=max ./12it 10000 10000 120
perf stat -r 20 -e instructions -e cycles -e branches -e branch-misses -ocyclesandmisses$program ./12it 10000 10000 120
perf stat -r 20 -e L1-dcache-loads -e L1-dcache-load-misses -e LLC-loads -e LLC-load-misses -ocachestats$program ./12it 10000 10000 120

