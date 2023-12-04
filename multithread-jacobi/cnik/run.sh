for cores in 1 2 3 4 
do
  for iters in 1 2 3 4 5 6 8 10 12
  do
    main='c'$cores'i'$iters'nd'
    time='time'$main
    g++ -std=c++23 -O3 -march=native -DITERS_PER_RUN=$iters -DCPU_CORES=$cores -DNO_DELTA main.cpp -o $main
    for i in 1 2 3 4 5 
    do
      ./$main 10000 10000 120 2>> $time;
    done
  done
done
