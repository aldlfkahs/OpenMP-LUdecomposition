# LU-decomposition by using OpenMP
This project is for studying parallel computing using openMP. I made code parallel especially in for loop. I tried to optimize parallelism using such as 'nowait' or 'lastprivate'. Through the studying this project, I could conclude that more core make faster execution but parallel efficiency decreases.

This program should be executed in multicore system. 

#### Makefile:
  
  a Makefile that includes recipes for building and running the program



#### usage:

  make # builds code
  
  make runp # runs a parallel version of code on W workers
  
  make runs # runs a serial version of code on one worker.
  
  make check # runs parallel code with Intel Thread Checker
  
