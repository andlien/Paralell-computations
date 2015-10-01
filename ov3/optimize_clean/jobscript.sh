#!/bin/bash
#PBS -N problem_set_3
#PBS -A ntnu605
#PBS -l walltime=00:01:00
#PBS -l select=1:ncpus=1:mpiprocs=1
  
cd $PBS_O_WORKDIR
 
make run