#!/usr/bin/env bash

#SBATCH --job-name=strassen2
#SBATCH --output=strassenfor.txt
#SBATCH --ntasks-per-node=7
#SBATCH --nodes=1
#SBATCH --time=05:00
#SBATCH -p short-24core
#SBATCH --mail-type=END
#SBATCH --mail-user=sophia.nolas@stonybrook.edu

module load slurm/17.11.12
module load intel/mpi/64/2020/20.0.2
module load intel/mkl/64/2020/20.0.2

mpicc strassenNforty.cpp -o sfor -lstdc++

mpirun -np 7 ./sfor
