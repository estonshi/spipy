#!/bin/bash

### Job Name
#PBS -N phasing3d
### OutPut Files
#PBS -o phasing3d.stdout
#PBS -e phasing3d.stderr
### Queue Name
#PBS -q low
### Number of nodes
#PBS -l nodes=1:ppn=24

source /home/ycshi/.bashrc
source /public/software/profile.d/mpi_openmpi-2.0.0.sh

cd $PBS_O_WORKDIR
