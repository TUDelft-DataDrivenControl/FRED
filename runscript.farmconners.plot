#!/bin/sh
#
#PBS -l nodes=1:ppn=16
#PBS -N cm.fc.multi

bash
cd $PBS_O_WORKDIR


# path to Python 3.6
export PATH=$HOME/anaconda3/bin:$PATH

# path to local program installations
export PATH=$HOME/opt/bin:$PATH

# specify PETSc environment variables
export PETSC_DIR=$HOME/petsc
export PETSC_ARCH=arch-linux2-c-debug

# set DOLFIN environment variables
source $HOME/opt/share/dolfin/dolfin.conf

# set ld search path
export LD_LIBRARY_PATH=$HOME/opt/lib:$HOME/opt/lib64:$LD_LIBRARY_PATH

# python
# todo install with SLEPC
# export SLEPC_DIR=$HOME/

# make sure cmake3 is used
alias cmake=cmake3

python ./runfarmconners.py plot

