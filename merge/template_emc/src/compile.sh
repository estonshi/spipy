#! /bin/bash
set -e

if [ -z $1 ]
then
    echo "Usage : ./compile.sh [topath]"
    exit 1
fi

savepath=$1
sys=`uname`
nowpath=`dirname $0`
if [ $sys = "Linux" ]
then
	ln -fs $nowpath/emc_LINUX $savepath/emc
	echo " "
	echo "Please try the command below to compile if this 'emc' executable doesn't work : "
	echo "'cd $nowpath; mpicc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_LINUX -I ./ -lgsl -lgslcblas -lm -O3'"
	echo " "
elif [ $sys = "Darwin" ]
then
	ln -fs $nowpath/emc_MAC $savepath/emc
	echo " "
	echo "[NOTICE] Try the command below to compile if 'emc' executable doesn't work : "
	echo "'cd $nowpath; /usr/local/bin/gcc-7 -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_MAC -I ./ -lgsl -lgslcblas -lm -O3 -lmpi'"
	echo " "
else
	echo " "
	echo "I can't recognize your system. Exit."
	echo " "
	exit 1
fi
