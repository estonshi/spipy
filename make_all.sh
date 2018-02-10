#! /bin/bash
set -e

# decide your system
sys=`uname`
root_folder=`pwd`

if [ $sys != "Linux" ] && [ $sys != "Darwin" ]
then
	echo "I can't recognize your system. Exit."
	exit 1
fi

# decide your gcc
if [ $sys = "Darwin" ]
then
	nowgcc=`which gcc`
	echo "I need openmp and MPI support. Do you want to use current gcc [y/n]? : $nowgcc"
	read answer
	if [ $answer = "n" ]
	then
		echo "Give me your specific gcc path : "
		read mygcc
	else
		mygcc=gcc
	fi
fi

# start compiling ...
echo "==> compile image/bhtsne_source"
cd $root_folder/image/bhtsne_source
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2

echo "==> compile merge/template_emc/src"
cd $root_folder/merge/template_emc/src
if [ $sys = "Linux" ]
then
	mpicc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_LINUX -I ./ -lgsl -lgslcblas -lm -O3
elif [ $sys = "Darwin" ]
then
	$mygcc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_MAC -I ./ -lgsl -lgslcblas -lm -O3 -lmpi
fi

echo "==> compile simulate/src"
cd $root_folder/simulate/src
if [ $sys = "Linux" ]
then
	mpicc -fopenmp make_data.c -o make_data_LINUX -I ./ -lgsl -lgslcblas -lm -O3
elif [ $sys = "Darwin" ]
then
	$mygcc -fopenmp make_data.c -o make_data_MAC -I ./ -lgsl -lgslcblas -lm -O3
fi

echo "Complete!"
