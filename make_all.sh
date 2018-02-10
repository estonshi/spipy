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
chmod u+x bh_tsne

echo "==> compile merge/template_emc/src"
cd $root_folder/merge/template_emc/src
chmod u+x compile.sh ../new_project
if [ $sys = "Linux" ]
then
	mpicc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_LINUX -I ./ -lgsl -lgslcblas -lm -O3
	chmod u+x emc_MAC
elif [ $sys = "Darwin" ]
then
	$mygcc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_MAC -I ./ -lgsl -lgslcblas -lm -O3 -lmpi
	chmod u+x emc_MAC
fi

echo "==> compile simulate/src"
cd $root_folder/simulate/src
chmod u+x compile.sh ../code/make_densities.py ../code/make_detector.py ../code/make_intensities.py
if [ $sys = "Linux" ]
then
	mpicc -fopenmp make_data.c -o make_data_LINUX -I ./ -lgsl -lgslcblas -lm -O3
	chmod u+x make_data_LINUX
elif [ $sys = "Darwin" ]
then
	$mygcc -fopenmp make_data.c -o make_data_MAC -I ./ -lgsl -lgslcblas -lm -O3
	chmod u+x make_data_MAC
fi

echo "==> others"
cd $root_folder/phase
chmod u+x ./template_2d/new_project ./template_3d/new_project

echo "Complete!"
