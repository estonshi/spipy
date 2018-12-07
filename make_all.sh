#! /bin/bash
root_folder=`pwd`

# get opts
SKIP_COMPLIE=0
while getopts ":xh" opt; do
	case $opt in
		x)
			SKIP_COMPLIE=1
			;;
		h)
			echo "Use : ./make_all.sh [-x] (skip compiling MPI part)"
			echo "                    [-h] (help info)"
			exit 0
			;;
		\?)
			echo "[Error] Invalid option : $OPTARG . Exit."
			exit 1
			;;
	esac
done

# check whether there is anaconda installed
Ana_path=`which python`
a='anaconda'
b='miniconda'
if [[ $Ana_path =~ $a ]] || [[ $Ana_path =~ $b ]]
then
	echo "[Info] Root folder is $root_folder"
else
	echo "[Error] Use anaconda2/miniconda2 please. Exit."
	exit 1
fi
py_version=`conda info | grep python`
a='2.7'
if [[ $py_version =~ $a ]]
then
	echo "==> Anaconda version authorized"
else
	echo "[Error] Your python version is not 2.7. Exit."
	exit 1
fi
if [ $SKIP_COMPLIE -eq 1 ]; then
	echo "[Info] Skip compiling merge.emc and simulate.sim module."
fi

# decide your system
sys=`uname`

if [ $sys != "Linux" ] && [ $sys != "Darwin" ]
then
	echo "[Error] I can't recognize your system. Exit."
	exit 1
fi

if [ $SKIP_COMPLIE -eq 0 ]; then
	# decide your gcc
	if [ $sys = "Darwin" ]
	then
		nowgcc=`which gcc`
		echo "[Warning] I need openmp and MPI support. Do you want to use current gcc? : $nowgcc [y/n]"
		flag=0
		while [ $flag = 0 ]
		do
			read answer
			if [ $answer = "n" ]
			then
				echo "==> Give me your specific gcc path : "
				read mygcc
				flag=1
			elif [ $answer = "y" ]
			then
				mygcc=gcc
				flag=1
			else
				echo "[Warning] Please give 'y' or 'n'."
			fi
		done
	fi
	# reject conda mpi
	if [ $sys = "Linux" ]
	then
		nowmpicc=`which mpicc`
		nowmpirun=`which mpirun`
		if [ $nowmpicc = "${Ana_path%/bin/python*}/bin/mpicc" ] || [ $nowmpirun = "${Ana_path%/bin/python*}/bin/mpirun" ]
		then
			echo "==>I can't use mpi in anaconda/miniconda to compile myself."
			echo "   Give me your specific mpicc path (type 'n' to exit) : "
			read mympicc
			if [ $mympicc = "n" ]
			then
				exit 1
			fi
		else
			mympicc=$nowmpicc
		fi
	fi
fi

# start compiling ...
echo "==> compile image/bhtsne_source"
cd $root_folder/image/bhtsne_source
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
if [ $? -ne 0 ];then echo $?; exit 1;fi
chmod u+x bh_tsne


if [ $SKIP_COMPLIE -eq 0 ]; then

	echo "==> compile merge/template_emc/src"
	cd $root_folder/merge/template_emc/src
	chmod u+x compile.sh ../new_project
	if [ $sys = "Linux" ]
	then
		$mympicc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_LINUX -I ./ -lgsl -lgslcblas -lm -O3
		if [ $? -ne 0 ];then echo $?; exit 1;fi
		chmod u+x emc_LINUX
	elif [ $sys = "Darwin" ]
	then
		$mygcc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_MAC -I ./ -lgsl -lgslcblas -lm -O3 -lmpi
		if [ $? -ne 0 ];then echo $?; exit 1;fi
		chmod u+x emc_MAC
	fi


	echo "==> compile simulate/src"
	cd $root_folder/simulate/src
	chmod u+x compile.sh ../code/make_densities.py ../code/make_detector.py ../code/make_intensities.py
	if [ $sys = "Linux" ]
	then
		$mympicc -fopenmp make_data.c -o make_data_LINUX -I ./ -lgsl -lgslcblas -lm -O3
		if [ $? -ne 0 ];then echo $?; exit 1;fi
		chmod u+x make_data_LINUX
	elif [ $sys = "Darwin" ]
	then
		$mygcc -fopenmp make_data.c -o make_data_MAC -I ./ -lgsl -lgslcblas -lm -O3
		if [ $? -ne 0 ];then echo $?; exit 1;fi
		chmod u+x make_data_MAC
	fi

fi


echo "==> install packages"
tmp=`conda list | grep "mrcfile"`
if [ -z "$tmp" ];then pip install mrcfile;fi
if [ $? -ne 0 ];then echo $?; exit 1;fi

tmp=`conda list | grep "mpi4py"`
if [ -z "$tmp" ];then pip install mpi4py;fi
if [ $? -ne 0 ];then echo $?; exit 1;fi


echo "==> others"
cd $root_folder/phase
chmod u+x ./template_2d/new_project ./template_3d/new_project
cd $root_folder/image/qlist_dir
chmod u+x ./gen_quat

# make soft link
if [ ! -d "${Ana_path%/bin/python*}/lib/python2.7/site-packages/spipy" ]
then
	ln -fs $root_folder ${Ana_path%/bin/python*}/lib/python2.7/site-packages/spipy
else
	echo "[Warning] spipy is already in python2.7/site-packages. Over-write it? [y/n]"
	flag=0
	while [ $flag = 0 ]
	do
		read overwrite
		if [ $overwrite = "y" ]
		then
			rm ${Ana_path%/bin/python*}/lib/python2.7/site-packages/spipy
			ln -fs $root_folder ${Ana_path%/bin/python*}/lib/python2.7/site-packages/spipy
			flag=1
		elif [ $overwrite = "n" ]
		then
			echo "Skip."
			flag=1
		else
			echo "[Warning] Please give 'y' or 'n'."
		fi
	done
fi

echo "==> Complete!"
