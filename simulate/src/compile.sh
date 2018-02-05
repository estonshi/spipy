#! /bin/bash
sys=`uname`
nowpath=`dirname $0`
if [ $sys = "Linux" ]
then
	ln -fs $nowpath/make_data_LINUX $nowpath/../make_data
	echo " "
	echo "Please try the command below to compile if this 'make_data' executable doesn't work : "
	echo "'cd src; mpicc -fopenmp make_data.c -o make_data_LINUX -I ./ -lgsl -lgslcblas -lm -O3; ln -fs make_data_LINUX ../make_data'"
	echo " "
elif [ $sys = "Darwin" ]
then
	ln -fs $nowpath/make_data_MAC $nowpath/../make_data
	echo " "
	echo "[NOTICE] Try the command blow to compile if 'make_data' executable doesn't work : "
	echo "'cd src; /usr/local/bin/gcc-7 -fopenmp make_data.c -o make_data_MAC -I ./ -lgsl -lgslcblas -lm -O3; ln -fs make_data_MAC ../make_data'"
	echo " "
else
	echo " "
	echo "I can't recognize your system. Exit."
	echo " "
	exit 1
fi
