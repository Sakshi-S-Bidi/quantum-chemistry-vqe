#!/usr/bin/env bash

export UCX_IB_MLX5_DEVX=no
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=1
export QULACS_NUM_THREADS=48

source $1/bin/activate
shift

if [ -z "${LD_PRELOAD}" ]; then
 export LD_PRELOAD=/lib64/libgomp.so.1
else
 export LD_PRELOAD=/lib64/libgomp.so.1:$LD_PRELOAD
fi

$@
