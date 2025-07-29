#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 更强力的限制
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export GOTO_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export BLAS_NUM_THREADS=8

nvidia-smi -i 0 --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > gpu_usage_disk.csv &
GPUPID=$!
numactl --cpunodebind=0,1 --membind=0,1 taskset -c 0-7 python3 -m flexllmgen.flex_opt --model facebook/opt-13b --path __DUMMY__ --percent 0 100 0 0 0 0 100 0 0
kill $GPUPID
sleep 1

nvidia-smi -i 0 --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > gpu_usage_numa.csv &
GPUPID=$!
numactl --cpunodebind=0,1 --membind=0,1 taskset -c 0-7 python3 -m flexllmgen.flex_opt --model facebook/opt-13b --path __DUMMY__ --percent 0 100 0 0 0 100 100 0 0
kill $GPUPID
sleep 1

python3 plot_gpu_usage.py