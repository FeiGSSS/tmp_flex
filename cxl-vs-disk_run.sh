#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

#  限制 8 个 core
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export GOTO_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export BLAS_NUM_THREADS=8


MODEL_NAME=llama2-7b
MODEL_PATH=/shared/model/Llama-2-7b-hf
GPU_BATCH_SIZE=10
GEN_LEN=128

nvidia-smi -i 0 --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > gpu_usage_disk.csv &
GPUPID=$!
numactl --cpunodebind=0,1 --membind=0,1 taskset -c 0-7 python -m flexgen --model $MODEL_NAME --path $MODEL_PATH --gpu-batch-size $GPU_BATCH_SIZE --gen-len $GEN_LEN --percent 100 0 0 0 0 0 100 0 0
kill $GPUPID
sleep 1

nvidia-smi -i 0 --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > gpu_usage_numa.csv &
GPUPID=$!
numactl --cpunodebind=0,1 --membind=0,1 taskset -c 0-7 python -m flexgen --model $MODEL_NAME --path $MODEL_PATH --gpu-batch-size $GPU_BATCH_SIZE --gen-len $GEN_LEN --percent 100 0 0 0 0 100 100 0 0
kill $GPUPID
sleep 1

python3 plot_gpu_usage.py