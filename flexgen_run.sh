# must activate conda environment
# if not activate, please write "conda activate xxxx" in this file or assign the conda environment name to CONDA_ENV_NAME
# like this: PYTHON_PATH="xxx/envs/envs_name//bin/python", and modifed run_eval_tasks function to use $PYTHON_PATH

CONDA_ENV_PATH="/home/xu/anaconda3/envs/flexgen"
PYTHON_PATH="$CONDA_ENV_PATH/bin/python"

MODEL="llama2-7b"
PATH="/shared/model/Llama-2-7b-hf"
PERCENT="100 00 0 100 0 0 100 0 0"
BATCH_SIZE=1    # 4 is the default value, if you want to use the default value, please delete this line

# python flexgen --model ${MODEL} --path ${PATH} --percent ${PERCENT} --gpu-batch-size ${BATCH_SIZE}
$PYTHON_PATH flexgen --model ${MODEL} --path ${PATH} --percent ${PERCENT} --gpu-batch-size ${BATCH_SIZE}