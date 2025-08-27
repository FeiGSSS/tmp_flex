MODEL_NAME="llama2-7b"
MODEL_PATH="/shared/model/Llama-2-7b-hf"
PERCENT="50 0 50 0 0 100 100 0 0"
BATCH_SIZE=4 

python flexgen --model ${MODEL_NAME} --path ${MODEL_PATH} --percent ${PERCENT} --gpu-batch-size ${BATCH_SIZE}