# must activate conda environment
# if not activate, please write "conda activate xxxx" in this file or assign the conda environment name to CONDA_ENV_NAME
# like this: PYTHON_PATH="xxx/envs/envs_name//bin/python", and modifed run_eval_tasks function to use $PYTHON_PATH


TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
BASE_OUTPUT_DIR="./eval_out"
# --- 辅助函数，用于执行一组评估任务 ---
# 参数: 1:模型HF名, 2:模型本地路径, 3:Batch Size, 4:Percent参数列表, 5:策略名称, 6:任务列表
run_eval_tasks() {
    # 从参数中读取变量
    local MODEL_HF_NAME=$1
    local MODEL_LOCAL_PATH=$2
    local BATCH_SIZE=$3
    local PERCENT_ARGS="$4"   # 将所有percent数字作为一个字符串
    local STRATEGY_NAME=$5
    local TASKS=$6
    # 从模型名中提取简称，如 "opt-6.7b"
    local MODEL_SHORT_NAME=$(basename ${MODEL_HF_NAME})

    echo "======================================================================"
    echo "🚀 开始测试模型: ${MODEL_SHORT_NAME}    策略: ${STRATEGY_NAME}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "Percent Config: ${PERCENT_ARGS}"
    echo "======================================================================"

    # 循环执行所有需要的任务
    for TASK in ${TASKS}; do
        # 创建更有条理的输出路径，例如: .../eval_out/Balanced/toxigen_opt-6.7b_flexllmgen
        local OUTPUT_DIR="${BASE_OUTPUT_DIR}/${STRATEGY_NAME}/${TIMESTAMP}"
        local OUTPUT_PATH="${OUTPUT_DIR}/${TASK}_${MODEL_SHORT_NAME}_${MODEL_TYPE}_${BATCH_SIZE}"
        local LOG_FILE_PATH="${OUTPUT_DIR}/log_${TASK}_${MODEL_SHORT_NAME}.log"
        
        # 确保输出目录存在
        mkdir -p "${OUTPUT_DIR}"

        echo "-------> 正在执行任务: ${TASK}"
        echo "         输出路径: ${OUTPUT_PATH}"
        echo "         日志文件: ${LOG_FILE_PATH}"
        # if [ "$TASK" = "xsum" ]; then  # 修正：在"]"前添加空格
        #     echo "         正在执行 XSUM 任务 需修改 BATCH_Size"
        #     if [ "$MODEL_SHORT_NAME" = "opt-6.7b" ]; then
        #         BATCH_SIZE=8
        #     else  # 修正：去掉多余的"; then"
        #         BATCH_SIZE=32
        #     fi
        # fi
        # 执行评估命令，并传入 --percent 参数
        # $PYTHON_PATH eval_utils \
        #     --model "${MODEL_HF_NAME}" \
        #     --model_type "${MODEL_TYPE}" \
        #     --path "${MODEL_LOCAL_PATH}" \
        #     --tasks "${TASK}" \
        #     --output_path "${OUTPUT_PATH}" \
        #     --gpu-batch-size "${BATCH_SIZE}" \
        #     --percent ${PERCENT_ARGS} 2>&1 | tee "${LOG_FILE_PATH}" # 注意：这里不需要引号，以传递多个参数
        python3 eval_utils \
            --model "${MODEL_HF_NAME}" \
            --model_type "${MODEL_TYPE}" \
            --path "${MODEL_LOCAL_PATH}" \
            --tasks "${TASK}" \
            --output_path "${OUTPUT_PATH}" \
            --gpu-batch-size "${BATCH_SIZE}" \
            --percent ${PERCENT_ARGS} 2>&1 | tee "${LOG_FILE_PATH}" # 注意：这里不需要引号，以传递多个参数


        echo "<------- ✅ 任务 ${TASK} 完成"
        echo
    done
}

BATCH_SIZE=8
MODEL_TYPE="flexgen_model" # Now Support flexgen_model, flexllmgen and hf, 
                           # flexgen_model and flexllmgen difference is computed by GPU or CPU
                           # hf is huggingface model
MODEL_NAME="facebook/opt-6.7b"
MODEL_PATH="/shared/model/opt/opt-6.7b"
PERCENT="100 0 0 100 0 0 100 0 0"
STRATEGY_NAME="ALL_in_GPU"
TASKS="toxigen mmlu xsum"
run_eval_tasks "${MODEL_NAME}" "${MODEL_PATH}" "${BATCH_SIZE}" "${PERCENT}" "${STRATEGY_NAME}" "${TASKS}"








