#!/bin/bash

# 设置输入文件夹和输出文件夹路径
BASE_INPUT_FOLDER=""
OUTPUT_BASE_FOLDER=""

# 获取所有输出文件夹名
INPUT_FOLDERS=($(find "$BASE_INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d))

GPUS=(0 1 2 3)
TOTAL_CORES=80
CORES_PER_JOB=20

running_jobs=0
total_start_time=$(date +%s)  # 记录总开始时间

for ((i=0; i<${#INPUT_FOLDERS[@]}; i++)); do
    INPUT_FOLDER="${INPUT_FOLDERS[$i]}"
    INPUT_NAME=$(basename "${INPUT_FOLDER}")

    if [ -d "${INPUT_FOLDER}/mesh" ]; then
        echo "Skipping folder: ${INPUT_FOLDER} (contains 'mesh' folder)"
        continue
    fi


    OUTPUT_FOLDER="${OUTPUT_BASE_FOLDER}/${INPUT_NAME}"

    GPU_ID=${GPUS[$((i % ${#GPUS[@]}))]}
    PORT=$((6009 + GPU_ID))

    CORE_START=$(( (i * CORES_PER_JOB) % TOTAL_CORES ))
    CORE_END=$(( CORE_START + CORES_PER_JOB - 1 ))
    CORE_LIST=$(seq -s "," $CORE_START $CORE_END)

    echo "Executing: CUDA_VISIBLE_DEVICES=${GPU_ID} OMP_NUM_THREADS=${CORES_PER_JOB} taskset -c ${CORE_LIST} python train.py -s ${INPUT_FOLDER} -m ${OUTPUT_FOLDER} --port ${PORT}"


    CUDA_VISIBLE_DEVICES=${GPU_ID} OMP_NUM_THREADS=${CORES_PER_JOB} taskset -c ${CORE_LIST} python train.py -s ${INPUT_FOLDER} -m ${OUTPUT_FOLDER} --max_abs_split_points 0 --opacity_cull_threshold 0.05 --port ${PORT} &

    ((running_jobs++))

    if (( running_jobs >= ${#GPUS[@]} )); then
        wait -n
        ((running_jobs--))
    fi
done

wait
