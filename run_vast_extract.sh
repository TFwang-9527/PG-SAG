#!/bin/bash

BASE_INPUT_FOLDER=""
OUTPUT_BASE_FOLDER=""

INPUT_FOLDERS=($(find "$BASE_INPUT_FOLDER" -mindepth 1 -maxdepth 1 -type d))

GPUS=(1 2 3)
TOTAL_CORES=80
CORES_PER_JOB=20

running_jobs=0
total_start_time=$(date +%s)

for ((i=0; i<${#INPUT_FOLDERS[@]}; i++)); do
    GPU_ID=${GPUS[$((i % ${#GPUS[@]}))]}
    INPUT_FOLDER="${INPUT_FOLDERS[$i]}"
    INPUT_NAME=$(basename "${INPUT_FOLDER}")

    OUTPUT_FOLDER="${OUTPUT_BASE_FOLDER}/${INPUT_NAME}"

    if [ -d "${OUTPUT_FOLDER}/mesh" ]; then
        echo "Skipping ${OUTPUT_FOLDER} because it contains a mesh folder."
        continue
    fi

    PORT=$((6009 + GPU_ID))

    CORE_START=$(( (i * CORES_PER_JOB) % TOTAL_CORES ))
    CORE_END=$(( CORE_START + CORES_PER_JOB - 1 ))
    CORE_LIST=$(seq -s "," $CORE_START $CORE_END)

    echo "Executing: CUDA_VISIBLE_DEVICES=${GPU_ID} OMP_NUM_THREADS=${CORES_PER_JOB} taskset -c ${CORE_LIST} python render.py -m ${OUTPUT_FOLDER} --max_depth 500 --voxel_size 0.002 --use_depth_filter --iteration 30000"


    CUDA_VISIBLE_DEVICES=${GPU_ID} OMP_NUM_THREADS=${CORES_PER_JOB} taskset -c ${CORE_LIST} python render.py -m ${OUTPUT_FOLDER} --max_depth 500 --voxel_size 0.005 --use_depth_filter --iteration 30000 &
    ((running_jobs++))

    if (( running_jobs >= ${#GPUS[@]} )); then
        wait -n
        ((running_jobs--))
    fi
done

wait
