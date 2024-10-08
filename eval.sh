#!/usr/bin/env bash

# 检查参数数量
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <MODEL> <DATASET>"
  exit 1
fi

MODEL=$1
DATASET=$2

CUR_DIR=$(cd $(dirname $0);pwd)
pwd_DIR=$(dirname $(dirname $CUR_DIR))
LLAMA_FACTORY_HOME="${pwd_DIR}/LLaMA-Factory"
DATASET_DIR="${pwd_DIR}/data"
OUTPUT_DIR="${pwd_DIR}/predict/$(basename $MODEL)/$DATASET"
echo "Evaluating model: $(basename $MODEL) on dataset: ${DATASET}"
# num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')

# deepspeed --num_gpus $num_gpus "${LLAMA_FACTORY_HOME}/src/train.py" \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path $MODEL \
#     --dataset $DATASET \
#     --dataset_dir $DATASET_DIR \
#     --template qwen \
#     --output_dir $OUTPUT_DIR \
#     --per_device_eval_batch_size 4 \
#     --predict_with_generate \
#     --max_samples 10000000 \
#     --cutoff_len 6400 \
#     --max_new_tokens 512 \
#     --bf16

# python evaluate/evaluate.py $OUTPUT_DIR