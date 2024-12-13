#!/usr/bin/env bash
# Check the number of parameters
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <MODEL> <DATASET>"
  exit 1
fi

MODEL=$1
DATASET=$2

# Check if the model name contains "Hammer2.1"
if [[ "$(basename $MODEL)" =~ [Hh]ammer2\.1 ]]; then
  DATASET="${DATASET}_hammer2.1"
  TEMPLATE="qwen_hammer"
else
  DATASET="${DATASET}_hammer"
  TEMPLATE="qwen"
fi
CUR_DIR=$(cd $(dirname $0);pwd)
pwd_DIR=$(dirname $CUR_DIR)
LLAMA_FACTORY_HOME="${pwd_DIR}/train/LLaMA-Factory"
DATASET_DIR="${pwd_DIR}/data"
OUTPUT_DIR="${pwd_DIR}/predict/$(basename $MODEL)/$DATASET"
echo "Evaluating model: $(basename $MODEL) on dataset: ${DATASET} $TEMPLATE"
num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')

# Set temperature to 0.000001 to ensure reproducibility

deepspeed --num_gpus $num_gpus "${LLAMA_FACTORY_HOME}/src/train.py" \
    --stage sft \
    --do_predict \
    --model_name_or_path $MODEL \
    --dataset $DATASET \
    --dataset_dir $DATASET_DIR \
    --template $TEMPLATE \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --max_samples 10000000 \
    --cutoff_len 6000 \
    --temperature 0.0001 \
    --max_new_tokens 512 \
    --bf16
python evaluation/evaluate.py $OUTPUT_DIR