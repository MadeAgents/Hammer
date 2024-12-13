#!/usr/bin/env bash
if [ $# -lt 1 ]; then
  echo "Usage: bash train.sh <MODEL>"
  exit 1
fi

MODEL=$1

CUR_DIR=$(cd $(dirname $0);pwd)
pwd_DIR=$(dirname $CUR_DIR)


LLAMA_FACTORY_HOME="${pwd_DIR}/train/LLaMA-Factory"
DATASET_DIR="${pwd_DIR}/data"
DATASET="masking_sft_data"
OUTPUT_DIR="${pwd_DIR}/ckpt/ $(basename $MODEL)-sft"
num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')

deepspeed --num_gpus $num_gpus "${LLAMA_FACTORY_HOME}/src/train.py" \
    --deepspeed "${LLAMA_FACTORY_HOME}/examples/deepspeed/ds_z2_config.json" \
    --stage sft \
    --do_train \
    --model_name_or_path $MODEL \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR \
    --template qwen \
    --dataset_dir $DATASET_DIR \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 32 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.00833 \
    --save_steps 1e9 \
    --evaluation_strategy "no" \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 1 \
    --max_samples 30000000 \
    --val_size 0.0 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --bf16