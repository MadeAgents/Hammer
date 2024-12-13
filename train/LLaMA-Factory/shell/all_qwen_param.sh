#!/bin/bash
# pip install gradio==3.39 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
# pip install -e . -i https://pypi.org/simple
# pip install pydantic==1.9.0 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
pip install pandas json-repair rouge -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
#pip install auto_gptq==0.5.0 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# python -m pip install auto_gptq
# python -m pip install optimum
python -m pip install autoawq
python -m pip install bitsandbytes

num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')
OUT_DIR="../checkpoint/all_qwen_newparam_missbad419_AWQint4"

deepspeed --num_gpus $num_gpus src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path /home/notebook/data/group/model_hub/huggingface/Qwen/Qwen1.5-1.8B-Chat-AWQ \
    --dataset newparam_all_miss \
    --output_dir $OUT_DIR \
    --template qwen \
    --dataset_dir ../data \
    --finetuning_type lora \
    --quantization_bit 4 \
    --lora_target all \
    --lora_rank 32 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --warmup_steps 20 \
    --save_steps 1e9 \
    --eval_steps 1000000000000000000000 \
    --evaluation_strategy steps \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --num_train_epochs 10 \
    --max_samples 30000000 \
    --val_size 0.0 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --bf16