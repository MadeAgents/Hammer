#!/bin/bash
# pip install gradio==3.39 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
# pip install -e . -i https://pypi.org/simple
# pip install pydantic==1.9.0 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
pip install pandas json-repair rouge -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local

num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')
OUT_DIR="../checkpoint/toolkit_qwen15_M0_rejchat"

deepspeed --num_gpus $num_gpus src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path /home/notebook/data/group/model_hub/huggingface/Qwen/Qwen1.5-1.8B-Chat \
    --dataset train2_seen_rejchat \
    --output_dir $OUT_DIR \
    --template qwen \
    --dataset_dir ../data \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 32 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
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

# pip install jieba nltk rouge-chinese -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local

wait
sleep 30
for((i=0;i<10;i+=2))
do
OUR_DIR="../checkpoint/toolkit_qwen15_M0_rejchat/epoch$i"
testdata="test2_seen"
deepspeed --num_gpus $num_gpus src/train.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/notebook/data/group/model_hub/huggingface/Qwen/Qwen1.5-1.8B-Chat \
    --adapter_name_or_path $OUR_DIR \
    --dataset $testdata \
    --dataset_dir ../data \
    --template qwen \
    --finetuning_type lora \
    --output_dir $OUR_DIR \
    --per_device_eval_batch_size 4 \
    --max_samples 4500 \
    --predict_with_generate \
    --bf16

echo $OUR_DIR
cd ../Evaluate
python evaluate_gpt.py $OUR_DIR $testdata
cd ../LLaMA-Factory2
done

wait
sleep 30

for((i=0;i<10;i+=2))
do
OUR_DIR="../checkpoint/toolkit_qwen15_M0_rejchat/epoch$i"
testdata="unseenall2"
deepspeed --num_gpus $num_gpus src/train.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/notebook/data/group/model_hub/huggingface/Qwen/Qwen1.5-1.8B-Chat \
    --adapter_name_or_path $OUR_DIR \
    --dataset $testdata \
    --dataset_dir ../data \
    --template qwen \
    --finetuning_type lora \
    --output_dir $OUR_DIR \
    --per_device_eval_batch_size 4 \
    --max_samples 4500 \
    --predict_with_generate \
    --bf16

echo $OUR_DIR
cd ../Evaluate
python evaluate_gpt.py $OUR_DIR $testdata
cd ../LLaMA-Factory2
done

wait
sleep 30

for((i=0;i<10;i+=2))
do
OUR_DIR="../checkpoint/toolkit_qwen15_M0_rejchat/epoch$i"
testdata="test_chat"
deepspeed --num_gpus $num_gpus src/train.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/notebook/data/group/model_hub/huggingface/Qwen/Qwen1.5-1.8B-Chat \
    --adapter_name_or_path $OUR_DIR \
    --dataset $testdata \
    --dataset_dir ../data \
    --template qwen \
    --finetuning_type lora \
    --output_dir $OUR_DIR \
    --per_device_eval_batch_size 4 \
    --max_samples 4500 \
    --predict_with_generate \
    --bf16

echo $OUR_DIR
cd ../Evaluate
python evaluate_reject.py $OUR_DIR $testdata
cd ../LLaMA-Factory2
done