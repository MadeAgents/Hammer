#!/bin/bash
# pip install gradio==3.39 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
# pip install -e . -i https://pypi.org/simple
# pip install pydantic==1.9.0 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
pip install pandas json-repair rouge -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local

num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')

for((i=0;i<10;i+=2))
do
OUR_DIR="/home/notebook/data/group/ComplexTaskDecision/ToolResearch/generalize/checkpoint/toolkit_qwen_pershot_qq1/epoch$i"
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