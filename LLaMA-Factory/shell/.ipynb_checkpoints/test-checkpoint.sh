# pip install gradio==3.39 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
# pip install -e . -i https://pypi.org/simple
# pip install pydantic==1.9.0 -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
pip install jieba nltk rouge-chinese -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
pip install pandas json-repair rouge -i http://mirrors.oppo.local/pypi --trusted-host mirrors.oppo.local
num_gpus=$(python -c 'import torch; print(torch.cuda.device_count())')

models=("Qwen/Qwen2-7B-Instruct")

for model in "${models[@]}"; do
echo $model
OUR_DIR="/home/notebook/data/group/model_hub/huggingface/$model"

testdata="unseenall_1d"
deepspeed --num_gpus $num_gpus src/train.py \
    --stage sft \
    --do_predict \
    --model_name_or_path $OUR_DIR \
    --dataset $testdata \
    --dataset_dir ../data \
    --template qwen \
    --finetuning_type full \
    --output_dir $OUR_DIR \
    --per_device_eval_batch_size 1 \
    --max_samples 4500 \
    --predict_with_generate \

echo $OUR_DIR
cd ../Evaluate
python evaluate_gpt.py $OUR_DIR $testdata
cd ../LLaMA-Factory2

# wait
# sleep 30

# testdata="unseenall2_1d"
# deepspeed --num_gpus $num_gpus src/train_bash.py \
#     --stage sft \
#     --do_predict \
#     --model_name_or_path $OUR_DIR \
#     --dataset $testdata \
#     --dataset_dir ../data \
#     --template qwen \
#     --finetuning_type full \
#     --output_dir $OUR_DIR \
#     --per_device_eval_batch_size 4 \
#     --max_samples 4500 \
#     --predict_with_generate \

# echo $OUR_DIR
# cd ../Evaluate
# python evaluate_gpt.py $OUR_DIR $testdata
# cd ../LLaMA-Factory
done