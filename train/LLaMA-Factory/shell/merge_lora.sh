OUR_DIR="/home/notebook/data/group/ComplexTaskDecision/ToolResearch/RAG_SFT/checkpoint/all_qwen15_param"
python src/export_model.py \
    --model_name_or_path /home/notebook/data/group/model_hub/huggingface/Qwen/Qwen1.5-1.8B-Chat \
    --adapter_name_or_path /home/notebook/data/group/ComplexTaskDecision/ToolResearch/RAG_SFT/checkpoint/all_qwen_param/epoch4 \
    --template qwen \
    --finetuning_type lora \
    --export_dir $OUR_DIR \
    --export_size 1 \
    --export_legacy_format False