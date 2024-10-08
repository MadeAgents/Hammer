# Hammer

The source code and dataset mentioned in the paper [```Hammer: Robust Function-Calling for On-Device Language Models via Function Masking```](https://arxiv.org/pdf/2410.04587). 

## Overview

The Hammer series employs function masking to selectively focus on specific functions during inference, improving the robustness and efficiency of function-calling. These methods are especially valuable for on-device language processing applications where computational resources are limited.


We have released several models based on the ```Function Masking``` techniques discussed in the paper, available on [MadeAgents on Hugging Face](https://huggingface.co/MadeAgents).

## Fine-Tuning

### Install Dependencies

Install required packages by running the following command:

```pip install -r requirements.txt```

### Data Processing

Based on the original data of 'xlam_function_calling_60k' and 'xlam-7.5k-irrelevance' (available at [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), [MadeAgents/XLAM-7.5k-Irrelevance](https://huggingface.co/datasets/MadeAgents/XLAM-7.5k-Irrelevance)), download these datasets and place them in the `data` directory. Then, process the data including function masking by running:

```python data_processing.py```

### Train the Model
To start training the model, use the shell script `train.sh` with the path of the base model as an argument:

```bash train.sh model_path```

Here, 'model_path' represents the path of the base model, e.g., **Qwen/Qwen2-1.5B-Instruct**.

## Evaluation

To evaluate a model on a specific dataset, you can run the evaluation script 'eval.sh'. The script requires two parameters: MODEL (the path to the model) and DATASET (the dataset to be used for evaluation). And our evaluation code supports models trained with our framework or directly evaluating Hammer series models available on Hugging Face.

### Available Datasets


- **apibank_l1** ([API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank/tree/main)): This dataset underwent prompt conversion and incorporated conversation history as query input to the model, resulting in 399 samples.
- **apibank_l2** ([API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank/tree/main)): Prompt conversion was applied, and 8 problematic samples (where the label function was not in the candidate functions) were filtered out, resulting in 127 samples.
- **NexusRaven** ([NexusRaven](https://huggingface.co/datasets/Nexusflow/Function_Call_Definitions)): This dataset was processed with prompt conversion, resulting in 318 samples.
- **sealtool** ([Seal-Tools](https://github.com/fairyshine/Seal-Tools)): Prompt conversion was applied, and only single-turn test set samples were retained, resulting in a total of 294 samples.
- **toolalpaca** ([ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca)): Textual tool definitions were converted to JSON format, and prompt conversion was applied, resulting in 114 samples.

These datasets are processed versions of the original datasets, reconstructed based on the Hammer function calling prompt (examples available at [Hammer dataset example](https://github.com/MadeAgents/Hammer/blob/main/data/masking_sft_data_example.json))

### Running the Evaluation

Run the following command to start the evaluation:

```bash eval.sh <MODEL> <DATASET>```

For example, to evaluate the Hammer1.0-3b model on the NexusRaven dataset:

```bash eval.sh /path/to/Hammer1.0-3b NexusRaven```

## Citation

If you use Hammer, please cite our paper:
```@misc{lin2024hammer,
      title={Hammer: Robust Function-Calling for On-Device Language Models via Function Masking}, 
      author={Qiqiang Lin and Muning Wen and Qiuying Peng and Guanyu Nie and Junwei Liao and Jun Wang and Xiaoyun Mo and Jiamu Zhou and Cheng Cheng and Yin Zhao and Jun Wang and Weinan Zhang},
      year={2024},
      eprint={2410.04587},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.04587 }, 
}```