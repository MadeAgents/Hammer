# Hammer

The source code and dataset mentioned in the paper [**Hammer: Robust Function-Calling for On-Device Language Models via Function Masking**](https://arxiv.org/pdf/2410.04587).

## Overview
**Hammer** is a series of lightweight language models with strong **function calling** capabilities, enabling developers to create personalized, on-device agentic applications. We have released several models based on **Function Masking** techniques discussed in the paper. These models are available on [MadeAgents on Hugging Face](https://huggingface.co/MadeAgents).

## Fine-Tuning

### Install Dependencies

You should install dependencies using the following command:

```
pip install -r requirements.txt
```

### Data Processing

Download the datasets [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and [`MadeAgents/XLAM-7.5k-Irrelevance`](https://huggingface.co/datasets/MadeAgents/XLAM-7.5k-Irrelevance) and place them in the `data` directory. Simply run the command below to prepare the training data:

```
python data_processing.py
```

### Train the Model

After setting up the training data, you can now train the model. Replace `\<MODEL\>` with the path or name of the base model you want to use:

```
bash train.sh <MODEL>
```

## Evaluation

We conduct a comprehensive evaluation of the performance of the model on tool use leaderboards such as [Berkley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html), [API-Bank](https://arxiv.org/abs/2304.08244), [Tool-Alpaca](https://arxiv.org/abs/2306.05301), [Nexus Raven](https://github.com/nexusflowai/NexusRaven-V2) and [Seal-Tools](https://arxiv.org/abs/2405.08355). For the evaluation code of the BFCL leaderboard, please directly refer to the official documentation. Other evaluation sets present minor issues such as inconsistent formats and errors in labels. We have made appropriate processing, including format conversion and removal of error samples. Specifically:

- **apibank_l1** ([API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank/tree/main)): Only the format of the data has been converted, resulting in 399 samples.
- **apibank_l2** ([API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank/tree/main)): 8 samples for which the ground truth function is not in the candidate function list are filtered out, and the data format is converted, resulting in 127 samples.
- **NexusRaven** ([NexusRaven](https://huggingface.co/datasets/Nexusflow/Function_Call_Definitions)): Only the format of the data has been converted, resulting in 318 samples.
- **sealtool** ([Seal-Tools](https://github.com/fairyshine/Seal-Tools)): Only single-turn test data is considered, and the data format is converted, resulting in 294 samples.
- **toolalpaca** ([ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca)): Textual tool definitions were converted to JSON format, and prompt conversion was applied, resulting in 114 samples.

The processed evaluation datasets are placed under the `data` directory, and are all in the Hammer function calling prompt format (examples available at [Hammer dataset example](https://github.com/MadeAgents/Hammer/blob/main/data/masking_sft_data_example.json))

### Evaluate the Model

Use the following command for LLM inference of the specific dataset with specific models:

```
bash eval.sh <MODEL> <DATASET>
```

For instance, to evaluate the Hammer1.0-3b model on the NexusRaven dataset:

```
bash eval.sh /path/to/Hammer1.0-3b NexusRaven
```


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