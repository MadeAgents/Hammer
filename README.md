# Hammer

Welcome to the Hammer repository! This is the official implementation of our method introduced in the paper **"Robust Function-Calling for On-Device Language Models via Function Masking"**. The Hammer series models enhance the function-calling capabilities of language models, making them ideal for deployment on resource-constrained devices and empowering developers to create personalized, on-device applications


## Overview

The Hammer series employs function masking to selectively focus on specific functions during inference, improving the robustness and efficiency of function-calling. These methods are especially valuable for on-device language processing applications where computational resources are limited.

For an in-depth discussion on Hammer and the validation experiments, please refer to our [paper](https://arxiv.org/pdf/2410.04587).

## Models

We have released several models in the Hammer series, fine-tuned for various applications. These models are available on Hugging Face:

### Hammer1.0 Series
- [Hammer1.0-1.5b](https://huggingface.co/MadeAgents/Hammer1.0-1.5b)
- [Hammer1.0-3b](https://huggingface.co/MadeAgents/Hammer1.0-3b)
- [Hammer1.0-7b](https://huggingface.co/MadeAgents/Hammer1.0-7b)

### Hammer2.0 Series
- [Hammer2.0-0.5b](https://huggingface.co/MadeAgents/Hammer2.0-0.5b)
- [Hammer2.0-1.5b](https://huggingface.co/MadeAgents/Hammer2.0-1.5b)
- [Hammer2.0-3b](https://huggingface.co/MadeAgents/Hammer2.0-3b)
- [Hammer2.0-7b](https://huggingface.co/MadeAgents/Hammer2.0-7b)

And many more to come as we continue to expand the Hammer series.

## Fine-Tuning

### Install Dependencies

Install required packages by running the following command:
```pip install -r requirements.txt```

### Data Processing
Based on the original data of 'xlam_function_calling_60k' and 'xlam-7.5k-irrelevancek', run the code to process the data, including function masking processing and generating corresponding SFT data.
```python data_processing.py```

### Train the Model
```bash train.sh```
In the 'train.sh' file, 'MODEL' represents the path of the base model. You can choose a base model from the Qwen series. 'OUTPUT_DIR' refers to the target path for saving the LoRA adapter.



## Evaluation
To evaluate a model on a specific dataset, you can run the evaluation script 'eval.sh'. The script requires two parameters: MODEL (the path to the model) and DATASET (the dataset to be used for evaluation).

### Available Datasets
- apibank_l1
- apibank_l2
- NexusRaven
- sealtool
- toolalpaca

### Running the Evaluation
Run the following command to start the evaluation:

```bash eval.sh <MODEL> <DATASET>```

For example, to evaluate the Hammer1.0-3b model on the NexusRaven dataset:

```bash eval.sh /path/to/Hammer1.0-3b NexusRaven```
