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