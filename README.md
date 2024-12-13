<div align="center">
  <img width="600px" height="auto" src="./images/hammer.png">
</div>

<p align="center">
  <a href="https://arxiv.org/pdf/2410.04587">Paper</a> |
  <a href="https://huggingface.co/MadeAgents">Model</a> |
    <a href="https://github.com/MadeAgents/Hammer/tree/main?tab=readme-ov-file#Usage">Usage</a> |
  <a href="https://github.com/MadeAgents/Hammer/tree/main?tab=readme-ov-file#Fine-Tuning">Fine-Tuning</a> |
  <a href="https://github.com/MadeAgents/Hammer/tree/main?tab=readme-ov-file#Evaluation">Evaluation</a> |

</p>

---

## 🎉 News
- **[12.2024]**: We are excited to announce the release of [Hammer2.1](https://huggingface.co/collections/MadeAgents/hammer21-675a97053753e8fa70a3f0ac), our suite of Large Action Models! These models have achieved impressive rankingson the [Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html#leaderboard).
- **[10.2024]**: We're excited to release lightweight [Hammer 2.0 models](https://huggingface.co/collections/MadeAgents/hammer20-66f4dee539f7b2c95224012a) (0.5B , 1.5B , 3B , and 7B) with strong function calling capability, which empower developers to build personalized, on-device agentic applications.
- **[10.2024]**: We have now made our code and accompanying paper for [**Hammer: Robust Function-Calling for On-Device Language Models via Function Masking**](https://arxiv.org/pdf/2410.04587) publicly available.
- **[09.2024]**: [Hammer model](https://huggingface.co/collections/MadeAgents/hammer10-66f4ddaad0ad3a5e3acd24f5) is released! Focusing on on-device applications, we release a number of models from 1.5B, 4B to 7B parameters.



---
# Overview
**Hammer** is a series of lightweight language models with strong **function calling** capabilities, enabling developers to create personalized, on-device agentic applications. We have released several models based on **Function Masking** techniques discussed in the paper. These models are available on [MadeAgents on Hugging Face](https://huggingface.co/MadeAgents).



# Usage
Hammer models offer flexibility in deployment and usage, fully supporting both **vLLM** deployment and **Hugging Face Transformers** tool calling. Below are the specifics on how to make use of these features:

## Using vLLM
### Option 1: Using Hammer client (Recommended)
Before using vLLM, first clone the Hammer code repository and change directory to the 'Hammer':
```
git clone https://github.com/MadeAgents/Hammer.git
cd Hammer
```
vLLM offers efficient serving with lower latency. To serve the model with vLLM:
```
vllm serve MadeAgents/Hammer2.1-1.5b --host 0.0.0.0 --port 8000 --tensor-parallel-size 1
```
Once the model is served, you can use the following Hammer client to interact with it for function calling:
~~~
from client import HammerChatCompletion,HammerConfig
config = HammerConfig(base_url="http://localhost:8000/v1/", model="MadeAgents/Hammer2.1-1.5b")
llm = HammerChatCompletion.from_config(config)

# Example conversation
messages = [
    {"role": "user", "content": "What's the weather like in New York?"},
    {"role": "assistant","content": '```\n{"name": "get_weather", "arguments": {"location": "New York, NY ", "unit": "celsius"}\n```'},
    {"role": "tool", "name": "get_weather", "content": '{"temperature": 72, "description": "Partly cloudy"}'},
    {"role": "user", "content": "Now, search for the weather in San Francisco."}
]

# Example function definition (optional)
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "respond",
        "description": "When you are ready to respond, use this function. This function allows the assistant to formulate and deliver appropriate replies based on the input message and the context of the conversation. Generate a concise response for simple questions, and a more detailed response for complex questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The content of the message to respond to."}
            },
            "required": ["message"]
        }
    }
]

response = llm.completion(messages, tools=tools)
print(response)
~~~


### Option 2: Using vLLM’s built-in tool calling
Hammer2.1 supports vllm’s built-in tool calling. This functionality requires vllm>=0.6. If you want to enable this functionality, please start vllm’s OpenAI-compatible service with:
~~~
vllm serve MadeAgents/Hammer2.1-1.5b --enable-auto-tool-choice --tool-call-parser hermes
~~~
And then use it in the same way you use GPT’s tool calling:
~~~
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                        "default": "celsius"
                    },
                },
                "required": ["location","format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                        "default": "celsius"
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                        "default": 1
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]


from openai import OpenAI
openai_api_key = "None"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

query = """What's the weather like today in San Francisco"""

chat_response = client.chat.completions.create(
    model="MadeAgents/Hammer2.1-1.5b",
    messages=[
        {"role": "user", "content": query},],
    tools = tools,
    temperature=0
)
print(chat_response.choices[0].message.content)
~~~


## Using Hugging Face Transformers
Hammer2.1’s chat template also includes a tool calling template, meaning that you can use Hugging Face transformers’ tool calling support. This is a simple example of how to use our model using Transformers.
~~~
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("MadeAgents/Hammer2.1-1.5b")
model = AutoModelForCausalLM.from_pretrained("MadeAgents/Hammer2.1-1.5b", torch_dtype=torch.bfloat16, device_map="auto")

# Example conversation
messages = [
    {"role": "user", "content": "What's the weather like in New York?"},
    {"role": "assistant","content": '```\n{"name": "get_weather", "arguments": {"location": "New York, NY ", "unit": "celsius"}\n```'},
    {"role": "tool", "name": "get_weather", "content": '{"temperature": 72, "description": "Partly cloudy"}'},
    {"role": "user", "content": "Now, search for the weather in San Francisco."}
]

# Example function definition (optional)
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "respond",
        "description": "When you are ready to respond, use this function. This function allows the assistant to formulate and deliver appropriate replies based on the input message and the context of the conversation. Generate a concise response for simple questions, and a more detailed response for complex questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The content of the message to respond to."}
            },
            "required": ["message"]
        }
    }
]

inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
~~~





# Fine-Tuning

### Install Dependencies

You should install dependencies using the following command:

```
pip install -r requirements.txt
```

### Data Processing

Download the datasets [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) and [`MadeAgents/xlam-irrelevance-7.5k`](https://huggingface.co/datasets/MadeAgents/xlam-irrelevance-7.5k) and place them in the `data/train` directory. Simply run the command below to prepare the training data:

```
python train/data_processing.py
```

### Train the Model

After setting up the training data, you can now train the model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Replace `<MODEL>` with the path or name of the base model you want to use:

```
bash scripts/train.sh <MODEL>
```

# Evaluation

We conduct a comprehensive evaluation of the performance of the model on tool use leaderboards such as [Berkley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html), [API-Bank](https://arxiv.org/abs/2304.08244), [Tool-Alpaca](https://arxiv.org/abs/2306.05301), [Nexus Raven](https://github.com/nexusflowai/NexusRaven-V2) and [Seal-Tools](https://arxiv.org/abs/2405.08355). For the evaluation code of the BFCL leaderboard, please directly refer to the official documentation. Other evaluation sets present minor issues such as inconsistent formats and errors in labels. We have made appropriate processing, including format conversion and removal of error samples. Specifically:

- **apibank_l1** ([API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank/tree/main)): Only the format of the data has been converted, resulting in 399 samples.
- **apibank_l2** ([API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank/tree/main)): 8 samples for which the ground truth function is not in the candidate function list are filtered out, and the data format is converted, resulting in 127 samples.
- **NexusRaven** ([NexusRaven](https://huggingface.co/datasets/Nexusflow/Function_Call_Definitions)): Only the format of the data has been converted, resulting in 318 samples.
- **sealtool** ([Seal-Tools](https://github.com/fairyshine/Seal-Tools)): Only single-turn test data is considered, and the data format is converted, resulting in 294 samples.
- **toolalpaca** ([ToolAlpaca](https://github.com/tangqiaoyu/ToolAlpaca)): Textual tool definitions were converted to JSON format, and prompt conversion was applied, resulting in 114 samples.

The processed evaluation datasets are placed under the `data/train` directory, and are all in the Hammer function calling prompt format (examples available at [Hammer dataset example](https://github.com/MadeAgents/Hammer/blob/main/data/train/masking_sft_data_example.json))

### Evaluate Hammer Model

Use the following command for LLM inference of the specific dataset with specific models:

```
bash scripts/eval.sh <MODEL> <DATASET>
```

For instance, to evaluate the Hammer2.1-7b model on the NexusRaven dataset:

```
bash scripts/eval.sh /path/to/Hammer2.1-7b NexusRaven
```

### Evaluate Other Models

If you want to test the performance of other models, you can obtain the original datasets from the [`data/test/original`](https://github.com/MadeAgents/Hammer/blob/main/data/test/original) directory. Use the model you wish to test to perform inference, generating a JSONL file that stores the JSON results, which should contain `label` and `predict` fields. You can refer to the format in [`data/examples_eval.jsonl`](https://github.com/MadeAgents/Hammer/blob/main/data/example_eval.jsonl). Finally, run the evaluation script with the following command:
```
python evaluation/evaluate.py <outputs_dir>
```

# Licenses
This code is licensed under cc-by-4.0.

# Citation

If you use Hammer, please cite our paper:
```
@misc{lin2024hammer,
      title={Hammer: Robust Function-Calling for On-Device Language Models via Function Masking}, 
      author={Qiqiang Lin and Muning Wen and Qiuying Peng and Guanyu Nie and Junwei Liao and Jun Wang and Xiaoyun Mo and Jiamu Zhou and Cheng Cheng and Yin Zhao and Jun Wang and Weinan Zhang},
      year={2024},
      eprint={2410.04587},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.04587 }, 
}
```