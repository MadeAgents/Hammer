# Hammer

Welcome to the Hammer repository! This is the official implementation of our method introduced in the paper **"Robust Function-Calling for On-Device Language Models via Function Masking"**. The Hammer series models enhance robust function-calling capabilities in language models, making them suitable for deployment on resource-constrained devices.

## Overview

The Hammer series employs function masking to selectively focus on specific functions during inference, improving the robustness and efficiency of function-calling. These methods are particularly valuable for on-device language processing applications where computational resources are restricted.

For an in-depth discussion on Hammer and the validation experiments, please refer to our [paper](https://huggingface.co/MadeAgents).

## Models

We have released several models in the Hammer series, fine-tuned for various applications. These models are available on Hugging Face:

### Hammer1.0 Series
- [Hammer1.0-1.5b](https://huggingface.co/MadeAgents/Hammer1.0-1.5b): A 1.5-billion parameter language model.
- Hammer1.0-3b: A 3-billion parameter variant.
- Hammer1.0-7b: A 7-billion parameter variant.

### Hammer2.0 Series
- Hammer2.0-0.5b: A 0.5-billion parameter model.
- Hammer2.0-1.5b: A 1.5-billion parameter model.
- Hammer2.0-3b: A 3-billion parameter model.
- [Hammer2.0-7b](https://huggingface.co/MadeAgents/Hammer2.0-7b): A 7-billion parameter model.

And many more to come as we continue to expand the Hammer series.

## Upcoming Features

We are preparing to release the fine-tuning data and the complete codebase soon. This will include:

- Detailed tuning data used for fine-tuning the Hammer models.
- Scripts and tools to replicate our experiments and fine-tune your models using the Hammer method.

Stay tuned to this repository for updates!
