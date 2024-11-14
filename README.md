# FTTP Code Repository

This repository contains the code for the paper **Forget the Token and Pixel: Rethinking Gradient Ascent for Concept Unlearning in Multimodal Generative Models**. The implementation is organized into two main directories, each representing the application of our method on LLaVA and different models.

## LLAVA<br>
### Training Script: 
```bash
bash scripts/v1_5/finetune_lora.sh
```
This script is used for fine-tuning the LLAVA model using the LoRA with our method.

### Evaluation Script: 
```bash
python scripts/v1_5/finetune_lora.sh
```

## Diffusion Model<br>
### Training Script: 
```bash
bash stable-diffusion/train-scripts/gradient_ascent.py
```
This script implements the gradient ascent method for training the diffusion model.

### Evaluation Script: 
```bash
python stable-diffusion/train-scripts/
```

## Environment
The environment configuration for this experiment is consistent with the original environments used for LLAVA and Stable Diffusion.



