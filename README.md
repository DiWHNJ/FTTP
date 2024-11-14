<span style="font-size: 25px;">FTTP Code Repository</span>

This repository contains the code for the paper **Forget the Token and Pixel: Rethinking Gradient Ascent for Concept Unlearning in Multimodal Generative Models**. The implementation is organized into two main directories, each representing the application of our method on LLaVA and different models.

LLAVA
Training Script: scripts/v1_5/finetune_lora.sh
This script is used for fine-tuning the LLAVA model using the LoRA with our method.

Diffusion Model
Training Script: stable-diffusion/train-scripts/gradient_ascent.py
This script implements the gradient ascent method for training the diffusion model.


