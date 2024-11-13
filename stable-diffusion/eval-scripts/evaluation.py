import argparse
import os

import pandas as pd
import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

def generate_images(
    model_name,
    prompts_path,
    save_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
    from_case=0,
):
    """
    Function to generate images from diffusers code

    The program requires the prompts to be in a csv format with headers
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)

    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    """

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(
        "/data2/ljq/stable-diffusion/stable-diffusion-v1-4", subfolder="vae"
    )
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(
        "/data2/ljq/stable-diffusion/stable-diffusion-v1-4", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained("/data2/ljq/stable-diffusion/stable-diffusion-v1-4/text_encoder")
    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        "/data2/ljq/stable-diffusion/stable-diffusion-v1-4", subfolder="unet"
    )
    
    if "SD" not in model_name:
        try:
            # model_path = (
            #     f'models/{model_name}/{model_name.replace("compvis","diffusers")}.pt'
            # )
            model_path = model_name
            unet.load_state_dict(torch.load(model_path))
        except Exception as e:
            print(
                f"Model path is not valid, please check the file name and structure: {e}"
            )
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    df = pd.read_csv(prompts_path)
    prompt = ["A image of Elon Musk"]

    height = 512 # 稳定扩散的默认高度
    width = 512 # 稳定扩散的默认宽度

    num_inference_steps = 100 # 去噪步骤数
    folder_path = f"{save_path}/{model_name}"
    os.makedirs(folder_path, exist_ok=True)
    guidance_scale = 7.5 # 分类器自由指导的比例

    generator = torch.manual_seed(0) # 用于创建初始潜在噪声的种子生成器

    batch_size = len(prompt)
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
    # 如果我们正在进行分类器自由指导，则扩展潜在变量，以避免进行两次前向传递。
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        # 预测噪声残差
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # 进行分类器自由指导
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 计算去噪图像的隐空间表示
        latents = scheduler.step(noise_pred, t, latents).prev_sample
# 将潜在变量缩放回去。
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    pil_images[0].save(f"{folder_path}/2223.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--model_name", help="name of model", type=str, default="/data1/dmz/diffusion_model/狗-有轮廓compvis-ga-method_full-alpha_0.1-epoch_20-lr_1e-05/diffusers-ga-method_full-alpha_0.1-epoch_20-lr_1e-05.pt", required=False)
    parser.add_argument(
        "--prompts_path", help="path to csv file with prompts", type=str, default="prompts/imagenette.csv", required=False
    )
    parser.add_argument(
        "--save_path", help="folder where to save images", type=str, default="evaluation_folder76/" ,required=False
    )
    parser.add_argument(
        "--device",
        help="cuda device to run on",
        type=str,
        required=False,
        default="cuda:2",
    )
    parser.add_argument(
        "--guidance_scale",
        help="guidance to run eval",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=128,
    )
    parser.add_argument(
        "--from_case",
        help="continue generating from case_number",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples per prompt",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=100,
    )
    args = parser.parse_args()

    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples = args.num_samples
    from_case = args.from_case

    generate_images(
        model_name,
        prompts_path,
        save_path,
        device=device,
        guidance_scale=guidance_scale,
        image_size=image_size,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        from_case=from_case,
    )
