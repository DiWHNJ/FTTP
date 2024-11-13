# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"/data2/ljq/stable-diffusion/stable-diffusion-v1-4", 
	# use_auth_token=True
).to("cuda")

prompt = "a photo of a english springer"
with autocast("cuda"):
    # image = pipe(prompt)["sample"][0]  
    image = pipe(prompt).images
    
image[0].save("/data2/ljq/stable-diffusion/pic/astronaut_rides_horse.png")



# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "./sd-naruto-model"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "Lebron James with a hat"
# image = pipe(prompt).images[0]  
    
# image.save("result.png")
