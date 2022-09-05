import torch
from torch import autocast
import requests
from PIL import Image
from io import BytesIO
from glob import glob
import sys

# dronesynth_images = glob("photos/generatedDatarandrotate_7/trainingGrey/images/*")

from image_to_image import StableDiffusionImg2ImgPipeline, preprocess

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True).to(device)

# # let's download an initial image
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# ID = 5
# name = dronesynth_images[ID].split("/")[-1].split(".")[0]
input_file = sys.argv[1]
name = input_file.split(".")[0]
strength = 0.5
guidance_scale = 7.0
init_image = Image.open(input_file).convert("RGB")
init_image = init_image.resize((768, 512))
init_image.save(name + "_orig.png")
init_image = preprocess(init_image)

generic_prompt = "high resolution photograph"

if len(sys.argv) == 4:
    prompt = sys.argv[3]
    print("Using prompt", prompt)
else:
    prompt = generic_prompt
# prompt = "high resolution photograph of a jet airplane on the runway."

with autocast("cuda"):

    for strength_level in range(3,10):
        strength = strength_level/10.
        images = pipe(prompt=prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale)["sample"]
        images[0].save("samples/generated_{}_str{}_gs{}.png".format(name, strength, guidance_scale))

    # for strength_level in range(3,10):
    #     strength = strength_level/10.
    #     images = pipe(prompt=generic_prompt, init_image=init_image, strength=strength, guidance_scale=guidance_scale)["sample"]
    #     images[0].save("samples/generic_prompt_{}_str{}_gs{}.png".format(name, strength, guidance_scale))
