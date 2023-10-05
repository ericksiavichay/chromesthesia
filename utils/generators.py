"""
Abstraction over diffuser for easy to use video generators
"""
from tqdm import tqdm
import torch
from torch import Tensor

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
)
from transformers import Pipeline
import os
from . import processing

os.environ["SAFETENSORS_FAST_GPU"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ChromasthesiaDiffuser:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5") -> None:
        self.models = []
        self.model_id = model_id
        if model_id == "runwayml/stable-diffusion-v1-5":
            self.models.append(
                StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    use_safetensors=True,
                )
            )
            self.models.append(
                StableDiffusionImg2ImgPipeline(**self.models[0].components)
            )
        elif model_id == "stabilityai/stable-diffusion-xl-base-1.0-with-refiner":
            base = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                safety_checker=None,
                variant="fp16",
                use_safetensors=True,
            )
            # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

            self.models.append(base)
            self.models.append(
                StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    text_encoder_2=base.text_encoder_2,
                    vae=base.vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None,
                )
            )
            self.models.append(
                StableDiffusionXLImg2ImgPipeline(**self.models[0].components)
            )
        else:
            raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self.model_id == "runwayml/stable-diffusion-v1-5":
            if "image" in kwargs:
                self.models[1].to(device)
                outputs = self.models[1](*args, **kwargs).images[0]

            else:
                self.models[0].to(device)
                outputs = self.models[0](*args, **kwargs).images[0]
        elif self.model_id == "stabilityai/stable-diffusion-xl-base-1.0-with-refiner":
            n_steps = 40
            high_noise_frac = 0.8

            if "image" in kwargs:
                self.models[2].to(device)
                base_image = self.models[2](
                    num_inference_steps=n_steps,
                    denoising_end=high_noise_frac,
                    output_type="latent",
                    *args,
                    **kwargs,
                ).images

            else:
                self.models[0].to(device)
                base_image = self.models[0](
                    num_inference_steps=n_steps,
                    output_type="latent",
                    *args,
                    **kwargs,
                ).images

            self.models[1].to(device)
            outputs = self.models[1](
                num_inference_steps=n_steps,
                denoising_start=high_noise_frac,
                image=base_image,
                prompt=kwargs["prompt"],
            ).images[0]

        return outputs


def generate_video(
    youtube_url,
    model,
    prompt,
    negative_prompt=None,
    output_path="~/video_export/",
    num_frames=300,
    fps=30,
    strength=0.5,
    guidance_scale=7.5,
):
    """
    TODO: change desc
    Generate a video from a Youtube URL (WIP)

    model should be a function that takes in the prompts and scalers and returns a diffuser
    """

    if not os.path.exists(os.path.dirname(output_path + "images/")):
        os.makedirs(os.path.dirname(output_path + "images/"))

    # if there are images from a previous run, delete them
    if os.path.exists(output_path + "images/"):
        for filename in os.listdir(output_path + "images/"):
            if filename.endswith((".png", ".mp4")):
                os.remove(output_path + "images/" + filename)

    init_image = model(prompt, negative_prompt=negative_prompt)

    # generate frames
    current_image = init_image
    for index in tqdm(range(num_frames)):
        print("Generating frame:", index)
        current_image = model(
            prompt,
            negative_prompt=negative_prompt,
            image=current_image,
            strength=strength,
            guidance_scale=guidance_scale,
        )

        current_image.save(output_path + "images/" + f"image_{index}.png")
        print("Frame saved to:", output_path + "images/" + f"image_{index}.png")

    # convert frames to video
    # processing.create_mp4_from_pngs(
    #     output_path + "images/", output_path + "video.mp4", fps=fps
    # )
