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
)
from transformers import Pipeline
import os
from . import processing

os.environ["SAFETENSORS_FAST_GPU"] = "1"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ChromasthesiaDiffuser(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        _forward_kwargs = {}
        postprocess_kwargs = {}
        if "image" in kwargs:
            _forward_kwargs["image"] = kwargs["image"]
        return preprocess_kwargs, _forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs):
        # Diffuser will consist of the text to image and also an image to image pipeline
        self._set_models(inputs["model"])
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        if model_inputs["image"] is not None:
            self.main_model.to(device)
            outputs = self.main_model(**model_inputs).images[0]

        else:
            self.init_model.to(device)
            outputs = self.init_model(**model_inputs).images[0]
        return outputs

    def postprocess(self, model_outputs):
        pass

    def _set_models(self, model="runwayml/stable-diffusion-v1-5"):
        """
        Function to create the custome diffuser pipeline. Create a new ID here
        and construct your own. The init model is just a text to image and the
        main model is just an image to image.
        """
        if model == "runwayml/stable-diffusion-v1-5":
            self.init_model = StableDiffusionPipeline.from_pretrained(
                model, torch_dtype=torch.float16, safety_checker=None
            )
            self.main_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                **self.init_model.components
            )


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
