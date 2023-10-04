"""
Abstraction over diffuser for easy to use video generators
"""
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import os
from . import processing

os.environ["SAFETENSORS_FAST_GPU"] = "1"


def generate_video(
    youtube_url,
    model_id,
    output_path="/video_export/",
    num_frames=300,
    fps=30,
    strength=0.5,
    scale=7.5,
):
    """
    Generate a video from a Youtube URL
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_id.endswith(".safetensors"):
        model_text_to_img = StableDiffusionPipeline.from_single_file(
            model_id, torch_dtype=torch.float16
        )
    else:
        model_text_to_img = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )

    model_text_to_img.to(device)  # Move model to GPU

    song_lyrics = [
        "Golden beams pierce the canopy, as night begins to fall,",
        "Whispers of ancient trees, I can hear them call.",
        "The path ahead, bathed in a twilight glow,",
        "Into the heart of the forest, there's so much I want to know.",
        "Moonlight filters through the leaves, casting shadows on the ground,",
        "Every step I take, nature's symphony surrounds.",
        "Creatures of the night, begin their serenade,",
        "In this enchanted forest, magic is displayed.",
        "The river's song, a gentle lullaby,",
        "Reflecting stars from the deep blue sky.",
        "Fireflies dance, a spectacle of light,",
        "Guiding me through the forest, in the heart of the night.",
        "Golden hour's embrace, a moment so profound,",
        "In this nighttime forest, peace is all around.",
        "Journeying through, with every breath I take,",
        "I feel the world's wonder, and my spirit awakes.",
        "The night's embrace, so tender and so sweet,",
        "In this forest, where dreams and reality meet.",
        "As dawn approaches, and the golden hour ends,",
        "I'll cherish this journey, with the forest as my friend.",
    ]

    style_prompt = (
        "watercolor, cyberpunk, serious, gloomy, HD, 4K, neon-lit, dark yet cheerful"
    )
    negative_prompt = "canvas frame, cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render"

    prompt = song_lyrics[0] + f", {style_prompt}"
    init_image = model_text_to_img(prompt, negative_prompt=negative_prompt).images[0]

    model_img_to_img = StableDiffusionImg2ImgPipeline(
        **model_text_to_img.components
    ).to(device)

    # generate frames
    current_image = init_image
    index = 0
    semantic_size = num_frames // len(song_lyrics)
    for lyric in song_lyrics:
        prompt = f"{lyric}, {style_prompt}"
        for _ in tqdm(range(semantic_size)):
            current_image = model_img_to_img(
                prompt,
                negative_prompt=negative_prompt,
                image=current_image,
                strength=strength,
                guidance_scale=scale,
            ).images[0]
            if not os.path.exists(os.path.dirname(output_path + "images/")):
                os.makedirs(os.path.dirname(output_path + "images/"))
            current_image.save(output_path + "images/" + f"image_{index}")
            index += 1

    # convert frames to video
    processing.create_mp4_from_pngs(
        output_path + "images/", output_path + "video.mp4", fps=fps
    )
