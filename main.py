"""
Chromasthesia hackathon @ AGI House recreation by Erick Siavichay

Original hackers include Yuxi Liu, Ethan Goldfarb, Shawn Dimantha, and Erick Siavichay
"""

from utils.generators import generate_video, ChromasthesiaDiffuser

if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    num_frames = 30
    strength = 0.5
    scale = 7.5

    model = ChromasthesiaDiffuser(model_id=model_id)

    generate_video(
        youtube_url="https://www.youtube.com/watch?v=9ZrAYxWPN6c",
        output_path="./video_export/",
        model=model,
        num_frames=num_frames,
        strength=strength,
        guidance_scale=scale,
    )
