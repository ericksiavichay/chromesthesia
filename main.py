"""
Chromasthesia hackathon @ AGI House recreation by Erick Siavichay

Original hackers include Yuxi Liu, Ethan Goldfarb, Shawn Dimantha, and Erick Siavichay
"""

from utils.generators import generate_video

if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    num_frames = 30
    strength = 0.5
    scale = 7.5

    generate_video(
        youtube_url="https://www.youtube.com/watch?v=9ZrAYxWPN6c",
        model_id=model_id,
        num_frames=num_frames,
        strength=strength,
        scale=scale,
    )
