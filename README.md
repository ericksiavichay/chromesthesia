# Chromesthesia

1st place hackthon project at the AGI house. Contributors: Erick Siavichay, Ethan Goldfarb, Yuxi Liu, and Shawn Dimantha

## Settings you can play with

### model_id

The name of the model you want to use to generate the frames. Different models are optimized to generate in a specific manner, so try out different ones or consider building your own and updating the repo.

Currently supported models:

- runwayml/stable-diffusion-v1-5
- stabilityai/stable-diffusion-xl-base-1.0-with-refiner

### num_frames

Number of frames to generate

### strength

From @runwayml: "strength is a value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input". If you want the video to take more creative risks in terms of the semantic similarity to the input image, increase this value. If you want the video to be more consistent with the input image, decrease this value. Start with 0.5 as a default.

### guidance_scale

This value determines how strong the images should adhere to the prompts. If you have a very descriptive prompt, you might want to try out higher scales such as > 8. If you want the images to mostly generally follow the prompts but also want to allow for introduction of new concepts in the scene, try out smaller values. Suggested to be between 7-9.

### fps

The FPS of the output video. Suggested between 12-24. Higher values tend to make the transitions really fast.

### prompt

What you want in the scene. Take a look at stable diffusion prompting guides

### negative_prompt

What you don't want in the scene. Use this after generating a few images to remove things you notice that are unappealing.

### init_image

If you want to start with a specific image, you can upload it here. Otherwise, the model will start with a random image.

Linked HuggingFace 🤗 spaces:

- [Audio Track(song) to Stable Diffusion prompts (v1)](https://huggingface.co/spaces/shawndimantha/transcribesong1)
