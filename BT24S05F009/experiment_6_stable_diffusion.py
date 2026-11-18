"""
Experiment 6: Generate images using the Stable Diffusion text-to-image model.

Run:
    pip install -U torch diffusers transformers accelerate safetensors
    python experiment_6_stable_diffusion.py

Requires internet on first run to download the model. A CUDA-capable GPU is strongly
recommended. Change MODEL_ID only to a model you are licensed/allowed to use.
"""

import torch
from diffusers import StableDiffusionPipeline

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"

def main():
    prompt = input("Enter an image prompt: ").strip()
    if not prompt:
        prompt = "A beautiful futuristic city at sunset, detailed digital art"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)

    image = pipe(
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    image.save("stable_diffusion_output.png")
    print("Image saved as stable_diffusion_output.png")

if __name__ == "__main__":
    main()
