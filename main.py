import torch
from diffusers import FluxPipeline
import gradio as gr

# Load the model from the local directory and move to GPU
model_dir = "models/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
pipe.to("cuda")  # Move the model to GPU

# Function to generate images
def generate_images(prompts, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256, seed=0):
    images = []
    generator = torch.Generator("cuda").manual_seed(seed)  # Use GPU for generation
    for prompt in prompts.split("\n"):
        image = pipe(
            prompt.strip(),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            generator=generator
        ).images[0]
        images.append(image)
    return images

# Gradio interface
interface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter prompts, one per line"),
        gr.Slider(0, 10, value=0.0, step=0.1, label="Guidance Scale"),
        gr.Slider(1, 100, value=4, step=1, label="Number of Inference Steps"),
        gr.Slider(10, 512, value=256, step=1, label="Max Sequence Length"),
        gr.Slider(0, 10000, value=0, step=1, label="Random Seed")
    ],
    outputs=gr.Gallery(label="Generated Images"),
    title="Flux Schnell Image Generator"
)

# Launch the interface
interface.launch()
