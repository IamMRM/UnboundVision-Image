import torch
from diffusers import FluxPipeline
import gradio as gr

model_dir = "models/FLUX.1-schnell"
#pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
#pipe.save_pretrained(model_dir)
pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16)
def generate_image(prompt, guidance_scale, height, width, num_inference_steps):
    out = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        max_sequence_length=256,
    ).images[0]
    return out

interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(lines=2, label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=0, maximum=20, step=0.5, value=7.5, label="Guidance Scale"),
        gr.Slider(minimum=256, maximum=1024, step=64, value=768, label="Height"),
        gr.Slider(minimum=256, maximum=2048, step=64, value=1360, label="Width"),
        gr.Slider(minimum=1, maximum=100, step=1, value=2, label="Number of Inference Steps")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Image Generation with FluxPipeline",
    description="Generate images based on a text prompt using the FluxPipeline model.",
)


interface.launch()
