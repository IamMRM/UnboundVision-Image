import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def load_model(model_dir):

    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, use_safetensors=True, device_map="balanced")
    # #pipe.save_pretrained("models/FLUX.1-dev")
    # print("Base model loaded successfully with safetensors")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("moved models to gpus")
    return pipe

def load_quantized_model(model_dir, quantized_model_path):
    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, use_safetensors=True)#, device_map="balanced")
    print("Base model loaded successfully with safetensors")

    quantized_state_dict = load_file(quantized_model_path)
    print("Quantized weights loaded successfully")

    # Remove 'model.' prefix if present
    new_state_dict = OrderedDict((k.replace('model.', ''), v) for k, v in quantized_state_dict.items())

    # Apply quantized weights
    pipe.lora_state_dict(new_state_dict, strict=False)
    print("Quantized weights applied successfully")
    #pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe

def generate_image(pipe, prompt, height=256, width=256):
    try:
        with torch.no_grad():
            torch.cuda.synchronize()
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=7.5,
                num_inference_steps=20,  # Reduced steps
                max_sequence_length=256
            ).images[0]
        return image
    except RuntimeError as e:
        print(f"Error during image generation: {e}")
        return None

# Main execution
if __name__ == "__main__":
    quantized_model_path = "flux1-schnell-fp8.safetensors"
    model_dir = "models/FLUX.1-dev"
    #pipe = load_quantized_model(model_dir, quantized_model_path)

    pipe = load_model(model_dir)
    print(pipe.hf_device_map)
    prompt = "dancing people in a festival"
    image = generate_image(pipe, prompt)
    if image:
        image.save("image333.png")
        print("Image generated and saved successfully")
    else:
        print("Failed to save generated image")
