import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from collections import OrderedDict
#repo_id = "Kijai/flux-fp8"
#filename_fp8 = "flux1-dev-fp8.safetensors"
#filename_schnell_fp8 = "flux1-schnell-fp8.safetensors"
#fp8_path = hf_hub_download(repo_id=repo_id, filename=filename_fp8)
#schnell_fp8_path = hf_hub_download(repo_id=repo_id, filename=filename_schnell_fp8)


def load_quantized_model(model_dir, quantized_model_path):
    pipe = FluxPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, use_safetensors=True)
    print("Base model loaded successfully with safetensors")

    quantized_state_dict = load_file(quantized_model_path)
    print("Quantized weights loaded successfully")

    # Remove 'model.' prefix if present
    new_state_dict = OrderedDict((k.replace('model.', ''), v) for k, v in quantized_state_dict.items())

    # Apply quantized weights
    pipe.lora_state_dict(new_state_dict, strict=False)
    print("Quantized weights applied successfully")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe

def generate_image(pipe, prompt, height, width):
    try:
        with torch.no_grad():
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=7.5,
                num_inference_steps=30,
                max_sequence_length=256
            ).images[0]
        return image
    except RuntimeError as e:
        print(f"Error during image generation: {e}")
        return None

# Main execution
if __name__ == "__main__":
    quantized_model_path = "flux1-schnell-fp8.safetensors"
    model_dir = "models/FLUX.1-schnell"

    pipe = load_quantized_model(model_dir, quantized_model_path)

    prompt = "dancing black and white four men and women with ethnic masks, dressed very professionally. These should be looking straight from the photo."
    image = generate_image(pipe, prompt, height=3840, width=2160)

    if image:
        image.save("image.png")
        print("Image generated and saved successfully")
    else:
        print("Failed to generate image")