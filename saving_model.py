from diffusers import FluxPipeline
import torch

# Specify the local directory to save the model
model_dir = "models/FLUX.1-schnell"

#pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload()
#model_dir = "models/FLUX.1-dev"
#pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16) # can replace schnell with dev
# to run on low vram GPUs (i.e. between 4 and 32 GB VRAM)
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

pipe.to(torch.float16)
# Save the model to the specified directory
pipe.save_pretrained(model_dir)