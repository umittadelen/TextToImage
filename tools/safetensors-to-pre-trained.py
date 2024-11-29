from diffusers import StableDiffusionXLPipeline
from pathlib import Path

# Load the model from the safetensors file (or the format you're using)
model_name ="./tools/kiwimixXL_v3.safetensors"

pipe = StableDiffusionXLPipeline.from_single_file(model_name)

pipe.save_pretrained(Path(model_name).stem)