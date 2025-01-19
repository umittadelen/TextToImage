import os
from diffusers import StableDiffusionPipeline
from pathlib import Path

# Path to your model
model_path = './models/fluffyTart_v70.safetensors'

# Check if the model file exists
if not Path(model_path).exists():
    print(f"Error: Model file does not exist at the specified path: {model_path}")
else:
    print(f"Model file found at: {model_path}")

    # Attempt to load the model
    try:
        # If your model is a safetensors file, you may need to use the correct loading method.
        # For example, using Diffusers with safetensors requires safetensors support.
        pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=True)
        
        # If the model is loaded successfully, print its configuration
        print("Model loaded successfully!")
        print(f"Model config: {pipe.config}")
        
        # Optionally, generate an image to further test the pipeline
        prompt = "A futuristic cityscape"
        image = pipe(prompt).images[0]
        image.show()  # Display the image

    except Exception as e:
        print(f"Error loading model: {e}")
