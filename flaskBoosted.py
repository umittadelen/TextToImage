#*  ╔════════════════════════════════════════════════════════════════════════════╗
#*  ║                    Download And Install Required Libraries                 ║
#*  ╚════════════════════════════════════════════════════════════════════════════╝

import subprocess
import sys

# List of required packages and their import names
required_packages = {"flask": "flask","torch": "torch","transformers": "transformers","diffusers": "diffusers"}

# Install missing packages
for package, import_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Now import the required libraries
from flask import Flask, render_template, request, send_file, jsonify
import torch
import random
import os
import tempfile
from transformers import CLIPTokenizer
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

#*  ╔════════════════════════════════════════════════════════════════════════════╗
#*  ║                                Main code                                   ║
#*  ╚════════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__)

#load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "Linaqruf/animagine-xl-3.0",
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

# Create a temporary directory for images
temp_dir = tempfile.mkdtemp()

# Dictionary to store generated images
generated_images = {}

def preprocess_prompt(prompt):
    # Tokenize the prompt
    tokens = tokenizer(prompt)["input_ids"]
    if len(tokens) > 77:
        # Truncate to the first 77 tokens
        prompt = tokenizer.decode(tokens[:77])
    return prompt

# Route for generating images
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    negative_prompt = request.form['negative_prompt']
    width = int(request.form.get('width', 832))  # Default width
    height = int(request.form.get('height', 1216))  # Default height
    
    # Preprocess prompts
    prompt = preprocess_prompt(prompt)
    negative_prompt = preprocess_prompt(negative_prompt)

    # Generate images with seeds
    seeds = [random.randint(0, 100000000000) for _ in range(4)]
    images = []
    for seed in seeds:
        image_path, seed = generate_image_with_seed(prompt, negative_prompt, seed, width, height)
        generated_images[seed] = image_path
        images.append({
            'img': f"/temp/{os.path.basename(image_path)}",
            'seed': seed
        })

    return jsonify(images=images)

# Modify the image generation function to accept width and height
def generate_image_with_seed(prompt, negative_prompt, seed, width, height):
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=7,
        num_inference_steps=28,
        generator=generator
    ).images[0]
    
    # Save the image to the temporary directory
    image_path = os.path.join(temp_dir, f'image_{seed}.png')
    image.save(image_path, 'PNG')
    
    return image_path, seed

# Route for downloading an image
@app.route('/download/<int:seed>', methods=['GET'])
def download(seed):
    # Get the stored image path
    image_path = generated_images.get(seed)
    
    if image_path and os.path.exists(image_path):
        return send_file(
            image_path, 
            mimetype='image/png', 
            as_attachment=True, 
            download_name=f"image_{seed}.png"
        )
    return jsonify(error="Image not found"), 404

# Serve the temporary images
@app.route('/temp/<filename>', methods=['GET'])
def serve_temp_image(filename):
    return send_file(os.path.join(temp_dir, filename), mimetype='image/png')

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.2', port=5000)