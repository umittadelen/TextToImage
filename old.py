from flask import Flask, render_template, request, send_file, jsonify
import torch
import random
import os
import tempfile
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

app = Flask(__name__)

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

# Function to generate an image with a random seed
def generate_image_with_seed(prompt, negative_prompt, seed):
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=832,
        height=1216,
        guidance_scale=7,
        num_inference_steps=28,
        generator=generator
    ).images[0]
    
    # Save the image to the temporary directory
    image_path = os.path.join(temp_dir, f'image_{seed}.png')
    image.save(image_path, 'PNG')
    
    return image_path, seed

# Route for generating images
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    negative_prompt = request.form['negative_prompt']

    # Generate images with seeds
    seeds = [random.randint(0, 100000) for _ in range(4)]
    images = []
    for seed in seeds:
        image_path, seed = generate_image_with_seed(prompt, negative_prompt, seed)
        generated_images[seed] = image_path  # Store the image path for future downloads
        images.append({
            'img': f"/temp/{os.path.basename(image_path)}",
            'seed': seed
        })

    return jsonify(images=images)

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