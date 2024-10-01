# import the required libraries
from flask import Flask, render_template, request, send_file, jsonify
import torch
import random
import utils
import json
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL
)
import os
import math
import logging
import time
import threading
from config import Config
from nudenet import NudeDetector

config = Config()

# Define model options from models.json
with open('models.json', 'r') as f:
    model_options = json.load(f)

with open('examplePrompts.json', 'r') as f:
    prompt_examples = json.load(f)

def load_pipeline(model_name):
    # Load the VAE model
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )

    # Set the pipepline
    pipeline = (
        StableDiffusionXLPipeline.from_single_file
        if model_name.endswith(".safetensors")
        else StableDiffusionXLPipeline.from_pretrained
    )

    # Load the pipeline
    pipe = pipeline(
        model_name,
        vae=vae,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        use_safetensors=True,
        add_watermarker=False,
        use_auth_token=config.HF_TOKEN,  # Assuming HF_TOKEN is set elsewhere
    )

    # Move the pipeline to the appropriate device
    pipe.to('cuda')  # or 'cpu' if needed

    return pipe

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/generate', methods=['POST'])
def generate():
    # Initialize global state
    global pipe, seed

    # Check if generation is already in progress
    if config.generating:
        return jsonify(status='Image generation already in progress'), 400

    config.generating = True
    config.generated_image.clear()
    config.imgprogress = "Starting image generation..."

    # Get parameters from the request
    model_name = request.form['model']
    prompt = utils.preprocess_prompt(request.form['prompt'])
    negative_prompt = utils.preprocess_prompt(request.form['negative_prompt'])
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    config.IMAGE_COUNT = int(request.form.get('image_count', 4))

    # Load the model pipeline
    try:
        pipe = load_pipeline(model_name)
    except Exception as e:
        config.generating = False
        return jsonify(status=f"Error loading model: {str(e)}"), 500

    # Function to generate images
    def generate_images():
        for i in range(config.IMAGE_COUNT):
            if config.generation_stopped:
                config.imgprogress = "Generation stopped"
                config.generating = False
                config.generation_stopped = False
                break

            # Update the progress message
            config.imgprogress = f"Generating {config.IMAGE_COUNT - i} images..."

            # Generate a new seed for each image
            seed = random.randint(0, 100000000000)
            image_path, sensitive = generateImage(prompt, negative_prompt, seed, width, height)

            # Store the generated image path
            config.generated_image[seed] = [image_path, sensitive]

        config.imgprogress = "Generation complete"
        config.generating = False

    # Start image generation in a separate thread to avoid blocking
    threading.Thread(target=generate_images).start()

    return jsonify(status='Image generation started', count=config.IMAGE_COUNT)

@app.route('/status', methods=['GET'])
def status():
    # Convert the generated images to a list to send to the client
    images =[
        {
            'img': path[0],
            'seed': seed,
            'sensitive': path[1]
        } for seed, path in config.generated_image.items()]
    
    return jsonify(images=images, imgprogress=config.imgprogress)

def generateImage(prompt, negative_prompt, seed, width, height):
    # Generate image with progress tracking

    detector = NudeDetector()

    def progress(step, timestep, latents):
        config.imgprogress = int(math.floor(step / 28 * 100))

    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=7,
        num_inference_steps=28,
        generator=torch.manual_seed(seed),
        callback=progress,
        callback_steps=1,
    ).images[0]

    # Save the image to the temporary directory
    image_path = os.path.join(config.generated_dir, f'image{time.time()}_{seed}.png')
    image.save(image_path, 'PNG')

    detection_results = detector.detect(image_path)
    config.imgprogress = "done"

    # Define sensitive classes
    sensitive_classes = {
        "FEMALE_GENITALIA_COVERED",
        "BUTTOCKS_EXPOSED",
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "ANUS_EXPOSED",
        "BELLY_COVERED",
        "ARMPITS_COVERED",
        "ARMPITS_EXPOSED",
        "BELLY_EXPOSED",
        "MALE_GENITALIA_EXPOSED",
        "ANUS_COVERED",
        "BUTTOCKS_COVERED"
    }

    return image_path, next((detection['class'] for detection in detection_results if detection['class'] in sensitive_classes), False)

# Serve the temp images
@app.route('/generated/<filename>', methods=['GET'])
def serve_temp_image(filename):
    return send_file(os.path.join(config.generated_dir, filename), mimetype='image/png')

@app.route('/stop', methods=['POST'])
def stop_generation():
    config.generation_stopped = True
    return jsonify(status='Image generation stopped')

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html', model_options=model_options, prompt_examples=prompt_examples)

if __name__ == '__main__':
    app.run(host='192.168.0.2', port=5000)