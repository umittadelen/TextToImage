# import the required libraries
from flask import Flask, render_template, request, send_file, jsonify
import torch
import random
import os
import utils
import json
from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL
)
import threading
import math
import logging

HF_TOKEN = os.getenv("HF_TOKEN")

generation_stopped = False

# Define model options from models.json
with open('models.json', 'r') as f:
    model_options = json.load(f)

with open('loras.json', 'r') as f:
    lora_options = json.load(f)

def load_pipeline(model_name, lora_names=[], lora_weights={}):
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
        use_auth_token=HF_TOKEN,  # Assuming HF_TOKEN is set elsewhere
    )

    # Apply LoRA models if provided
    if lora_names:
        for lora_name in lora_names:
            if lora_name.endswith(".safetensors") or os.path.isfile(lora_name):
                pipe.unet.load_attn_procs(lora_name)
            else:
                # Load LoRA from HuggingFace or other sources
                pipe.load_lora_weights(lora_name)

            # Set the weight for each LoRA
            if lora_name in lora_weights:
                pipe.unet.set_lora_weight(lora_name, lora_weights[lora_name])

    # Move the pipeline to the appropriate device
    pipe.to('cuda')  # or 'cpu' if needed

    return pipe

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Create a temporary directory for images
generated_dir = './generated/'
os.makedirs(generated_dir, exist_ok=True)

# Dictionary to store generated images
generated_image = {}
imgprogress = ""

@app.route('/generate', methods=['POST'])
def generate():

    model_name = request.form['model']

    lora_names = request.form.getlist('lora_model')
    lora_weights = {name: float(request.form.get(f'lora_weight_{name}')) for name in lora_names}

    global pipe
    pipe = load_pipeline(model_name, lora_names, lora_weights)

    generated_image.clear()

    prompt = utils.preprocess_prompt(request.form['prompt'])
    negative_prompt = utils.preprocess_prompt(request.form['negative_prompt'])
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    IMAGE_COUNT = int(request.form.get('image_count', 4))

    # Function to generate images
    def generate_images():
        total_images = IMAGE_COUNT

        for i in range(total_images):
            global seed
            seed = random.randint(0, 100000000000)

            if generation_stopped:
                print("Generation stopped.")
                break  # Exit the loop if generation is stopped
                
            image_path = generateImage(prompt, negative_prompt, seed, width, height)
            generated_image[seed] = image_path

            global imgprogress
            imgprogress = "done"

            print(f"Generated image: {image_path}")


    # Start a new thread for image generation
    threading.Thread(target=generate_images).start()

    global generation_stopped
    generation_stopped = False

    return jsonify(status='Image generation started', count=IMAGE_COUNT)

@app.route('/status', methods=['GET'])
def status():
    # Convert the generated images to a list to send to the client
    global imgprogress
    images = [{'img': f"/generated/{os.path.basename(path)}", 'seed': seed} for seed, path in generated_image.items()]
    return jsonify(images=images, imgprogress=imgprogress)

def generateImage(prompt, negative_prompt, seed, width, height):
    # Generate image with progress tracking

    def progress(step, timestep, latents):
        global imgprogress
        imgprogress = str(math.floor(step / 28 * 100))

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
    image_path = os.path.join(generated_dir, f'image_{seed}.png')
    image.save(image_path, 'PNG')

    return image_path

# Route for downloading an image
@app.route('/generated/<int:seed>', methods=['GET'])
def download_image(seed):
    image_path = generated_image.get(seed)
    if image_path and os.path.exists(image_path):
        return send_file(
            image_path, 
            mimetype='image/png', 
            as_attachment=True, 
            download_name=f"image_{seed}.png"
        )
    return jsonify(error="Image not found"), 404

# Serve the temp images
@app.route('/generated/<filename>', methods=['GET'])
def serve_temp_image(filename):
    return send_file(os.path.join(generated_dir, filename), mimetype='image/png')

@app.route('/stop', methods=['POST'])
def stop_generation():
    global generation_stopped
    generation_stopped = True
    return jsonify(status='Image generation stopped')

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html', model_options=model_options, lora_options=lora_options)

if __name__ == '__main__':
    app.run(host='192.168.0.2', port=5000)