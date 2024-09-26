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

#!  ╔════════════════════════════════════════════════════════════════════════════╗
#!  ║                        Load the required models                            ║
#!  ╚════════════════════════════════════════════════════════════════════════════╝


HF_TOKEN = os.getenv("HF_TOKEN")

# Define model options from models.json
with open('models.json', 'r') as f:
    model_options = json.load(f)

#!  ╔════════════════════════════════════════════════════════════════════════════╗
#!  ║                              Load Pipeline                                 ║
#!  ╚════════════════════════════════════════════════════════════════════════════╝

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
        use_auth_token=HF_TOKEN,  # Assuming HF_TOKEN is set elsewhere
    )

    # Move the pipeline to the appropriate device
    pipe.to('cuda')  # or 'cpu' if needed

    return pipe

#!  ╔════════════════════════════════════════════════════════════════════════════╗
#!  ║                               Flask App                                    ║
#!  ╚════════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__)

# Create a temporary directory for images
generated_dir = './generated/'
os.makedirs(generated_dir, exist_ok=True)

# Dictionary to store generated images
generated_image = {}

@app.route('/generate', methods=['POST'])
def generate():
    global pipe
    pipe = load_pipeline(request.form['model'])

    prompt = utils.preprocess_prompt(request.form['prompt'])
    negative_prompt = utils.preprocess_prompt(request.form['negative_prompt'])
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    IMAGE_COUNT = int(request.form.get('image_count', 4))

    # Function to generate images
    def generate_images():
        for _ in range(IMAGE_COUNT):
            seed = random.randint(0, 100000000000)
            image_path = generateImage(prompt, negative_prompt, seed, width, height)
            generated_image[seed] = image_path
            
            # Here you might want to store the URLs directly if needed
            print(f"Generated image: {image_path}")

    # Start a new thread for image generation
    threading.Thread(target=generate_images).start()

    return jsonify(status='Image generation started', count=IMAGE_COUNT)

@app.route('/status', methods=['GET'])
def status():
    # Convert the generated images to a list to send to the client
    images = [{'img': f"/generated/{os.path.basename(path)}", 'seed': seed} for seed, path in generated_image.items()]
    return jsonify(images=images)

def generateImage(prompt, negative_prompt, seed, width, height):
    # Generate image
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=7,
        num_inference_steps=28,
        generator=torch.manual_seed(seed)
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

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index3.html', model_options=model_options)

if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.2', port=5000)