# import the required libraries
from flask import Flask, render_template, request, send_file, jsonify
import torch
import random
import os
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)

#!  ╔════════════════════════════════════════════════════════════════════════════╗
#!  ║                        Load the required models                            ║
#!  ╚════════════════════════════════════════════════════════════════════════════╝


HF_TOKEN = os.getenv("HF_TOKEN")


#!  ╔════════════════════════════════════════════════════════════════════════════╗
#!  ║                              Load Pipeline                                 ║
#!  ╚════════════════════════════════════════════════════════════════════════════╝


def load_pipeline(model_name):
    # Load the VAE model
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )

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

    # Set the scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Move the pipeline to the appropriate device
    pipe.to('cuda')  # or 'cpu' if needed

    return pipe


#!  ╔════════════════════════════════════════════════════════════════════════════╗
#!  ║                                Main code                                   ║
#!  ╚════════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__)

# Create a temporary directory for images
generated_dir = './generated/'
os.makedirs(generated_dir, exist_ok=True)

# Dictionary to store generated images
generated_images = {}

def preprocess_prompt(prompt):
    return prompt

# Route for generating images
@app.route('/generate', methods=['POST'])
def generate():
    global pipe

    # Get the selected model from the form
    model_name = request.form['model']

    MODEL = os.getenv(
        "MODEL",
        model_name,
    )
    
    pipe = load_pipeline(MODEL)  # Load the pipeline with the selected model

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
    image_path = os.path.join(generated_dir, f'image_{seed}.png')
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
    return send_file(os.path.join(generated_dir, filename), mimetype='image/png')

# Serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='192.168.0.2', port=5000)