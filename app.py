# import the required libraries
import utils
utils.check_and_install()
from flask import Flask, render_template, request, send_file, jsonify
import torch, random, os, math, time, threading, sys, subprocess, glob, gc, logging, cv2
from PIL import PngImagePlugin, Image
import numpy as np
from config import Config
from io import BytesIO

from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    FluxPipeline,
    DiffusionPipeline,
    ControlNetModel,
    AutoencoderKL
)
from diffusers.utils import load_image
from downloadModelFromCivitai import downloadModelFromCivitai
from downloadModelFromHuggingFace import downloadModelFromHuggingFace

config = Config()
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

#TODO: function to load the selected scheduler from name
def load_scheduler(pipe, scheduler_name):
    if scheduler_name == "DPM++ 2M": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++ 2M Karras": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "DPM++ 2M SDE": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="sde-dpmsolver++")
    elif scheduler_name == "DPM++ 2M SDE Karras": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif scheduler_name == "DPM++ SDE": pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM++ SDE Karras": pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "DPM2": pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM2 Karras": pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "DPM2 a": pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "DPM2 a Karras": pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == "Euler": pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "Euler a": pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "Heun": pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "LMS": pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "LMS Karras": pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    return pipe

#TODO:  function to load pipeline from given huggingface repo and scheduler
def load_pipeline(model_name, model_type, scheduler_name):

    config.imgprogress = "Loading Pipeline..."

    if model_name not in config.model_cache:
        config.imgprogress = "Loading New Pipeline..."
        config.model_cache = {}

        config.imgprogress = "Loading New Pipeline... (loading Pipeline)"
        #TODO: Set the pipeline

        kwargs = {}

        if "controlnet" in model_type and "SDXL" in model_type:
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
            kwargs["controlnet"] = controlnet

        if "SD1.5" in model_type and "txt2img" in model_type: kwargs["custom_pipeline"] = "lpw_stable_diffusion"
        elif "SDXL" in model_type and "txt2img" in model_type: kwargs["custom_pipeline"] = "lpw_stable_diffusion_xl"

        if "img2img" in model_type:
            pipeline = (
                StableDiffusionXLImg2ImgPipeline.from_single_file
                if "SDXL" in model_type and model_name.endswith(".safetensors") else

                StableDiffusionXLImg2ImgPipeline.from_pretrained
                if "SDXL" in model_type else

                StableDiffusionImg2ImgPipeline.from_single_file
                if "SD1.5" in model_type and model_name.endswith(".safetensors") else

                StableDiffusionImg2ImgPipeline.from_pretrained
                if "SD1.5" in model_type else

                DiffusionPipeline.from_pretrained
            )
        elif "controlnet" in model_type:
            pipeline = (
                StableDiffusionXLControlNetPipeline.from_single_file
                if "SDXL" in model_type and model_name.endswith(".safetensors") else

                StableDiffusionXLControlNetPipeline.from_pretrained
                if "SDXL" in model_type else

                StableDiffusionControlNetPipeline.from_pretrained
                if "SD1.5" in model_type else

                DiffusionPipeline.from_pretrained
            )
        elif "FLUX" in model_type:
            pipeline = FluxPipeline.from_pretrained
        else:
            pipeline = (
                StableDiffusionXLPipeline.from_single_file
                if "SDXL" in model_type and model_name.endswith(".safetensors") else

                StableDiffusionXLPipeline.from_pretrained
                if "SDXL" in model_type else

                StableDiffusionPipeline.from_single_file
                if "SD1.5" in model_type and model_name.endswith(".safetensors") else

                StableDiffusionPipeline.from_pretrained
                if "SD1.5" in model_type else

                DiffusionPipeline.from_pretrained
            )

        config.imgprogress = "Loading New Pipeline... (Pipeline loaded)"

        config.imgprogress = "Loading New Pipeline... (pipe)"
        #TODO: Load the pipeline

        pipe = pipeline(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            use_auth_token=config.HF_TOKEN,
            **kwargs
        )

        config.imgprogress = "Loading New Pipeline... (loading VAE)"
        #TODO: Load the VAE model
        if not hasattr(pipe, "vae") or pipe.vae is None:
            config.imgprogress = "Model does not include a VAE. Loading external VAE..."
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,
            )
            pipe.vae = vae
            config.imgprogress = "External VAE loaded."

        config.imgprogress = "Loading New Pipeline... (VAE loaded)"

        if scheduler_name != "None":
            pipe = load_scheduler(pipe, scheduler_name)
        config.imgprogress = "Loading New Pipeline... (pipe loaded)"

        if torch.cuda.is_available():
            pipe.to('cuda')
        else:
            pipe.to('cpu')
            config.imgprogress = "Using CPU..."
        
        if config.enable_attention_slicing:
            pipe.enable_attention_slicing()
        if config.enable_xformers_memory_efficient_attention:
            pipe.enable_xformers_memory_efficient_attention()

        config.model_cache[model_name] = pipe
        config.imgprogress = "Pipeline Loaded..."
        return pipe
    else:
        config.imgprogress = "Using Cached Pipeline..."
        return config.model_cache[model_name]

def generateImage(pipe, prompt, original_prompt, negative_prompt, seed, width, height, img_input, strength, model_type, cfg_scale, samplingSteps):
    #TODO: Generate image with progress tracking

    def progress(pipe, step_index, timestep, callback_kwargs):
        config.imgprogress = int(math.floor(step_index / samplingSteps * 100))
        config.allPercentage = int(math.floor((config.IMAGE_COUNT - config.remainingImages + (step_index / samplingSteps)) / config.IMAGE_COUNT * 100))

        if config.generation_stopped:
            config.imgprogress = "Generation Stopped"
            config.allPercentage = 0
            raise Exception("Generation Stopped")

        return callback_kwargs

    config.imgprogress = "Generating Image..."
    kwargs = {}

    try:
        if "controlnet" in model_type:
            if img_input != "":
                try:
                    image = load_image(img_input).convert("RGB")
                except:
                    #TODO: If the image is not valid, return False
                    config.imgprogress = "Image Invalid"
                    logging.log(logging.ERROR, msg=f"Cannot acces to image:{e}")
                    config.model_cache = {}
                    return False
                image = np.array(image)

                # Apply Canny edge detection
                canny_edges = cv2.Canny(image, 100, 200)
                canny_edges = canny_edges[:, :, None]  # Add channel dimension
                canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)  # Convert to 3 channels
                canny_image = Image.fromarray(canny_edges)

                kwargs["image"] = canny_image
                kwargs["prompt"] = prompt
                kwargs["negative_prompt"] = negative_prompt
                kwargs["strength"] = strength
                kwargs["guidance_scale"] = cfg_scale
                kwargs["num_inference_steps"] = samplingSteps
                kwargs["generator"] = torch.manual_seed(seed)
                kwargs["callback_on_step_end"] = progress
                kwargs["num_images_per_prompt"] = 1
            else:
                return False
        elif "img2img" in model_type and "SDXL" not in model_type:
            if img_input != "":
                # Load and preprocess the image for img2img
                image = load_image(img_input).convert("RGB")
                image = utils.resize_image(image, width, height)

                # Pass the original image to the pipeline
                kwargs["image"] = image
                kwargs["prompt"] = prompt
                kwargs["negative_prompt"] = negative_prompt
                kwargs["strength"] = strength
                kwargs["guidance_scale"] = cfg_scale
                kwargs["num_inference_steps"] = samplingSteps
                kwargs["generator"] = torch.manual_seed(seed)
                kwargs["callback_on_step_end"] = progress
                kwargs["num_images_per_prompt"] = 1
            else:
                return False
        else:
            # For txt2img pipelines
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt
            kwargs["width"] = width
            kwargs["height"] = height
            kwargs["guidance_scale"] = cfg_scale
            kwargs["num_inference_steps"] = samplingSteps
            kwargs["generator"] = torch.manual_seed(seed)
            kwargs["callback_on_step_end"] = progress
            kwargs["num_images_per_prompt"] = 1

        try:
            image = pipe(
                **kwargs
            ).images[0]
        except Exception as e:
            raise Exception(e)

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Prompt", prompt)
        metadata.add_text("OriginalPrompt", original_prompt)
        metadata.add_text("NegativePrompt", negative_prompt)
        metadata.add_text("Width", str(width))
        metadata.add_text("Height", str(height))
        metadata.add_text("CFGScale", str(cfg_scale))
        metadata.add_text("ImgInput", str(img_input) if "img2img" in model_type else "N/A")
        metadata.add_text("Strength", str(strength) if "img2img" in model_type else "N/A")
        metadata.add_text("Seed", str(seed))
        metadata.add_text("SamplingSteps", str(samplingSteps))
        metadata.add_text("Model", str(list(config.model_cache.keys())[0]))
        metadata.add_text("Scheduler", config.scheduler_name)

        #TODO: Save the image to the temporary directory
        image_path = os.path.join(config.generated_dir, f'image{time.time()}_{seed}.png')
        image.save(image_path, 'PNG', pnginfo=metadata)

        config.imgprogress = "DONE"
        config.allPercentage = 0

        return image_path

    except Exception as e:
        #TODO: If generation was stopped, handle it gracefully
        config.imgprogress = "Generation Stopped"
        logging.log(logging.ERROR, msg=f"Generation Stopped with reason:{e}")
        config.model_cache = {}
        return False

@app.route('/generate', methods=['POST'])
def generate():
    #TODO: Check if generation is already in progress
    if config.generating or config.downloading:
        return jsonify(status='Image generation already in progress'), 400

    config.generating = True
    config.generated_image = {}
    config.imgprogress = "Starting Image Generation..."

    #TODO: Get parameters from the request
    model_name = request.form.get('model', 'https://huggingface.co/cagliostrolab/animagine-xl-3.1/blob/main/animagine-xl-3.1.safetensors')
    model_type = request.form.get('model_type', 'SDXL')
    config.scheduler_name = request.form.get('scheduler', 'Euler a')
    original_prompt = request.form.get('prompt', '1girl, cute, kawaii, full body')
    prompt = utils.preprocess_prompt(request.form.get('prompt', '1girl, cute, kawaii, full body')) if int(request.form.get("prompt_helper", 0)) == 1 else request.form.get('prompt', '1girl, cute, kawaii, full body')
    negative_prompt = request.form.get('negative_prompt', 'default_negative_prompt')
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    strength = float(request.form.get('strength', 0.5))
    img_input = request.form.get('img_input', "")
    generation_type = request.form.get('generation_type', 'txt2img')
    cfg_scale = float(request.form.get('cfg_scale', 7))
    config.IMAGE_COUNT = int(request.form.get('image_count', 4))
    config.CUSTOM_SEED = int(request.form.get('custom_seed', 0))
    samplingSteps = int(request.form.get('sampling_steps', 28))

    if config.CUSTOM_SEED != 0:
        config.IMAGE_COUNT = 1
    
    if img_input != "":
        model_type = model_type+generation_type

    #TODO: Function to generate images
    def generate_images():
        try:
            pipe = load_pipeline(model_name, model_type, config.scheduler_name)
        except Exception as e:
            config.generating = False
            config.imgprogress = f"Error Loading Model...{e}"
            config.model_cache = {}
            config.allPercentage = 0
            return

        try:
            for i in range(config.IMAGE_COUNT):
                if config.generation_stopped:
                    config.allPercentage = 0
                    config.imgprogress = "Generation Stopped"
                    config.generating = False
                    config.generation_stopped = False
                    break

                #TODO: Update the progress message
                config.remainingImages = config.IMAGE_COUNT - i
                config.imgprogress = f"Generating {config.remainingImages} Images..."
                config.allPercentage = 0

                #TODO: Generate a new seed for each image
                if config.CUSTOM_SEED == 0:
                    seed = random.randint(0, 100000000000)
                else:
                    seed = config.CUSTOM_SEED

                image_path = generateImage(pipe, prompt, original_prompt, negative_prompt, seed, width, height, img_input, strength, model_type, cfg_scale, samplingSteps)

                #TODO: Store the generated image path
                if image_path:
                    config.generated_image[seed] = [image_path]
        except Exception as e:
            config.generating = False
            config.imgprogress = f"Error Generating Images...<br>{e}"
            config.model_cache = {}
            config.allPercentage = 0

        finally:
            del pipe
            config.model_cache = {}
            torch.cuda.ipc_collect()
            gc.collect()
            torch.cuda.empty_cache()
            config.imgprogress = "Generation Complete"

        config.allPercentage = 0
        config.generating = False
        config.generation_stopped = False

    #TODO: Start image generation in a separate thread to avoid blocking
    threading.Thread(target=generate_images).start()
    return jsonify(status='Image generation started', count=config.IMAGE_COUNT)

@app.route('/addmodel', methods=['POST'])
def addmodel():
    #TODO: Download the model
    model_url = request.form['model-name']

    config.imgprogress = "Downloading Model..."

    if config.generating:
        return jsonify(status='Image generation in progress. Please wait'), 400

    #! civitai.com
    if "civitai" in model_url:
        if not config.generating:

            config.downloading = True
            downloadModelFromCivitai(model_url)
            config.downloading = False

            return jsonify(status='Model Downloaded')

    #! handle huggingface "{}/{}" format
    elif model_url.count('/') == 1:
        if not config.generating:
            config.downloading = True
            downloadModelFromHuggingFace(model_url)
            config.downloading = False

            return jsonify(status='Model Downloaded')
    else:
        return jsonify(status='Invalid or Unsupported Model URL')
    
@app.route('/changejson', methods=['POST'])
def changejson():
    try:
        json_data = request.get_json()  # Parse JSON data from the request
        
        import json
        # Save the JSON data to the file
        with open('./static/json/models.json', 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)  # Ensure proper encoding

        return jsonify({"message": "JSON saved successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/serve_canny', methods=['POST'])
def serve_canny():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if not file:
        return 'No file provided', 400

    # Convert the uploaded file to a NumPy array
    img = Image.open(file)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Apply Canny edge detection
    edges = cv2.Canny(img, 100, 200)

    # Convert the result to a PIL Image
    edges_image = Image.fromarray(edges)

    # Save the result to a byte buffer
    buf = BytesIO()
    edges_image.save(buf, format='PNG')
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

@app.route('/canny')
def canny():
    return render_template('canny_preview.html')

@app.route('/status', methods=['GET'])
def status():
    #TODO: Convert the generated images to a list to send to the client
    images =[{
            'img': path[0],
            'seed': seed
        } for seed, path in config.generated_image.items()]

    return jsonify(
        images=images,
        imgprogress=config.imgprogress,
        allpercentage=config.allPercentage,
        remainingimages=config.remainingImages-1 if config.remainingImages > 0 else config.remainingImages
    )

@app.route('/generated/<filename>', methods=['GET'])
def serve_temp_image(filename):
    size = request.args.get('size')
    image_path = os.path.join(config.generated_dir, filename)
    size_map = {'mini': 4, 'small': 3, 'medium': 2}

    if size in size_map:
        with Image.open(image_path) as img:
            new_size = (img.width // size_map[size], img.height // size_map[size])
            img = img.resize(new_size, Image.LANCZOS)
            img_io = BytesIO()
            img.save(img_io, format='PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
    
    return send_file(image_path, mimetype='image/png')

@app.route('/image/<filename>', methods=['GET'])
def image(filename):
    return render_template('image.html', image=filename)

@app.route('/stop', methods=['POST'])
def stop_generation():
    config.generation_stopped = True
    return jsonify(status='Image generation stopped')

@app.route('/clear', methods=['POST'])
def clear_images():
    config.generated_image = {}
    files = glob.glob(os.path.join(config.generated_dir, '*'))
    
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            config.allPercentage = 0
            config.imgprogress = f"Error Deleteing File... {e}"
    return jsonify(status='Images cleared')

@app.route('/restart', methods=['POST'])
def restart_app():
    config.allPercentage = 0
    subprocess.Popen([sys.executable] + sys.argv)
    os._exit(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/metadata')
def metadata():
    return render_template('metadata.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
    config.imgprogress = "Server Started"