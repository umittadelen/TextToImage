# import the required libraries
import utils
utils.check_and_install()
from flask import Flask, render_template, request, send_file, jsonify
import torch, random, os, math, time, threading, sys, subprocess, glob, gc, logging
from PIL import PngImagePlugin, Image
from config import Config
from nudenet import NudeDetector
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
    LMSDiscreteScheduler,
    StableDiffusionXLPipeline,
    AutoencoderKL
)

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
def load_pipeline(model_name, scheduler_name):
    config.imgprogress = "Loading Pipeline..."

    if model_name not in config.model_cache:
        config.imgprogress = "Loading New Pipeline..."
        config.model_cache = {}

        config.imgprogress = "Loading New Pipeline... (loading Pipeline)"
        #TODO: Set the pipeline
        pipeline = (
            StableDiffusionXLPipeline.from_single_file
            if model_name.endswith(".safetensors")
            else StableDiffusionXLPipeline.from_pretrained
        )
        config.imgprogress = "Loading New Pipeline... (Pipeline loaded)"

        config.imgprogress = "Loading New Pipeline... (pipe)"
        #TODO: Load the pipeline
        pipe = pipeline(
            model_name,
            torch_dtype=torch.float16,
            custom_pipeline="lpw_stable_diffusion_xl",
            use_safetensors=True,
            add_watermarker=False,
            use_auth_token=config.HF_TOKEN
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
            raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and drivers installed.")

        config.model_cache[model_name] = pipe
        config.imgprogress = "Pipeline Loaded..."
        return pipe
    else:
        config.imgprogress = "Using Cached Pipeline..."
        return config.model_cache[model_name]

def generateImage(pipe, prompt, original_prompt, negative_prompt, seed, width, height, cfg_scale, samplingSteps):
    #TODO: Generate image with progress tracking

    detector = NudeDetector()

    def progress(pipe, step_index, timestep, callback_kwargs):
        config.imgprogress = int(math.floor(step_index / samplingSteps * 100))
        config.allPercentage = int(math.floor((config.IMAGE_COUNT - config.remainingImages + (step_index / samplingSteps)) / config.IMAGE_COUNT * 100))

        if config.generation_stopped:
            config.imgprogress = "Generation Stopped"
            config.allPercentage = 0
            raise StopIteration

        return callback_kwargs

    config.imgprogress = "Generating Image..."

    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=cfg_scale,
            num_inference_steps=samplingSteps,
            generator=torch.manual_seed(seed),
            callback_on_step_end=progress,
            num_images_per_prompt=1
        ).images[0]

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Prompt", prompt)
        metadata.add_text("OriginalPrompt", original_prompt)
        metadata.add_text("NegativePrompt", negative_prompt)
        metadata.add_text("Width", str(width))
        metadata.add_text("Height", str(height))
        metadata.add_text("CFGScale", str(cfg_scale))
        metadata.add_text("Seed", str(seed))
        metadata.add_text("SamplingSteps", str(samplingSteps))
        metadata.add_text("Model", str(list(config.model_cache.keys())[0]))
        metadata.add_text("Scheduler", config.scheduler_name)

        #TODO: Save the image to the temporary directory
        image_path = os.path.join(config.generated_dir, f'image{time.time()}_{seed}.png')
        image.save(image_path, 'PNG', pnginfo=metadata)

        detection_results = detector.detect(image_path)
        config.imgprogress = "DONE"
        config.allPercentage = 0

        #TODO: Define sensitive classes
        sensitive_classes = {
            "FEMALE_GENITALIA_COVERED",
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
            "BELLY_COVERED",
            "BELLY_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "ANUS_COVERED",
            #"FEMALE_BREAST_COVERED",
            "BUTTOCKS_COVERED"
        }

        return image_path, next((detection['class'] for detection in detection_results if detection['class'] in sensitive_classes), False)

    except StopIteration:
        #TODO: If generation was stopped, handle it gracefully
        config.imgprogress = "Generation Manually Stopped"
        return False, False

@app.route('/generate', methods=['POST'])
def generate():
    #TODO: Check if generation is already in progress
    if config.generating:
        return jsonify(status='Image generation already in progress'), 400

    config.generating = True
    config.generated_image = {}
    config.imgprogress = "Starting Image Generation..."

    #TODO: Get parameters from the request
    model_name = request.form['model']
    config.scheduler_name = request.form['scheduler']
    original_prompt = request.form['prompt']
    prompt = utils.preprocess_prompt(request.form['prompt']) if int(request.form.get("prompt_helper", 0)) == 1 else request.form['prompt']
    negative_prompt = str(request.form['negative_prompt'])
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    cfg_scale = float(request.form.get('cfg_scale', 7))
    config.IMAGE_COUNT = int(request.form.get('image_count', 4))
    config.CUSTOM_SEED = int(request.form.get('custom_seed', 0))
    samplingSteps = int(request.form.get('sampling_steps', 28))

    if config.CUSTOM_SEED != 0:
        config.IMAGE_COUNT = 1

    #TODO: Function to generate images
    def generate_images():
        try:
            pipe = load_pipeline(model_name, config.scheduler_name)
        except Exception as e:
            config.generating = False
            config.imgprogress = "Error Loading Model..."
            config.allPercentage = 0
            print(e)
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

                image_path, sensitive = generateImage(pipe, prompt, original_prompt, negative_prompt, seed, width, height, cfg_scale, samplingSteps)

                #TODO: Store the generated image path
                if image_path:
                    config.generated_image[seed] = [image_path, sensitive]
        finally:
            del pipe
            config.model_cache = {}
            torch.cuda.ipc_collect()
            gc.collect()
            torch.cuda.empty_cache()

        config.imgprogress = "Generation Complete"
        config.allPercentage = 0
        config.generating = False
    #TODO: Start image generation in a separate thread to avoid blocking
    threading.Thread(target=generate_images).start()

    return jsonify(status='Image generation started', count=config.IMAGE_COUNT)

@app.route('/status', methods=['GET'])
def status():
    #TODO: Convert the generated images to a list to send to the client
    images =[
        {
            'img': path[0],
            'seed': seed,
            'sensitive': path[1]
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
    if size == 'small':
        with Image.open(image_path) as img:
            new_size = (img.width // 3, img.height // 3)
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
    return render_template('index.html', use_hidden=False)

@app.route('/metadata')
def metadata():
    return render_template('metadata.html')

@app.route('/hidden')
def hidden_index():
    return render_template('index.html', use_hidden=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
    config.imgprogress = "Server Started"