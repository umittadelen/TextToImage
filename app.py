# import the required libraries
from auto_installer import install_requirements
install_requirements()

import utils
from flask import Flask, render_template, request, send_file, jsonify
import torch, random, os, math, time, threading, sys, subprocess, glob, gc, logging, cv2, json
from PIL import PngImagePlugin, Image
import numpy as np
from io import BytesIO
import traceback
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
from transformers import CLIPTokenizer, CLIPModel, CLIPProcessor
from compel import Compel, ReturnedEmbeddingsType
from diffusers.utils import load_image
from downloadModelFromCivitai import downloadModelFromCivitai
from downloadModelFromHuggingFace import downloadModelFromHuggingFace

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def isDirectory(a):
    return os.path.isdir(a)
def isFile(a):
    return os.path.isfile(a)

gconfig = {
    "generation_stopped":False,
    "generating": False,
    "generated_dir": './generated/',
    "status": "",
    "progress": 0,
    "image_count": 0,
    "custom_seed": 0,
    "remainingImages": 0,
    "image_cache": {},
    "downloading": False,

    "enable_attention_slicing": True,
    "enable_xformers_memory_efficient_attention": False,
    "enable_model_cpu_offload": True,
    "enable_sequential_cpu_offload": False,
    "use_long_clip": True,
    "show_latents": True
}

gconfig["HF_TOKEN"] = (open(f'C:/Users/{os.getlogin()}/.cache/huggingface/token', 'r').read().strip() 
    if os.path.exists(f'C:/Users/{os.getlogin()}/.cache/huggingface/token') 
    else open(f'./civitai-api.key', 'r').read().strip() 
    if os.path.exists(f'./civitai-api.key') and open(f'./civitai-api.key', 'r').read().strip() != "" 
    else json.load(open('./static/json/settings.json', 'r', encoding='utf-8'))["HF_TOKEN"]
    if isFile("./static/json/settings.json")
    else "")

def checkModelsAvailability():
    print("Checking Models Availability...")
    try:
        with open('./static/json/models.json', 'r', encoding='utf-8') as f:
            models = json.load(f)
    except:
        print("Models file not found.")
        return
    
    for model in models:
        if "link" in models[model]:
            
            #! if path
            if isDirectory(models[model]["link"]):
                #TODO: if model exist but file not exist
                try:
                    if not os.path.exists(models[model]["link"]):
                        models.remove(model)
                except:
                    pass

            #! if civitai
            elif "civitai" in models[model]["link"]:
                #TODO: if model exist but file not exist
                try:
                    if not os.path.exists(models[model]["path"]):
                        downloadModelFromCivitai(models[model]["link"])
                except:
                    pass

            #! if huggingface
            elif models[model]["path"].count('/') == 1:
                #TODO: if model exist but file not exist
                #? not supported yet
                pass

checkModelsAvailability()

#TODO:  function to load the selected scheduler from name
def load_scheduler(pipe, scheduler_name):
    if   scheduler_name == "DPM++ 2M": pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
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
def load_pipeline(model_name, model_type, generation_type, scheduler_name):
    gconfig["status"] = "Loading New Pipeline... (loading Pipeline)"
    #TODO: Set the pipeline

    kwargs = {}

    if "controlnet" in generation_type and "SDXL" in model_type:
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
        kwargs["controlnet"] = controlnet

    if "SD 1.5" in model_type and "txt2img" in generation_type:
        kwargs["custom_pipeline"] = "lpw_stable_diffusion"
    elif "SDXL" in model_type and "txt2img" in generation_type:
        kwargs["custom_pipeline"] = "lpw_stable_diffusion_xl"
        kwargs["clip_skip"] = 2
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        kwargs["tokenizer"] = tokenizer

    if "img2img" in generation_type:
        pipeline = (
            StableDiffusionXLImg2ImgPipeline.from_single_file
            if "SDXL" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionXLImg2ImgPipeline.from_pretrained
            if "SDXL" in model_type else

            StableDiffusionImg2ImgPipeline.from_single_file
            if "SD 1.5" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionImg2ImgPipeline.from_pretrained
            if "SD 1.5" in model_type else

            DiffusionPipeline.from_pretrained
        )
    elif "controlnet" in generation_type:
        pipeline = (
            StableDiffusionXLControlNetPipeline.from_single_file
            if "SDXL" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionXLControlNetPipeline.from_pretrained
            if "SDXL" in model_type else

            StableDiffusionControlNetPipeline.from_pretrained
            if "SD 1.5" in model_type else

            DiffusionPipeline.from_pretrained
        )
    elif "FLUX" in generation_type:
        pipeline = FluxPipeline.from_pretrained
    else:
        pipeline = (
            StableDiffusionXLPipeline.from_single_file
            if "SDXL" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionXLPipeline.from_pretrained
            if "SDXL" in model_type else

            StableDiffusionPipeline.from_single_file
            if "SD 1.5" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

            StableDiffusionPipeline.from_pretrained
            if "SD 1.5" in model_type else

            DiffusionPipeline.from_pretrained
        )

    gconfig["status"] = "Loading New Pipeline... (Pipeline loaded)"

    gconfig["status"] = "Loading New Pipeline... (pipe)"
    #TODO: Load the pipeline

    pipe = pipeline(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        use_auth_token=gconfig["HF_TOKEN"],
        safety_checker=None,
        do_classifier_free_guidance=False,
        **kwargs
    )

    gconfig["status"] = "Loading New Pipeline... (loading VAE)"
    if gconfig["use_long_clip"]:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float16)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    pipe.clip_model = clip_model
    pipe.clip_processor = clip_processor

    #TODO: Load the VAE model
    if not hasattr(pipe, "vae") or pipe.vae is None:
        gconfig["status"] = "Model does not include a VAE. Loading external VAE..."
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )
        pipe.vae = vae
        gconfig["status"] = "External VAE loaded."

    gconfig["status"] = "Loading New Pipeline... (VAE loaded)"

    if scheduler_name != "None":
        pipe = load_scheduler(pipe, scheduler_name)
    gconfig["status"] = "Loading New Pipeline... (pipe loaded)"

    if torch.cuda.is_available():
        pipe.to('cuda')
    else:
        pipe.to('cpu')
        gconfig["status"] = "Using CPU..."
    
    if gconfig["enable_attention_slicing"]:
        pipe.enable_attention_slicing()
    if gconfig["enable_xformers_memory_efficient_attention"]:
        pipe.enable_xformers_memory_efficient_attention()
    if gconfig["enable_model_cpu_offload"]:
        pipe.enable_model_cpu_offload()
    if gconfig["enable_sequential_cpu_offload"]:
        pipe.enable_sequential_cpu_offload()

    gconfig["status"] = "Pipeline Loaded..."
    return pipe

def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)

def generateImage(pipe, model, prompt, original_prompt, negative_prompt, seed, width, height, img_input, strength, model_type, generation_type, image_size, cfg_scale, samplingSteps, scheduler_name, image_count):
    #TODO: Generate image with progress tracking
    current_time = time.time()

    def progress(pipe, step_index, timestep, callback_kwargs):
        gconfig["status"] = int(math.floor(step_index / samplingSteps * 100))
        gconfig["progress"] = int(math.floor((image_count - gconfig["remainingImages"] + (step_index / samplingSteps)) / image_count * 100))

        if gconfig["show_latents"]:
            image_path = os.path.join(gconfig["generated_dir"], f'image{current_time}_{seed}.png')
            image = latents_to_rgb(callback_kwargs["latents"][0])
            image.save(image_path, 'PNG')

            gconfig["image_cache"][seed] = [image_path]

        if gconfig["generation_stopped"]:
            gconfig["status"] = "Generation Stopped"
            gconfig["progress"] = 0
            pipe._interrupt = True

        return callback_kwargs

    gconfig["status"] = "Generating Image..."
    kwargs = {}

    try:
        #! Pass the parameters to the pipeline - (default kwargs for all pipelines)
        kwargs["generator"] = torch.manual_seed(seed)
        kwargs["guidance_scale"] = cfg_scale
        kwargs["num_inference_steps"] = samplingSteps
        kwargs["callback_on_step_end"] = progress
        kwargs["num_images_per_prompt"] = 1

        if "controlnet" in generation_type:
            if img_input != "":
                try:
                    image = load_image(img_input).convert("RGB")
                    if image_size == "resize":
                        image = utils.resize_image(image, width, height)

                except Exception:
                    #TODO: If the image is not valid, return False
                    gconfig["status"] = "Image Invalid"
                    traceback_details = traceback.format_exc()
                    print(f"Cannot acces to image:{traceback_details}")
                    return False
                image = np.array(image)

                # Apply Canny edge detection
                canny_edges = cv2.Canny(image, 100, 200)
                canny_edges = canny_edges[:, :, None]  # Add channel dimension
                canny_edges = np.concatenate([canny_edges, canny_edges, canny_edges], axis=2)  # Convert to 3 channels
                canny_image = Image.fromarray(canny_edges)

                #! Pass the image to pipeline - (kwargs for controlnet)
                kwargs["image"] = canny_image
                kwargs["strength"] = strength
            else:
                return False
        elif "img2img" in generation_type and "SDXL" not in model_type:
            if img_input != "":
                # Load and preprocess the image for img2img
                image = load_image(img_input).convert("RGB")
                if image_size == "resize":
                    image = utils.resize_image(image, width, height)

                #! Pass the image to pipeline - (kwargs for img2img)
                kwargs["image"] = image
                kwargs["strength"] = strength
            else:
                return False
        else:
            #! Pass the parameters to the pipeline - (kwargs for txt2img)
            kwargs["width"] = width
            kwargs["height"] = height
        
        if not gconfig["enable_sequential_cpu_offload"]:
            if hasattr(pipe, "tokenizer_2"):
                compel = Compel(
                    tokenizer=[pipe.tokenizer, pipe.tokenizer_2] ,
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )
                kwargs["prompt_embeds"], kwargs["pooled_prompt_embeds"] = compel(prompt)
                kwargs["negative_prompt_embeds"], kwargs["negative_pooled_prompt_embeds"] = compel(negative_prompt)
            else:
                compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
                kwargs["prompt_embeds"] = compel(prompt)
                kwargs["negative_prompt_embeds"] = compel(negative_prompt)
        else:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt

        with torch.no_grad():
            image = pipe(
                **kwargs
            ).images[0]

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
        metadata.add_text("Model", str(model))
        metadata.add_text("Scheduler", str(scheduler_name))

        #TODO: Save the image to the temporary directory
        image_path = os.path.join(gconfig["generated_dir"], f'image{current_time}_{seed}.png')
        image.save(image_path, 'PNG', pnginfo=metadata)

        gconfig["status"] = "DONE"
        gconfig["progress"] = 0

        return image_path

    except Exception:
        traceback_details = traceback.format_exc()
        gconfig["status"] = f"Generation Stopped with reason:<br>{traceback_details}"
        print(f"Generation Stopped with reason:\n{traceback_details}")
        return False

@app.route('/generate', methods=['POST'])
def generate():
    #TODO: Check if generation is already in progress
    if gconfig["generating"] or gconfig["downloading"]:
        return jsonify(status='Image generation already in progress'), 400

    gconfig["generating"] = True
    gconfig["image_cache"] = {}
    gconfig["status"] = "Starting Image Generation..."

    #TODO: Get parameters from the request
    model_name = request.form.get('model', 'https://huggingface.co/cagliostrolab/animagine-xl-3.1/blob/main/animagine-xl-3.1.safetensors')
    model_type = request.form.get('model_type', 'SDXL')
    scheduler_name = request.form.get('scheduler', 'Euler a')
    original_prompt = request.form.get('prompt', '1girl, cute, kawaii, full body')
    prompt = utils.preprocess_prompt(request.form.get('prompt', '1girl, cute, kawaii, full body')) if int(request.form.get("prompt_helper", 0)) == 1 else request.form.get('prompt', '1girl, cute, kawaii, full body')
    negative_prompt = request.form.get('negative_prompt', 'default_negative_prompt')
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    strength = float(request.form.get('strength', 0.5))
    img_input_link = request.form.get('img_input_link', "")
    img_input_img = request.files.get('img_input_img', "")
    generation_type = request.form.get('generation_type', 'txt2img')
    image_size = request.form.get('image_size', 'original')
    cfg_scale = float(request.form.get('cfg_scale', 7))
    image_count = int(request.form.get('image_count', 4))
    custom_seed = int(request.form.get('custom_seed', 0))
    samplingSteps = int(request.form.get('sampling_steps', 28))

    if custom_seed != 0:
        image_count = 1

    #save the temp image if provided and if not txt2img
    if img_input_img and generation_type != "txt2img":
        temp_image = Image.open(img_input_img).convert("RGB")
        temp_image.save("./generated/temp_image.png")

    img_input = "./generated/temp_image.png" if img_input_img else img_input_link

    #TODO: Function to generate images
    def generate_images():
        try:
            pipe = load_pipeline(model_name, model_type, generation_type, scheduler_name)
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["generating"] = False
            gconfig["status"] = f"Error Loading Model...<br>{traceback_details}"
            print(f"Error Loading Model...\n{traceback_details}")
            gconfig["progress"] = 0
            return

        try:
            for i in range(image_count):
                if gconfig["generation_stopped"]:
                    gconfig["progress"] = 0
                    gconfig["status"] = "Generation Stopped"
                    gconfig["generating"] = False
                    gconfig["generation_stopped"] = False
                    break

                #TODO: Update the progress message
                gconfig["remainingImages"] = image_count - i
                gconfig["status"] = f"Generating {gconfig["remainingImages"]} Images..."
                gconfig["progress"] = 0

                #TODO: Generate a new seed for each image
                if gconfig["custom_seed"] == 0:
                    seed = random.randint(0, 100000000000)
                else:
                    seed = gconfig["custom_seed"]

                image_path = generateImage(pipe, model_name, prompt, original_prompt, negative_prompt, seed, width, height, img_input, strength, model_type, generation_type, image_size, cfg_scale, samplingSteps, scheduler_name, image_count)

                #TODO: Store the generated image path
                if image_path:
                    gconfig["image_cache"][seed] = [image_path]
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["generating"] = False
            gconfig["status"] = f"Error Generating Images...<br>{traceback_details}"
            print(f"Error Generating Images...\n{traceback_details}")
            gconfig["progress"] = 0

        finally:
            del pipe
            torch.cuda.ipc_collect()
            gc.collect()
            torch.cuda.empty_cache()
            gconfig["status"] = "Generation Complete"

        gconfig["progress"] = 0
        gconfig["generating"] = False
        gconfig["generation_stopped"] = False

    #TODO: Start image generation in a separate thread to avoid blocking
    threading.Thread(target=generate_images).start()
    return jsonify(status='Image generation started', count=image_count)

@app.route('/save_prompt', methods=['POST'])
def save_prompt():
    prompt = request.form['prompt']
    if isFile('./static/json/saved_prompts.json'):
        prompts = json.load(open('./static/json/saved_prompts.json'))
    else:
        prompts = []

    if prompt in prompts:
        return jsonify(status='Prompt already exists')

    prompts.append(prompt)
    with open('./static/json/saved_prompts.json', 'w') as f:
        json.dump(prompts, f, indent=4)

    return jsonify(status='Prompt saved')

@app.route('/addmodel', methods=['POST'])
def addmodel():
    #TODO: Download the model
    model_url = request.form['model-name']

    gconfig["status"] = "Downloading Model..."

    if gconfig["generating"]:
        return jsonify(status='Image generation in progress. Please wait'), 400

    #! civitai.com
    if "civitai" in model_url:
        gconfig["downloading"] = True
        downloadModelFromCivitai(model_url)
        gconfig["downloading"] = False

        return jsonify(status='Model Downloaded')

    #! handle huggingface "{}/{}" format
    elif model_url.count('/') == 1:
        gconfig["downloading"] = True
        downloadModelFromHuggingFace(model_url)
        gconfig["downloading"] = False

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
    except Exception:
        traceback_details = traceback.format_exc()
        return jsonify({"error": str(traceback_details)}), 400

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
        } for seed, path in gconfig["image_cache"].items()]

    return jsonify(
        images=images,
        imgprogress=gconfig["status"],
        allpercentage=gconfig["progress"],
        remainingimages=gconfig["remainingImages"]-1 if gconfig["remainingImages"] > 0 else gconfig["remainingImages"]
    )

@app.route('/generated/<filename>', methods=['GET'])
def serve_temp_image(filename):
    size = request.args.get('size')
    image_path = os.path.join(gconfig["generated_dir"], filename)
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
    gconfig["generation_stopped"] = True
    return jsonify(status='Image generation stopped')

@app.route('/clear', methods=['POST'])
def clear_images():
    gconfig["image_cache"] = {}
    files = glob.glob(os.path.join(gconfig["generated_dir"], '*'))
    
    for file in files:
        try:
            os.remove(file)
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["progress"] = 0
            gconfig["status"] = f"Error Deleteing File... {traceback_details}"
            print(f"Error Deleteing File... {traceback_details}")
    return jsonify(status='Images cleared')

@app.route('/restart', methods=['POST'])
def restart_app():
    gconfig["progress"] = 0
    subprocess.Popen([sys.executable] + sys.argv)
    os._exit(0)

@app.route('/')
def index():
    return render_template('index.html')

def get_clip_token_info(text):
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Get tokens and their IDs
    clip_tokens = clip_tokenizer.encode(text, add_special_tokens=True)
    
    # Decode individual tokens for display
    clip_decoded = []
    for token in clip_tokens:
        decoded = clip_tokenizer.decode([token], skip_special_tokens=False)
        if decoded.isspace():
            decoded = "␣"
        elif decoded == "":
            decoded = "∅"
        clip_decoded.append(decoded)
    
    # Return the dictionary
    return {
        "CLIP Token Count": len(clip_tokens),
        "Tokens": clip_decoded
    }

@app.route('/clip_token_count', methods=['POST'])
def clip_token_count():
    text = request.form['text']
    result = get_clip_token_info(text)
    return jsonify(result)

@app.route('/clip_token')
def clip_token():
    return render_template('clip_token_count.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
def save_settings():
    settings = request.get_json()
    with open('./static/json/settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=4)
    gconfig.update(settings)
    return jsonify(status='Settings saved')

@app.route('/metadata')
def metadata():
    return render_template('metadata.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
    gconfig["status"] = "Server Started"