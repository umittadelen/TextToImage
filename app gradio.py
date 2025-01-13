import utils
import gradio as gr
import torch, random, os, math, time, gc, logging, cv2, json
from PIL import PngImagePlugin, Image
import numpy as np

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

globals = {
    "generated_images": {},
    "generating": False,
    "generation_stopped": False,
    "progression": 0,
    "log":"",
    "scheduler_name": "",
    "model_cache": {},
    "image_count": 4,
    "remaining_images": 4,
    "custom_seed": 0,
    "enable_attention_slicing": False,
    "enable_xformers_memory_efficient_attention": False,
    "HF_TOKEN": open(f'C:\\Users\\{os.getlogin()}\\.cache\\huggingface\\token').read().strip()

}

os.makedirs('./generated', exist_ok=True)

#TODO:  function to load pipeline from given huggingface repo and scheduler
def load_pipeline(model_name, model_type, scheduler_name):

    globals["log"] = "Loading Pipeline..."

    if model_name not in globals["model_cache"]:
        globals["log"] = "Loading New Pipeline..."
        globals["model_cache"]

        globals["log"] = "Loading New Pipeline... (loading Pipeline)"
        #TODO: Set the pipeline

        kwargs = {}

        if "controlnet" in model_type and "SDXL" in model_type:
            controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)
            kwargs["controlnet"] = controlnet

        if "sd 1.5" in model_type and "txt2img" in model_type:
            kwargs["custom_pipeline"] = "lpw_stable_diffusion"
        elif "SDXL" in model_type and "txt2img" in model_type:
            kwargs["custom_pipeline"] = "lpw_stable_diffusion_xl"
            kwargs["clip_skip"] = 2

        if "img2img" in model_type:
            pipeline = (
                StableDiffusionXLImg2ImgPipeline.from_single_file
                if "SDXL" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

                StableDiffusionXLImg2ImgPipeline.from_pretrained
                if "SDXL" in model_type else

                StableDiffusionImg2ImgPipeline.from_single_file
                if "sd 1.5" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

                StableDiffusionImg2ImgPipeline.from_pretrained
                if "sd 1.5" in model_type else

                DiffusionPipeline.from_pretrained
            )
        elif "controlnet" in model_type:
            pipeline = (
                StableDiffusionXLControlNetPipeline.from_single_file
                if "SDXL" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

                StableDiffusionXLControlNetPipeline.from_pretrained
                if "SDXL" in model_type else

                StableDiffusionControlNetPipeline.from_pretrained
                if "sd 1.5" in model_type else

                DiffusionPipeline.from_pretrained
            )
        elif "FLUX" in model_type:
            pipeline = FluxPipeline.from_pretrained
        else:
            pipeline = (
                StableDiffusionXLPipeline.from_single_file
                if "SDXL" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

                StableDiffusionXLPipeline.from_pretrained
                if "SDXL" in model_type else

                StableDiffusionPipeline.from_single_file
                if "sd 1.5" in model_type and model_name.endswith((".ckpt", ".safetensors")) else

                StableDiffusionPipeline.from_pretrained
                if "sd 1.5" in model_type else

                DiffusionPipeline.from_pretrained
            )

        globals["log"] = "Loading New Pipeline... (Pipeline loaded)"

        globals["log"] = "Loading New Pipeline... (pipe)"
        #TODO: Load the pipeline

        pipe = pipeline(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            use_auth_token=globals["HF_TOKEN"],
            **kwargs
        )

        globals["log"] = "Loading New Pipeline... (loading VAE)"
        #TODO: Load the VAE model
        if not hasattr(pipe, "vae") or pipe.vae is None:
            globals["log"] = "Model does not include a VAE. Loading external VAE..."
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=torch.float16,
            )
            pipe.vae = vae
            globals["log"] = "External VAE loaded."

        globals["log"] = "Loading New Pipeline... (VAE loaded)"

        if scheduler_name != "None":
            pipe = load_scheduler(pipe, scheduler_name)
        globals["log"] = "Loading New Pipeline... (pipe loaded)"

        if torch.cuda.is_available():
            pipe.to('cuda')
        else:
            pipe.to('cpu')
            globals["log"] = "Using CPU..."
        
        if globals["enable_attention_slicing"]:
            pipe.enable_attention_slicing()
        if  globals["enable_xformers_memory_efficient_attention"]:
            pipe.enable_xformers_memory_efficient_attention()

        globals["model_cache"][model_name] = pipe
        globals["log"] = "Pipeline Loaded..."
        return pipe
    else:
        globals["log"] = "Using Cached Pipeline..."
        return globals["model_cache"][model_name]

def generate_images(prompt, negative_prompt, seed, width, height, img_input, strength, model_name, model_type, image_size, cfg_scale, samplingSteps):
        try:
            pipe = load_pipeline(model_name, model_type, globals["scheduler_name"])
        except Exception as e:
            globals["generating"] = False
            globals["log"] = f"Error Loading Model...{e}"
            print(f"Error Loading Model...{e}")
            globals["model_cache"] = {}
            return

        try:
            for i in range(globals["image_count"]):
                if globals["generation_stopped"]:
                    globals["log"] = "Generation Stopped"
                    globals["generating"] = False
                    globals["generation_stopped"] = False
                    break

                #TODO: Update the progress message
                globals["remaining_images"] = globals["image_count"] - i
                globals["log"] = f"Generating {globals['remaining_images']} Images..."

                #TODO: Generate a new seed for each image
                if globals["custom_seed"] == 0:
                    seed = random.randint(0, 100000000000)
                else:
                    seed = globals["custom_seed"]

                image_path = generateImage(pipe, prompt, negative_prompt, seed, width, height, img_input, strength, model_type, image_size, cfg_scale, samplingSteps)

                #TODO: Store the generated image path
                if image_path:
                    globals["generated_images"][seed] = [image_path]

                    global output_gallery
                    output_gallery.value(get_all_images())

        except Exception as e:
            globals["generating"] = False
            globals["log"] = f"Error Generating Images...<br>{e}"
            globals["model_cache"] = {}

        finally:
            del pipe
            globals["model_cache"] = {}
            torch.cuda.ipc_collect()
            gc.collect()
            torch.cuda.empty_cache()
            globals["log"] = "Generation Complete"

        globals["generating"] = False
        globals["generation_stopped"] = False
        return

#TODO: function to load the selected scheduler from name
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

def generateImage(pipe, prompt, negative_prompt, seed, width, height, img_input, strength, model_type, image_size, cfg_scale, samplingSteps):
    #TODO: Generate image with progress tracking

    def progress(pipe, step_index, timestep, callback_kwargs):
        globals["progression"] = int(math.floor(step_index / samplingSteps * 100))

        if globals["generation_stopped"]:
            globals["log"] = "Generation Stopped"
            raise Exception("Generation Stopped")

        return callback_kwargs

    globals["log"] = "Generating Image..."
    kwargs = {}

    try:
        #! Pass the parameters to the pipeline - (default kwargs for all pipelines)
        kwargs["prompt"] = prompt
        kwargs["negative_prompt"] = negative_prompt
        kwargs["generator"] = torch.manual_seed(seed)
        kwargs["guidance_scale"] = cfg_scale
        kwargs["num_inference_steps"] = samplingSteps
        kwargs["callback_on_step_end"] = progress
        kwargs["num_images_per_prompt"] = 1

        if "controlnet" in model_type:
            if img_input != "":
                try:
                    image = load_image(img_input).convert("RGB")
                    if image_size == "resize":
                        image = utils.resize_image(image, width, height)

                except Exception as e:
                    #TODO: If the image is not valid, return False
                    globals["log"] = "Image Invalid"
                    logging.log(logging.ERROR, msg=f"Cannot acces to image:{e}")
                    globals["model_cache"] = {}
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
        elif "img2img" in model_type and "SDXL" not in model_type:
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

        image = pipe(
            **kwargs
        ).images[0]

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("Prompt", prompt)
        metadata.add_text("NegativePrompt", negative_prompt)
        metadata.add_text("Width", str(width))
        metadata.add_text("Height", str(height))
        metadata.add_text("CFGScale", str(cfg_scale))
        metadata.add_text("ImgInput", str(img_input) if "img2img" in model_type else "N/A")
        metadata.add_text("Strength", str(strength) if "img2img" in model_type else "N/A")
        metadata.add_text("Seed", str(seed))
        metadata.add_text("SamplingSteps", str(samplingSteps))
        metadata.add_text("Model", str(list(globals["model_cache"].keys())[0]))
        metadata.add_text("Scheduler", globals["scheduler_name"])

        #TODO: Save the image to the temporary directory
        image_path = os.path.join("./generated", f'image{time.time()}_{seed}.png')
        image.save(image_path, 'PNG', pnginfo=metadata)

        globals["log"] = "Generation Complete"

        return image_path

    except Exception as e:
        #TODO: If generation was stopped, handle it gracefully
        globals["log"] = "Generation Stopped"
        logging.log(logging.ERROR, msg=f"Generation Stopped with reason:{e}")
        globals["model_cache"] = {}
        return False

def get_model_buttons():
    with open("./model_data.json", "r") as file:
        models = json.load(file)

    buttons = []
    for model in models:
        buttons.append(
            {
                "label": model["name"],
                "value": model["files"]["path"],
                "image": model["images"]["path"],
            }
        )
    return buttons

def stop_generation():
    globals["generation_stopped"] = True
    globals["generating"] = False
    globals["log"] = "Generation Stopped"

model_buttons = get_model_buttons()

def get_all_images():
    return list(globals["generated_images"].values()) if len(globals["generated_images"]) > 0 else None

# Create the Gradio interface using `with gr.Blocks`
with gr.Blocks() as iface:
    with gr.Row():
        prompt = gr.Textbox(value="1girl, cute, kawaii", label="Prompt")
        negative_prompt = gr.Textbox(value="default_negative_prompt", label="Negative Prompt")
    with gr.Row():
        seed = gr.Number(value=0, label="Seed (0 for random)")
        width = gr.Number(value=832, label="Width")
        height = gr.Number(value=1216, label="Height")
    with gr.Row():
        image_input = gr.File(label="Image Input (Optional)")
        strength = gr.Slider(0.0, 1.0, value=0.5, label="Strength")
    with gr.Row():
        with open("./model_data.json", "r") as file:
            models = json.load(file)
            model_paths = ["./civitaiModels/"+(model["files"]["path"].lstrip("./")) for model in models]
        model_name = gr.Dropdown(choices=model_paths, value=model_paths[0], label="Model Name")
        model_type = gr.Dropdown(choices=["SDXL", "SD 1.5"], value="SDXL", label="Model Type")
        image_size = gr.Dropdown(choices=["original", "resize"], value="original", label="Image Size (for img2img)")
    with gr.Row():
        cfg_scale = gr.Slider(1, 20, step=1, value=7, label="CFG Scale")
        sampling_steps = gr.Slider(10, 100, step=1, value=28, label="Sampling Steps")
    with gr.Row():
        generate_button = gr.Button("Generate")
    
    # Define output
    global output_gallery
    output_gallery = gr.Gallery(label="Generated Images")

    btn = gr.Button("refresh images")

    btn.click(get_all_images(), output_gallery)

    # Connect the button to the image generation function
    generate_button.click(generate_images, 
        inputs=[prompt, negative_prompt, seed, width, height, image_input, strength, model_name, model_type, image_size, cfg_scale, sampling_steps], 
        outputs=output_gallery)

# Launch the interface
iface.launch(server_name="0.0.0.0", server_port=8080)