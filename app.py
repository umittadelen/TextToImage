from flask import Flask, render_template, request, send_file, jsonify
import torch, os, time, traceback, threading, glob, sys, subprocess, json
from stablepy import Model_Diffusers

app = Flask(__name__)

def isDirectory(a):
    return os.path.isdir(a)
def isFile(a):
    return os.path.isfile(a)

gconfig = {
    "generation_stopped": False,
    "generating": False,
    "generated_dir": './generated',
    "status": "",
    "image_count": 0,
    "custom_seed": -1,
    "remainingImages": 0,
    "image_cache": {},

    "theme": False,
    "load_previous_data": True,
}

def load_model(model_name):
    gconfig["status"] = "Loading Model..."

    kwargs = {}

    model = Model_Diffusers(
        base_model_id=model_name,
        task_name="txt2img",
        type_model_precision=torch.float16,
        env_components=None,
        retain_task_model_in_cache=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    )

    gconfig["status"] = "Model Loaded..."
    return model

def generateImage(pipe, prompt, negative_prompt, width, height, cfg_scale, samplingSteps, clip_skip, custom_seed, scheduler_name):
    gconfig["status"] = "Generating Image..."
    if gconfig["generation_stopped"]:
        gconfig["generation_stopped"] = False
        return False
    try:
        img, info_img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=samplingSteps,
            guidance_scale=cfg_scale,
            clip_skip=clip_skip,
            filename_pattern=f'image_{time.time()},seed',
            sampler=scheduler_name,
            image_storage_location='./generated',
            seed=custom_seed if custom_seed != -1 else None,
            img_width=width,
            img_height=height,
        )
        gconfig["status"] = "DONE"

        return info_img[1][0].replace("\\", "/")

    except Exception:
        traceback_details = traceback.format_exc()
        gconfig["status"] = f"Generation Stopped with reason:<br>{traceback_details}"
        print(f"Generation Stopped with reason:\n{traceback_details}")
        return False

# API Routes for the app

@app.route('/generate', methods=['POST'])
def generate():
    gconfig["generating"] = True
    gconfig["image_cache"] = {}
    gconfig["status"] = "Starting Image Generation..."

    model_name = request.form.get('model', './models/shiitakeMix_v10.safetensors')
    scheduler_name = request.form.get('scheduler', 'Euler a')
    prompt = request.form.get('prompt', '1girl, cute, kawaii, full body')
    negative_prompt = request.form.get('negative_prompt', 'default_negative_prompt')
    width = int(request.form.get('width', 832))
    height = int(request.form.get('height', 1216))
    cfg_scale = float(request.form.get('cfg_scale', 7))
    clip_skip = int(request.form.get('clip_skip', 2))
    image_count = int(request.form.get('image_count', 4))
    custom_seed = int(request.form.get('custom_seed', -1))
    samplingSteps = int(request.form.get('sampling_steps', 28))

    def generate_images():
        try:
            model = load_model(model_name)
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["generating"] = False
            gconfig["status"] = f"Error Loading Model...<br>{traceback_details}"
            print(f"Error Loading Model...\n{traceback_details}")
            return

        try:
            for i in range(image_count):
                if gconfig["generation_stopped"]:
                    gconfig["status"] = "Generation Stopped"
                    gconfig["generating"] = False
                    gconfig["generation_stopped"] = False
                    break

                gconfig["remainingImages"] = image_count - i
                gconfig["status"] = f"Generating {gconfig['remainingImages']} Images..."

                image_path = generateImage(model, prompt, negative_prompt, width, height, cfg_scale, samplingSteps, clip_skip, custom_seed, scheduler_name)

                if image_path:
                    gconfig["image_cache"][image_path] = [image_path]
        except Exception:
            traceback_details = traceback.format_exc()
            gconfig["generating"] = False
            gconfig["status"] = f"Error Generating Images...<br>{traceback_details}"
            print(f"Error Generating Images...\n{traceback_details}")

        finally:
            del model
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            gconfig["status"] = "Generation Complete"

        gconfig["generating"] = False
        gconfig["generation_stopped"] = False

    threading.Thread(target=generate_images).start()
    return jsonify(status='Image generation started', count=image_count)

@app.route('/status', methods=['GET'])
def status():
    images = [{
        'img': path[0],
        'seed': seed
    } for seed, path in gconfig["image_cache"].items()]

    return jsonify(
        images=images,
        imgprogress=gconfig["status"],
        remainingimages=gconfig["remainingImages"] - 1 if gconfig["remainingImages"] > 0 else gconfig["remainingImages"]
    )

@app.route('/generated/<filename>', methods=['GET'])
def serve_temp_image(filename):
    image_path = os.path.join(gconfig["generated_dir"], filename)
    if not isFile(image_path):
        return jsonify(status='Image not found'), 404
    return send_file(image_path, mimetype='image/png')

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
            gconfig["status"] = f"Error Deleting File... {traceback_details}"
            print(f"Error Deleting File... {traceback_details}")
    return jsonify(status='Images cleared')

@app.route('/restart', methods=['POST'])
def restart_app():
    subprocess.Popen([sys.executable] + sys.argv)
    os._exit(0)

@app.route('/save_form_data', methods=['POST'])
def save_form_data():
    form_data = request.get_json()
    with open('./static/json/form_data.json', 'w', encoding='utf-8') as f:
        json.dump(form_data, f, indent=4)
    return jsonify(status='Form data saved')

@app.route('/load_form_data', methods=['GET'])
def load_form_data():
    if gconfig["load_previous_data"]:
        if not isFile('./static/json/form_data.json'):
            with open('./static/json/form_data.json', 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4)
            return jsonify({})
        with open('./static/json/form_data.json', 'r', encoding='utf-8') as f:
            form_data = json.load(f)
        return jsonify(form_data)
    else:
        return jsonify({})

@app.route('/reset_form_data', methods=['GET'])
def reset_form_data():
    with open('./static/json/form_data.json', 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=4)
    return jsonify(status='Form data reset')

@app.route('/load_settings', methods=['GET'])
def load_settings():
    with open('./static/json/settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    gconfig.update(settings)
    return jsonify(settings)

# HTML Routes for the app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image/<filename>', methods=['GET'])
def image(filename):
    if isFile(filename) or isFile(os.path.join(gconfig["generated_dir"], filename)):
        return render_template('image.html', image=filename)
    return render_template('image.html', image="")

@app.route('/metadata')
def metadata():
    return render_template('metadata.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
    gconfig["status"] = "Server Started"