import requests
from flask import Flask, render_template, jsonify
from flask import request
import json
import os

app = Flask(__name__)

a = {
    'id': 7425,
    'baseModel': 'SD 1.5',
    'files': {
        'name': 'counterfeitV30_25.safetensors',
        'downloadUrl': 'https://civitai.com/api/download/models/7425'
    },
    'images': {
        'url': 'https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/79a71ba8-5efb-4b47-9903-d3ab125f2800/width=450/69402.jpeg',
        'width': 1152,
        'height': 1728
    }
}

def downloadWithTool(link, rootFolderPath="./civitaiModels", additionFolder=""):
    import shutil, subprocess
    def is_command_available(cmd):
            return shutil.which(cmd) is not None
    if is_command_available("aria2c"):
        print("Using aria2c for download.")
        subprocess.run(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {rootFolderPath + additionFolder.replace("./","")} {link}", shell=True)
    elif is_command_available("wget"):
        print("Using wget for download.")
        subprocess.run(f"wget -q -c -P {rootFolderPath + additionFolder.replace("./","")} '{link}'", shell=True)
    elif is_command_available("curl"):
        print("Using curl for download.")
        subprocess.run(f"curl -L -o \"{rootFolderPath + additionFolder.replace("./","")}\" \"{link}\"", shell=True)

def getModeldata(model_id, version_id):
    url = f"https://civitai.com/api/v1/models/{model_id}?token={token}"

    response = requests.get(url)

    if response.status_code == 200:
        model_data = response.json()
        modelVersion = next((m for m in model_data["modelVersions"] if m["id"] == version_id), model_data["modelVersions"][0])
        modelVersion["base_id"] = model_id
        return modelVersion
    else:
        return False

def shortenModelData(modelData):
    tmp = {
        "name": modelData["files"][0]["name"].split(".")[0].replace("_", " ") if "files" in modelData else None,
        "baseModel": modelData["baseModel"] if "baseModel" in modelData else None,
        "files": {
            "path": f"./{modelData["files"][0]["name"].split(".")[0]}/{modelData["files"][0]["name"]}" if "files" in modelData else None,
            "name": modelData["files"][0]["name"] if "files" in modelData else None,
            "downloadUrl": modelData["files"][0]["downloadUrl"] if "files" in modelData else None

        },
        "images": {
            "path": f"./{modelData["files"][0]["name"].split(".")[0]}/{modelData["images"][0]["url"].split("/")[-1]}" if "images" in modelData else None,
            "url": modelData["images"][0]["url"] if "images" in modelData else None,
            "width": modelData["images"][0]["width"] if "images" in modelData else None,
            "height": modelData["images"][0]["height"] if "images" in modelData else None
        },
        "base_id": int(modelData["base_id"]) if "base_id" in modelData else None
    }
    return tmp

def checkForModelFiles(modelData, folderPath="./civitaiModels"):
    print(f"Model: {folderPath}/{modelData['files']['name'].replace(".safetensors","")}/{modelData['files']['name']}")
    if os.path.exists(f"{folderPath}/{modelData['files']['name'].replace(".safetensors","")}"):
        return True
    else:
        return False

def downloadModel(modelData, token, folderPath="./civitaiModels"):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    if not os.path.exists(f"{folderPath}/{modelData['files']['name'].replace(".safetensors","")}"):
        if not os.path.exists(f"{folderPath}/{modelData['files']['name'].replace(".safetensors","")}/{modelData['files']['name']}"):
            downloadWithTool(modelData["files"]["downloadUrl"]+f"?token={token}", folderPath, f"/{modelData['files']['name'].replace('.safetensors', '')}/")
        with open(f"{folderPath}/{modelData["files"]["path"]}.json", "w") as file:
            json.dump(modelData, file, indent=4)
    else:
        return False

with open("civitai-api.key", "r") as file:
    token = file.read().strip()

@app.route('/')
def clip_token():
    return render_template('add_model.html')

@app.route('/add_model', methods=['POST'])
def add_model():
    model_id = request.form.get('model_id', "")
    version_id = request.form.get('version_id', "")
    model_data = getModeldata(model_id, version_id)
    
    # Ensure the model data was retrieved successfully
    if not model_data:
        return jsonify(status='Model data retrieval failed'), 400
    
    modelData = shortenModelData(model_data)
    
    # Check if model_data.json exists and initialize it as an empty list if not
    if not os.path.exists("model_data.json"):
        with open("model_data.json", "w") as file:
            file.write("[]")  # Initialize as an empty list
    
    with open("model_data.json", "r+") as file:
        data = json.load(file)
        
        # Ensure data is a list
        if not isinstance(data, list):
            data = []  # Reinitialize as an empty list if not a list
        
        # Check if the model is already in the data
        model_exists = any(existing_model['name'] == modelData['name'] for existing_model in data)
        
        if model_exists:
            if checkForModelFiles(modelData):
                return jsonify(status='Model already downloaded')
            else:
                downloadModel(modelData, token)
                return jsonify(status='Model added')
        else:
            downloadModel(modelData, token)
            data.append(modelData)  # Add the new model data to the list
        
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()
    
    return jsonify(status='Model added successfully'), 200  # Return a success response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)