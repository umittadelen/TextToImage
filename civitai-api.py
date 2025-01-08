import requests

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
        "id": modelData["id"] if "id" in modelData else None,
        "files": {
            "name": modelData["files"][0]["name"] if "files" in modelData else None,
            "downloadUrl": modelData["files"][0]["downloadUrl"] if "files" in modelData else None
        },
        "images": {
            "url": modelData["images"][0]["url"] if "images" in modelData else None,
            "width": modelData["images"][0]["width"] if "images" in modelData else None,
            "height": modelData["images"][0]["height"] if "images" in modelData else None
        },
        "base_id": modelData["base_id"] if "base_id" in modelData else None
    }
    return tmp

with open("civitai-api.key", "r") as file:
    token = file.read().strip()

model_id = 4468
version_id = 7425

print(getModeldata(model_id, version_id))