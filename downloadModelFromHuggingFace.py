from config import Config
config = Config()

# just load the model from huggingface to make it save to default directory
def downloadModelFromHuggingFace(model_name):
    # load the model with pipeline just to save it to default directory
    updateModelsJson(model_name)

def updateModelsJson(model_name):
    import json, os
    models_data = (json.load(open('./static/json/models.json', 'r', encoding='utf-8')) if os.path.exists('./static/json/models.json') else {})
    models_data[model_name.split("/")[1]] = [model_name, "7"]
    json.dump(models_data, open('./static/json/models.json', 'w', encoding='utf-8'), indent=4)