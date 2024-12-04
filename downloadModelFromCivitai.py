import os
import subprocess
import shutil
import json
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs

def is_command_available(cmd):
    return shutil.which(cmd) is not None

def append_token(url, key, value):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query[key] = value
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))

def update_models_json(file_name, file_path):
    models_json_path = './static/json/models.json'
    models_data = {} if not os.path.exists(models_json_path) else json.load(open(models_json_path, 'r', encoding='utf-8'))
    models_items = list(models_data.items())
    custom_index = next((i for i, entry in enumerate(models_items) if entry[0] == "Custom"), None)
    models_items.insert(custom_index if custom_index is not None else len(models_items), (file_name, [file_path, "7"]))
    json.dump(dict(models_items), open(models_json_path, 'w', encoding='utf-8'), indent=4)

def fix_path_slashes(path):
    return path.replace("\\", "/")

url = input("Enter the download URL: ").strip()
api_key = open("civitai-api.key", "r").read().strip() if os.path.exists("civitai-api.key") else input("Enter API key: ").strip()

temp_directory = "./models/temp"
os.makedirs(temp_directory, exist_ok=True)

if "token" not in url:
    url = append_token(url, "token", api_key)

print(f"url: ({url})")

if is_command_available("aria2c"):
    print("Using aria2c for download.")
    subprocess.run(f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {temp_directory} {url}", shell=True)
elif is_command_available("wget"):
    print("Using wget for download.")
    subprocess.run(f"wget -q -c -P {temp_directory} '{url}'", shell=True)
elif is_command_available("curl"):
    print("Using curl for download.")
    sanitized_url = url.split('?')[0]
    file_name = os.path.basename(sanitized_url)
    subprocess.run(f"curl -L -o \"{os.path.join(temp_directory, file_name)}\" \"{url}\"", shell=True)
else:
    print("No suitable download tool found.")

files = os.listdir(temp_directory)
if files:
    file_name = files[0]
    file_path = os.path.join(temp_directory, file_name)
    destination_path = fix_path_slashes(os.path.join("./models", file_name))

    shutil.move(file_path, destination_path)
    print(f"Moved file to {destination_path}")

    shutil.rmtree(temp_directory)
    print("Deleted temp directory.")
    
    update_models_json(file_name, destination_path)
else:
    print("No files were downloaded.")
