import os
import subprocess
import shutil
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs

def is_command_available(command):
    """Check if a command is available on the system."""
    return shutil.which(command) is not None

def append_token_to_url(url, token_key, token_value):
    """Appends a token to a URL while respecting existing query parameters."""
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    query[token_key] = token_value
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed_url._replace(query=new_query))

# User inputs for the URL and API key
url = input("Enter the download URL: ").strip()
civitai_api_key = input("Enter your Civitai API key: ").strip()

# Generate the model directory dynamically based on the model ID
model_id = urlparse(url).path.split('/')[-1]
directory = os.path.join(".", "models", model_id)

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Append token to the URL
url = append_token_to_url(url, "token", civitai_api_key)

# Check if aria2c is available
if is_command_available("aria2c"):
    # Use aria2c if available
    subprocess.run(
        f"aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {directory} '{url}'",
        shell=True,
    )
elif is_command_available("wget"):
    # Fallback to wget
    subprocess.run(
        f"wget -q --show-progress --continue -P {directory} '{url}'",
        shell=True,
    )
elif is_command_available("curl"):
    # Fallback to curl
    subprocess.run(
        f"curl -L -o {os.path.join(directory, os.path.basename(url))} '{url}'",
        shell=True,
    )
else:
    print("No suitable download tool found. Please install aria2c, wget, or curl.")
