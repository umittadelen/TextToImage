import requests

url = "http://84.211.182.135:8080/generate"
form_data = {
    "model": "./models/kiwimixXL_v3.safetensors",
    "scheduler": "Euler a",
    "prompt": "1girl, solo, Kagamine Rin, shiny textures, detailed textures, looking at viewer, one hand on face, waving one hand, standing on one leg, :3, fang, fang out, sleeveless shirt, body, yellow hair, collared shirt, detached sleeves, hair ornament, bangs, white stockings, furry, animal paws, paw bean, full body",
    "negative_prompt": "(nsfw, lowres, (bad), text, error, fewer, extra, missing, (worst quality), (low quality),jpeg artifacts, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, abstract, bad anatomy, bad hands, bad feet, bad hand, bad hands, bad finger, bad fingers, extra finger, extra fingers, split finger, split fingers, extra digits, fused arms, fused hands:1.6)",
    "width": 1024,
    "height": 1024,
    "cfg_scale": 7,
    "image_count": 4,
    "custom_seed": 0,
    "sampling_steps": 28,
    "prompt_helper": "0"
}

response = requests.post(url, data=form_data)

# Print the JSON response (if any)
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
