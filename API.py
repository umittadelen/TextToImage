import requests

url = "http://localhost:5000/generate"
data = {
    "model": "your_model_name",
    "scheduler": "your_scheduler_name",
    "prompt": "your_prompt_text",
    "negative_prompt": "your_negative_prompt",
    "width": 832,
    "height": 1216,
    "cfg_scale": 7,
    "image_count": 4,
    "custom_seed": 0,
    "sampling_steps": 28,
    "prompt_helper": "OFF"
}

response = requests.post(url, data=data)
print(response.json())
