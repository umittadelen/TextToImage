import os
from huggingface_hub import login

class Config:
    def __init__(self):
        username = os.getlogin()
        HFHub_file = f'C:\\Users\\{username}\\.cache\\huggingface\\token'

        if os.path.exists(HFHub_file):
            with open(HFHub_file, 'r') as file:
                hf_token = file.read().strip()
        else:
            hf_token = None

        if hf_token in [None, "", "None"]:
            login()

        login(token=hf_token)

        self.HF_TOKEN = hf_token

        self.generation_stopped = False
        self.generating = False
        self.generated_dir = './generated/'
        self.downloading = False
        
        os.makedirs(self.generated_dir, exist_ok=True)
        
        self.generated_image = {}
        self.imgprogress = ""
        self.allPercentage = 0
        self.IMAGE_COUNT = 0
        self.CUSTOM_SEED = 0

        self.model_cache = {}

        self.remainingImages = 0
        self.scheduler_name = "Euler a"

        self.enable_attention_slicing = True
        self.enable_xformers_memory_efficient_attention = True

if __name__ == "__main__":
    config = Config()