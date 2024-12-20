import os

class Config():
    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        #self.HF_TOKEN = "HF_TOKEN"

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