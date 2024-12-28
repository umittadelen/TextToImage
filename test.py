import torch
from diffusers import StableDiffusionXLPipeline

# Test function to inspect pipeline output
def test_pipeline():
    # Sample test kwargs
    kwargs = {
        "prompt": "A test image",
        "negative_prompt": "None",
        "generator": torch.manual_seed(42),
        "guidance_scale": 7.0,
        "num_inference_steps": 28,
        "num_images_per_prompt": 1,
        "width": 512,
        "height": 512
    }

    # Create a dummy pipeline instance
    pipe = StableDiffusionXLPipeline.from_single_file(
        "./models/kiwimixXL_v3.safetensors",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
    ).to("cuda")

    # Call the pipeline and print the result
    result = pipe(**kwargs)
    print(f"Pipeline result: {result}")

    # Check if the result has 'images' attribute and access it
    if hasattr(result, 'images'):
        image = result.images[0]
        print("Image successfully generated!")
    else:
        print("No images in the pipeline result.")

# Run the test
test_pipeline()
