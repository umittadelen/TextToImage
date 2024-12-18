from collections import Counter
import re

def preprocess_prompt(prompt):
    # Define a function to extract keywords
    def extract_keywords(part):
        # Consider keywords to be adjectives, nouns, and verbs
        keywords = re.findall(r'\b\w+\b', part)
        return keywords

    # Split the prompt into parts based on commas
    parts = [part.strip() for part in prompt.split(",")]

    # Calculate importance for each part based on keyword frequency
    keyword_freq = Counter()
    for part in parts:
        keywords = extract_keywords(part)
        keyword_freq.update(keywords)

    # Determine importance scores
    importance = {}
    for part in parts:
        importance[part] = sum(keyword_freq[keyword] for keyword in extract_keywords(part))

    # Sort parts by their importance (highest first)
    sorted_parts = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Keep track of the total token count and selected parts
    token_count = 0
    selected_parts = []

    # Select parts while keeping within the 77-token limit
    for part, _ in sorted_parts:
        tokens = part.split()
        new_token_count = token_count + len(tokens)
        if new_token_count <= 77:
            selected_parts.append(part)
            token_count = new_token_count
        else:
            break

    # Join the selected parts back into a prompt
    return ", ".join(selected_parts)

def num_to_range(num, inMin, inMax, outMin, outMax):
    return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

import subprocess
import sys
import os

def check_and_install():
    if not os.path.exists("requirements.txt"):
        return

    try:
        import pkg_resources
        missing_packages = [pkg.strip() for pkg in open("requirements.txt") if pkg.strip().lower() not in {pkg.key for pkg in pkg_resources.working_set}]
    except ImportError:
        missing_packages = []

    if missing_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        except subprocess.CalledProcessError as e:
            return
        
from PIL import Image

def resize_image(image, width, height):
    """
    Resizes an image to the specified width and height.

    Args:
        image (PIL.Image.Image): The input image to resize.
        width (int): The desired width of the resized image.
        height (int): The desired height of the resized image.

    Returns:
        PIL.Image.Image: The resized image.
    """
    # Resize the image    
    return image.resize((width, height), resample=Image.BICUBIC)
