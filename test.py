from transformers import CLIPTokenizer

# Load the CLIP tokenizer
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def get_clip_token_info(text):
    # Get tokens and their IDs
    clip_tokens = clip_tokenizer.encode(text, add_special_tokens=True)
    
    # Decode individual tokens for display
    clip_decoded = []
    for token in clip_tokens:
        decoded = clip_tokenizer.decode([token], skip_special_tokens=False)
        if decoded.isspace():
            decoded = "␣"
        elif decoded == "":
            decoded = "∅"
        clip_decoded.append(decoded)
    
    # Return the dictionary
    return {
        "CLIP Token Count": len(clip_tokens),
        "Tokens": clip_decoded
    }

# Example usage:
text = "This is a test prompt"
result = get_clip_token_info(text)
print(result)
