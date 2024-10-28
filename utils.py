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