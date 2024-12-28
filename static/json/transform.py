import json

def transform_structure(old_json):
    new_json = {}
    for key, values in old_json.items():
        new_json[key] = {
            "path": values[0],
            "cfg": values[1],
            "disabled": len(values) > 2 and values[2] == "disabled"
        }
    return new_json

# Read the JSON file and transform the structure
with open("./static/json/models.json", "r") as f:
    old_json = json.load(f)

new_json = transform_structure(old_json)

# Write the transformed JSON to a new file
with open("./static/json/newmodels.json", "w") as nf:
    json.dump(new_json, nf, indent=4)
