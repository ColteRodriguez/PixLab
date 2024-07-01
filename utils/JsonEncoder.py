import json
import os

def create_json_string(label, points, imagePath, imageHeight, imageWidth):
    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None
            }
        ],
        "imagePath": imagePath,
        "imageData": "",
        "imageHeight": imageHeight,
        "imageWidth": imageWidth
    }
    
    return json.dumps(data, indent=2)

def add_shape_to_json(json_string, label, points):
    # Load the existing JSON string into a dictionary
    data = json.loads(json_string)
    
    # Create the new shape dictionary
    new_shape = {
        "label": label,
        "points": points,
        "group_id": None,
        "description": "",
        "shape_type": "polygon",
        "flags": {},
        "mask": None
    }
    
    # Append the new shape to the shapes list
    data['shapes'].append(new_shape)
    
    # Convert the dictionary back to a JSON string
    updated_json_string = json.dumps(data, indent=2)
    return updated_json_string
        