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
        
def get_shapes(img_path):
    # Load the JSON data from a file
    with open(img_path[:-3] + 'json') as f:
        data = json.load(f)

    # Create a dictionary to hold the labels and their corresponding points lists
    shapes_dict = {}

    # Loop through each shape in the shapes array
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        # If the label already exists in the dictionary, append the points to the list
        if label in shapes_dict:
            shapes_dict[label].append(points)
        else:
            # Otherwise, create a new entry with the label and initialize it with the points list
            shapes_dict[label] = [points]

    return shapes_dict
