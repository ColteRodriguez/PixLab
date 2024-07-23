###########################################
# Lmao this can't be run in ipynb becasue multiprocesser freaks out.
# Its spotty but does run in cmd or terminal.
###########################################

from colorama import Fore
import sys
import os
import subprocess
import distutils.core

def read_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
        dictionary = json.loads(data)
    return dictionary

def get_value_from_dict(file_path, key):
    dictionary = read_dict_from_file(file_path)
    return dictionary.get(key, None)

def add_key_value_to_dict(file_path, key, value):
    dictionary = read_dict_from_file(file_path)
    dictionary[key] = value
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, indent=4)
        
def install_packages():
    # Install pyyaml
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyyaml==5.1'])

    # Clone the detectron2 repository
    if not os.path.exists('detectron2'):
        subprocess.check_call(['git', 'clone', 'https://github.com/facebookresearch/detectron2'])

    # Run setup.py to get the install requires
    dist = distutils.core.run_setup("./detectron2/setup.py")

    # Install the required packages
    for package in dist.install_requires:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

    # Insert detectron2 into the system path
    sys.path.insert(0, os.path.abspath('./detectron2'))

# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import torch, detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger


# import some common libraries
import numpy as np
np.bool = np.bool_
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# This is terrible code organization but I'm tired of making dozens of .py files
import shutil
import json

def update_json_image_path(json_file_path, new_image_path):
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Update the imagePath
        data['imagePath'] = new_image_path
        
        # Write the updated JSON back to the file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(Fore.WHITE + f"Updated imagePath in {json_file_path}")

    except Exception as e:
        print(f"An error occurred while updating {json_file_path}: {e}")

def get_unique_labels(directory_path):
    unique_labels = set()
    
    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Check if the file is a JSON file
            if filename.endswith('.json') and os.path.isfile(file_path) and "._" not in file_path:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Iterate over the shapes in the JSON file and add labels to the set
                for shape in data.get('shapes', []):
                    label = shape.get('label')
                    if label:
                        unique_labels.add(label)
        
        return list(unique_labels)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def create_folders_and_move_files(new_dir_name, data_dir_path):
    try:
        # Get the parent directory of the parent directory of the parent directory of the data directory
        parent_dir = os.path.abspath(os.path.join(data_dir_path, "../../.."))
        
        # Create the new main folder path
        new_main_folder_path = os.path.join(parent_dir, new_dir_name)
        
        # Create the 'train' subfolder path
        train_folder_path = os.path.join(new_main_folder_path, 'train')
        test_folder_path = os.path.join(new_main_folder_path, 'test')
        
        # Create the new directories
        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(test_folder_path, exist_ok=True)
        
        print(Fore.WHITE + f"Created new directory structure at: {new_main_folder_path}")
        
        # Move all files from the data directory to the 'train' subfolder
        for filename in os.listdir(data_dir_path):
            file_path = os.path.join(data_dir_path, filename)
            if os.path.isfile(file_path):  # Check if it is a file
                new_file_path = os.path.join(train_folder_path, filename)
                shutil.move(file_path, train_folder_path)
                print(Fore.WHITE + f"Moved {filename} to {train_folder_path}")
                
                # If the file is a JSON file, update its imagePath
                if filename.endswith('.json'):
                    new_image_path = os.path.join(train_folder_path, os.path.basename(file_path)).replace("\\", "/")
                    new_image_path = new_image_path[:-4] + "png"
                    update_json_image_path(new_file_path, new_image_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        
    return get_unique_labels(train_folder_path), new_main_folder_path


# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def register_dataset(labels, dataset_name, dataset_path):
    def to_coco(directory, labels):
        classes = labels
        dataset_dicts = []
        # must add extra condition for external drives which keep helper copies in read-only
        for filename in [file for file in os.listdir(directory) if (file.endswith('.json') and "._" not in file)]:
            json_file = os.path.join(directory, filename)
            with open(json_file) as f:
                img_anns = json.load(f)

            record = {}

            filename = os.path.join(directory, img_anns["imagePath"])

            record["file_name"] = filename
            record["height"] = img_anns["imageHeight"]
            record["width"] = img_anns["imageWidth"]

            annos = img_anns["shapes"]
            objs = []
            for anno in annos:
                px = [a[0] for a in anno['points']]
                py = [a[1] for a in anno['points']]
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(anno["label"]),
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    for d in ["train", "test"]:
        try:
            DatasetCatalog.register(dataset_name + "_" + d, lambda d=d: to_coco(dataset_path + d, labels))
            MetadataCatalog.get(dataset_name + "_" + d).set(thing_classes=labels)
        except Exception as e:
            print(Fore.RED + f"Error: {e}" + Fore.WHITE)
            continue
    microfacies_metadata = MetadataCatalog.get(dataset_name + "_train")
    print(microfacies_metadata)
    return microfacies_metadata, dataset_name + "_train"
    
from detectron2.engine import DefaultTrainer
import yaml

def train_and_save(metadata, training_supplies, labels, model_name):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'  # cpu or gpu
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (training_supplies,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0 # Default is 2. Honestly idk why it works with 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = 300  # Number of iterations
    cfg.SOLVER.STEPS = []  # No learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # RoIHead batch size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(labels)  # Number of classes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Save the model weights
    model_weights_path = os.path.join(cfg.OUTPUT_DIR, model_name + ".pth")
    torch.save(trainer.model.state_dict(), model_weights_path)

    # Save the configuration file
    config_file_path = os.path.join(cfg.OUTPUT_DIR, model_name + "_config.yaml")
    with open(config_file_path, 'w') as f:
        f.write(cfg.dump())


if __name__ == '__main__':
    print(Fore.WHITE + "...Ensuring Correct Build for detectron2...\n")
    
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)

    # Some basic setup:
    # Setup detectron2 logger
    setup_logger()
    
    if TORCH_VERSION == '2.0' and detectron2.__version__ == '0.6':
        print(Fore.GREEN + "detectron2 successfully configured" + Fore.WHITE + "\n\n...Configuring '/labeled' into a detectron2 readable format...")
    else:
        print(Fore.YELLOW + f"Either {TORCH_VERSION} != 2.0 or {detectron2.__version__} != 0.6. Code may be more buggy than expected \n...Configuring '/labeled' into a detectron2 readable format... ")

    model_name = input("What would you like to name this model (No Spaces!)? ")
    dataset_path = input("And what is the path to the /labeled of this dataset? ")

    # Move the files and 
    labels, new_main_folder_path = create_folders_and_move_files(model_name, dataset_path)

    print(Fore.GREEN + f"\nSucessfully reconfigured /labeled. \n " + Fore.WHITE + f"...Proceeding to train model on the following labels: {labels} in {new_main_folder_path}...")
    
    metadata, training_supplies = register_dataset(labels, model_name, new_main_folder_path + "/")
    
    add_key_value_to_dict("MLtools/LoadModelMetadata.txt", model_name, labels)
    
    train_and_save(metadata, training_supplies, labels, model_name)
    
    