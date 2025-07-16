import time
from pathlib import Path
import os
from colorama import Fore
from PIL import Image

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import logging
import json

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

              
ML_zoo = {"Constituent": "'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' for foreground followed by 'LinReg' for background", "Background": "'LinReg' only for background analysis"}

def DETECT_OBJECTS(image, model_name, thresh, classes):
    start = time.time()
    
    for i in [1, 2]:
        try:
            MetadataCatalog.get(model_name).thing_classes
        except Exception as e:
            print(Fore.RED + f"WARNING: An error occured retrieving labels for {model_name}. This may be an issue with detectrons training process. Try running 'MetadataCatalog.get({model_name}).set(thing_classes=[array of classes for this dataset])'. ...Trying a workaround with local dataloaders... \n\n" + Fore.WHITE)
            MetadataCatalog.get(model_name).set(thing_classes=get_value_from_dict('MLtools/LoadModelMetadata.txt', model_name))
            continue

    logging.getLogger('matplotlib.font_manager').disabled = True

    # Configure the model for inference
    cfg = get_cfg()
    cfg.merge_from_file(cfg.OUTPUT_DIR + "/" + model_name + "_config.yaml")
    cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/" + model_name + ".pth"  # Load the saved model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh  # Set the testing threshold for this model
    predictor = DefaultPredictor(cfg)


    im = image
    outputs = predictor(im)

    #######################################
    # Get the total area of the image
    total_area = im.shape[0] * im.shape[1]

    # Get the predicted classes and masks
    pred_classes = outputs["instances"].pred_classes
    pred_masks = outputs["instances"].pred_masks
    pred_scores = outputs["instances"].scores

    # Count the number of instances per class and calculate the percentage of area
    unique_classes, counts = pred_classes.unique(return_counts=True)
    class_areas = np.zeros(len(unique_classes))

    for i, cls in enumerate(unique_classes):
        # Sum the area of masks for each class
        class_areas[i] = pred_masks[pred_classes == cls].sum().item()

    # Calculate the percentage area for each class
    class_percentages = (class_areas / total_area) * 100

    # Print the number of identified objects and percentage area for each class
    print("\n\nNumber and percentage of identified objects for each class:")
    objects = 0
    for cls, count, percentage in zip(unique_classes, counts, class_percentages):
        class_name = MetadataCatalog.get(model_name).thing_classes[cls]
        if class_name in classes:
            classes[class_name] = [classes[class_name][0]+percentage, classes[class_name][1]+count]
        else:
            classes[class_name] = [percentage, objects]
        print(f"{class_name}: {count} instances, {percentage:.2f}% of the image")
        objects += count
    ############################################

    # Visualize the results
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(model_name), scale=2.0)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    end = time.time()
    prediction = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    print(f"Process finished in {end-start} seconds")
    return prediction, (end-start)/objects, pred_scores.mean().item(), classes
    
    
def splice_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read the image.")

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Ensure the image is square
    if height != width:
        raise ValueError("The image must be square.")

    # Calculate the size of each segment
    segment_size = height // 4

    segments = []

    # Slice the image into 16 segments
    for i in range(4):
        for j in range(4):
            x_start = j * segment_size
            y_start = i * segment_size
            segment = image[y_start:y_start + segment_size, x_start:x_start + segment_size]
            segments.append(segment)

    return segments

def stitch_image(segments):
    if len(segments) != 16:
        raise ValueError("There must be exactly 16 segments to stitch the image.")

    # Get the size of each segment
    segment_size = segments[0].shape[0]

    # Create a blank canvas to stitch the segments
    stitched_image = np.zeros((segment_size * 4, segment_size * 4, 3), dtype=np.uint8)

    # Stitch the segments back together
    for i in range(4):
        for j in range(4):
            segment = segments[i * 4 + j]
            x_start = j * segment_size
            y_start = i * segment_size
            stitched_image[y_start:y_start + segment_size, x_start:x_start + segment_size] = segment

    return stitched_image

def display_image(image):
    # Convert BGR to RGB
    fig = plt.figure(figsize=(10, 10))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    
def plot_metrics(iters, confs, runtimes):
    # Create a figure and two subplots stacked on top of each other
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    
    fig.suptitle('Model Health', fontsize=16)
    
    # Plot Average Confidence on the top subplot
    ax1.plot(iters, confs, marker='o', linestyle='-', color='b', label='Average Confidence')
    ax1.set_ylabel('Average Confidence (‰)')
    ax1.set_ylim([0, 1])
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.set_title('Inference Confidence')
    
    # Plot Inference Time on the bottom subplot
    ax2.plot(iters, runtimes, marker='o', linestyle='-', color='r', label='Inference Time')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Time (s/object)')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='best')
    ax2.grid(True)
    ax2.set_title('Inference Time')

    # Show the plot
    plt.tight_layout()
    plt.show()


def DETECT_BACKGROUND():
    print("Background detection not currently supported")


Model_zoo_path = "output/"
models = []
for filename in [file for file in os.listdir(Model_zoo_path) if (file.endswith('.pth'))]:
    models.append(filename[:-4])

# Obtain the image from input
print(Fore.WHITE + "Welcome, please choose the image you would like to analyze\n")
time.sleep(2.0)
image = input("Full Image Path is: ")
print(Fore.WHITE + "Excelent choice. \n...Running image classification to determine sample type... \n")

# Run the image classification to determine what model will be used

# image_type = Torch_Interface.get_image_type(image)
image_type = "Constituent"

# Output results of image classification...image of type ____ detected. Proceeding to run {MaskRCNN, LinModel, ...}.
print(Fore.GREEN + "Image classfication was sucessful. \n...Spitting results... \n")
print(Fore.WHITE + f"Image type: {image_type} \nImage dimensions: {Image.open(image).width} x {Image.open(image).height} \nML Pipeline: {ML_zoo[image_type]}")


if image_type == "Constituent":
    print(f"Recomended Models: \n")
    for i in range(0, len(models)):
        print(Fore.YELLOW + models[i] + ": classes = " + ", ".join(get_value_from_dict('MLtools/LoadModelMetadata.txt', models[i])))
        
    print("\n")
    # Sire I have conjured up a suite of preloaded models listed above. Which will you be running?
    print(Fore.WHITE + "Sire, I have conjured up a suite of preloaded models and ones created by you. They are listed above.")
    modelName = input("Which will you be running?: ")
    print(Fore.GREEN + f"\nExcelent choice! running {image} on {ML_zoo[image_type]} framework with {modelName}")
    
    segments = splice_image(image)

    predictions = []
    runtimes = []
    confs = []
    iters = []
    classes = {}
    i = 0
    for segment in segments:
        prediction, runtime, conf, classes = DETECT_OBJECTS(segment, modelName, 0.1, classes)
        predictions.append(prediction)
        runtimes.append(runtime)
        confs.append(conf)
        iters.append(i)
        i+=1

    stitched_image = stitch_image(predictions)
    display_image(stitched_image)
    # Print Metrics
    print("\n\nNumber and percentage of identified objects for each class:")
    for classl in classes.keys():
        print(f"{classl}: {classes[classl][1]} instances, {(classes[classl][0]/16):.2f}% of the image")
    plot_metrics(iters, confs, runtimes)

    # DETECT_BACKGROUND()
    
elif image_type == "Background":
    print(Fore.GREEN + f"\nExcelent choice! running {image} on {ML_zoo[image_type]}.\n\n")
    DETECT_BACKGROUND()

