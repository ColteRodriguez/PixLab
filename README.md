# PixLab: Installation, Troubleshooting, and Documentation
The PixLab repository is an implementation of Facebook’s detectron2, which allows users to trace constituents on sample images as well as utilize a suite of ML tools to automate this process. The PixLab repository has 2 main components.

## Table of Contents

1. [Overview](#Overview)
2. [Notices](#NOTICES)
3. [Updates](#Updates)
4. [Full Documentation](#Doc)
5. [Citations](#Citations)
6. [How To Cite](#Citations2)

## Overview <a name="Overview"></a>

### Component 1: The PixLab GUI (PixLab.py)
Annotating sample images in the PixLab GUI does 2 things:
1. Keeps track of annotations (how much of the sample is what constituent). 
2. Saves annotations to a dataset, which can be used by component 2

### Component 2: The PixLab ML suite (Train_Custom_Dataset.py and Run_Image_Analysis.py)
Tracing constituents on images is accurate for small datasets. Say, 2 sample images. However, we can imagine larger sets of sample images, say 200 images of various carbonates where we want to know the facies composition. It would take months to trace all those facies! Thus, it may be advantageous to trace a couple manual annotations, and train a ML model to make the rest.
Once a user has completed a sufficient amount of annotations, they can train a detectron2 panoptic segmentation model to analyze other sample images. This can simply be done by running Train_Custom_Dataset.py. This will train a panoptic segmentation model, register the dataset with detectron2 and save the model to the modelzoo subdirectory in the PixLab repository. Alternatively, if the user does not want to train their own model, they can utilize one of three that are preloaded to the PixLab repository./

In order to utilize detectron2 to make annotations, simply run Run_Image_Analysis.py. This program:
1. Prompts the user for the image they would like analyzed
2. Determines the correct ML pipeline for the image. This pipeline is determined automatically using an image classification model which effectively determines
   - Are there features in the foreground that can be identified with a detectron2 mask r-cnn? Essentially is the image of two endmembers – grainstone or mudstone-packstone?
   - How complex are the sample features, is there another approach that will provide better results than panoptic segmentation?

3. Once the pipeline is determined, the user is prompted to select on which model the ML will make predictions. By default, PixLab comes preloaded with 3 models:
   - Just_Ooids: Most accurate at identifying circular-semicircular clasts. Makes predictions on 3 classes using mask r-cnn and 2 classes using a pixel-wise Linear Regression model:
     * Ooid
     * Altered Ooid
     * Compound Grain
     * Pore Space
     * Mud Matrix
       
   - Generalized_Carbonates: Accurate for a wide variety of facies. Makes predictions on 6 classes using mask r-cnn and 4 classes using a pixel-wise Linear Regression model:
     * Ooid
     * Altered Ooid
     * Compound Grain
     * Cemented Bioclast
     * Bioclast Bryozoa
     * Bioclast Foraminifers
     * Pore Space
     * Mud Matrix
     * Altered Cement-Mud Crystallization
     * Organic Veins
       
   - Faster_Bioclast_Recognition: Best for packstones with a few features. Makes predictions on 4 classes using mask r-cnn and 4 classes using a pixel-wise Linear Regression model
     * Cemented Bioclast
     * Bioclast Bryozoa
     * Bioclast Foraminifers
     * Ooid
     * Pore Space
     * Mud Matrix
     * Altered Cement-Mud Crystallization
     * Organic Veins

Additionally, any model that the user has trained with Train_Custom_Dataset.py will also appear as an option here

4. Once a model is selected, the ML pipeline will be run. This consists of a few steps:
   1. Breaking the large image into a couple smaller segments
   2. Running the pipeline on each segment, extracting the features identified by mask r-cnn, as well as the matrix composition determined by the pixel-wise Linear Regression model.
   3. Splicing the segments back together and displaying the results.

5. The output is a complete breakdown of the sample image constituents. The results are then saved for future reference.


# Notices: <a name="Notices"></a>
1. Should have a workable version updated by end of today July 23, 2024

2. Sorry for the visual bugs, our worker bees (me) are working so very hard to debug!
Here's a (very unprofessional) code demo video as a placeholder for README -- nobody reads those anyways right?
[https://www.youtube.com/watch?v=27rF6Az2xL4](https://youtu.be/sTqYdOxsvGE)

3. See [Detectron2 Useful Info](#Detectron2_Usefull_Info). It is highly recomended that you run the code in a conda environment. This ensures that detectron2 doesnt wreck your computer

## Updates (As of noon 07-02-2024): <a name="Updates"></a>
- Fixed specific->generic path to spreadsheet in SheetAPI
- Added the Time Machine! View annotations for all facies on saved images to compare annotations with others or to ensure consistent annotating

## Updates (As of noon 07-08-2024):
- Fixed spreadsheet indexing so alteration score is acurate
- Added primary/background constituent tracking to account for all pixels (while only sneding foreground to training data)
- Other minor bug fixes


# FULL DOCUMENTATION <a name="Doc"></a>

## Table of Contents

1. [Classes](#Classes)
   - [Point2D](#Point2D)
   - [Polygon](#Polygon)
   - [Segment](#Segment)
   - [ScrollableListApp](#ScrollableListApp)
   - [AutoCompleteApp](#AutoCompleteApp)
2. [Libraries](#Libraries)
   - [SheetAPI](#SheetAPI)
   - [SetupAPI](#SetupAPI)
   - [JsonEncoder](#JsonEncoder)
3. [MLtools](#MLtools)
   - [Train_Custom_Dataset](#Train_Custom_Dataset)
   - [Run_Image_Analysis](#Run_Image_Analysis)
4. [Miscellaneous](#Miscellaneous)
   - [Preclassifier](#Preclassifier)
   - [Detectron2 Useful Info](#Detectron2_Usefull_Info)
   - [Documented Errors, Bugs, and fixes](#Documented_Errors_Bugs_and_fixes)
   - [Off My Soapbox: Use Cases, Future Changes, etc.](#Soapbox)
___

## Classes <a name="Classes"></a>

### Point2D <a name="Point2D"></a>

#### Description
A standard Point2D class constructed from a double x and double y coordinate

#### Methods

##### x(self)
```python
def x(self):
    """
    Standard instance method to return the x coordinate

    Args:
        None
    Returns:
        double: x coordinate
    """
```

##### y(self)
```python
def y(self):
    """
    Standard instance method to return the y coordinate

    Args:
        None
    Returns:
        double: y coordinate
    """
```

##### isWithin(self, Point2D otherPoint)
```python
def isWithin(self, otherPoint):
    """
    Determines if Euclidean distance to another point is less than a set threshold (THRESHOLD defined in outer scope)

    Args:
        Point2D otherPoint: point to comapre distance with self
    Returns:
        bool: True if Euclidean distance to otherPoint is less than threshold, false otherwise.
    """
```

##### distance(self, otherPoint)
```python
def distance(self, otherPoint):
    """
    Helper method for isWithin(). Euclidian distance formula

    Args:
        Point2D otherPoint: point to comapre distance with self
    Returns:
        double: Euclidean distance to otherPoint
    """

```

##### toString(self)
```python
def toString(self):
    """
    Standard toString instance method for unit testing in main()

    Args:
        None
    Returns:
        String: String representation of "(x, y)"
    """
```
___
### Polygon <a name="Polygon"></a>

#### Description
A standard Polygon class constructed from an array of Point2d objects. self.points[] functions like a stack

#### Methods

##### isComplete(self)
```python
def isComplete(self):
    """
    Replaces instance variable isCompelete

    Args:
        None
    Returns:
        Bool: True if polygon is closed (self.points[0] == self.points[-1])
    """
```

##### addPoint(self, point)
```python
def addPoint(self, point):
    """
    Push new Point2D object onto self.points[]. If the polygon has > 2 vertices and the new point is within (see isWithin() Point2D), the new point coordinates are set to be identical to the first point in self.points[] in order ot close the polygon shape.

    Args:
        Point2D point: Point object toadd to the polygon
    Returns:
        None
    """
```

##### getRecent(self)
```python
def getRecent(self):
    """
    Pops the most recently added point from self.points[], returns it, then pushes it back

    Args:
        None
    Returns:
        Point2D: most recent point
    """
```

##### removePoint(self)
```python
def removePoint(self):
    """
    Pops the most recently added point from self.points[] without replacement

    Args:
        None
    Returns:
        Point2D: most recent point
    """
```

##### getPoints(self)
```python
def getPoints(self):
    """
    returns self.points[]

    Args:
        None
    Returns:
        Point2D[]: self.points[] instance variable
    """
```

##### getPointsScaled(self, sW, sH)
```python
def getPointsScaled(self, sW, sH):
    """
    returns self.points[], where each point x, y is scaled by sW and sH respectivly. Used to convert point coordinates in GUI canvas to actual image coordinates

    Args:
        double sW: x scaling factor
        double sH: y scaling factor
    Returns:
        Point2D[]: self.points[] where each point is constructed from x*sW and y*sH
    """
```

##### getTuplePointsScaled(self, sW, sH)
```python
def getTuplePointsScaled(self, sW, sH):
    """
    returns self.points[], where each point x, y is a tuple 

    Args:
        double sW: x scaling factor
        double sH: y scaling factor
    Returns:
        Tuple[]: self.points[] where each point is constructed from x*sW and y*sH and represented by (x, y)
    """
```

##### toString(self)
```python
def getTuplePointsScaled(self, sW, sH):
    """
    returns a string representation of self.points[]

    Args:
        None
    Returns:
        String: Returns the unscaled self.points[] as a string. to return scaled use toStringScaled(self, sW, sH)
    """
```
___
### Segment <a name="Segment"></a>

#### Description
Modified PaintApp class. Used to communicate Point2D and Polygon classes with the physical tkinter GUI by using keybinds to draw points, lines, and polygons while storing their references for use on the backend. Cnstructed from a image canvas and a transparent canvas overlay. Instance variables:

    self.root: tkinter root window
    self.CW, self.CH: Canvas width and height respectivly
    self.canvas: tkinter canvas object
    self.imageO: A reference to the original image to be displayed 
    self.OW, self.OH: A referecne to the original image dimensions  
    self.scaleW, self.scaleH: How much we need to divide positions by to get the actual coords    
    self.image: The oringal image resized to self.CW, self.CH
    self.img: a tkPhotoImage instance of self.image
    self.ref: reference to self.img to ward off sneaky garbage collection      
    self.overlay: Create an RGBA image for transparency
    self.overlay_img: a tkPhotoImage instance of self.overlay  
    self.prev_x: Store previous click x-coordinate
    self.prev_y: Store previous click y-coordinate   
    self.shapes: Array of Polygon objects
    self.current: The polygon currently being formed
    self.current_polygon: Defensive copy of self.current

#### Methods

##### on_return(self)
```python
def on_return(self):
    """
    Adds the current shape (self.current) to self.shapes[] and resets the polygon parameters (self.current, self.prev_x, self.prev_y = None, None, None)

    Args:
        None
    Returns:
        None
    """
```

##### on_backspace(self, event)
```python
def on_backspace(self, event):
    """
    Removes the last added point from self.current, deletes the point on the canvas, and resets self.prev_x, y.

    Args:
        tkinter_keybind event: keyboard or touchpad input (right click)
    Returns:
        None
    """
```

##### on_click(self, event)
```python
def on_backspace(self, event):
    """
    Adds a new point to the current polygon. draws the vertices and edges as needed

    Args:
        tkinter_keybind event: keyboard or touchpad input (left click)
    Returns:
        None
    """
```

##### draw_dot(self, x, y)
```python
def on_backspace(self, event):
    """
    Adds a new vertex (black oval) to self.canvas.

    Args:
        double x: x coordinate
        double y: y coordinate
    Returns:
        None
    """
```

##### draw_line(self, x1, y1, x2, y2)
```python
def on_backspace(self, event):
    """
    Adds a new line between two vertices to self.canvas.

    Args:
        double x1: starting x coordinate
        double y1: starting y coordinate
        double x2: ending x coordinate
        double y2: ending y coordinate
    Returns:
        None
    """
```

##### fill_polygon(self, polygon, a)
```python
def fill_polygon(self, polygon, a):
    """
    Draws a semitransparent fill to a completed polygon.

    Args:
        Polygon polygon: newly completed polygon (self.current)
        int a: opacity (a in rgba). 128 by default
    Returns:
        None
    """
```

##### killPolygon(self):
```python
def killPolygon(self):
    """
    Deletes and resets all of self.current. Deletion alg relies on resetting every pixel within the bounding box of self.current and thus, can cause visual bugs whereby oberlapping or closeby polygons get deleted. This has not affect on the backend and should just be a visual bug. This visual bug can be mended in future updates by making annotation in self.overlay rather than self.canvas.

    Args:
        None
    Returns:
        None
    """
```

##### delete_item(self, x1, y1, x2, y2):
```python
def delete_item(self, x1, y1, x2, y2):
    """
    Deletes an object from self.canvas

    Args:
        double x1, y1, x2, y2: Defines a bounding box for the object
    Returns:
        None
    """
```
___
### ScrollableListApp <a name="ScrollableListApp"></a>

#### Description
Tracks completed images in the right box in the gui. Its so unimportant that I wont document it. Honestly you could delete it if you want.

___
### AutoCompleteApp <a name="AutoCompleteApp"></a>

#### Description
A tkinter dependecy used to display options of annotations in the annotation window to keep naming consistent. Instance vars:

        self.parent: tkinter window (typically an instance of tk.TopLevel)
        self.options: String[] of options
        self.text_var: a tkinter StringVar()
        self.textbox: tkinter textbox populated with self.text_var for user input
        self.dropdown_frame: frame for options
        self.listbox: tkinter Listbox(self.dropdown_frame)
        self.scrollbar: tkinter Scrollbar(self.dropdown_frame)

#### Methods

##### show_dropdown(self, *args)
```python
def show_dropdown(self, *args):
    """
    I think this one explains itself

    Args:
        *args: options for constituents
    Returns:
        None
    """
```

##### populate_listbox(self)
```python
def populate_listbox(self):
    """
    populates dropdown with options

    Args:
        None
    Returns:
        None
    """
```

##### on_select(self, event)
```python
def on_select(self, event):
    """
    leftclick keybind on an option in the dropdown will autofill it in the textbox

    Args:
        tkinter_keybind event: left click
    Returns:
        None
    """
```
___
## Libraries <a name="Libraries"></a>

### SheetAPI <a name="SheetAPI"></a>

#### Description
NOT AN API I DON'T KNOW WHY I CALLED IT AN API THERE AREN'T EVEN ANY CLASSES ITS JUST A LIBRARY BUT I DON'T WANT TO CHANGE THE NAME NOW. Uses openpyxl to manipulate Point_Counts.xlsx and track annotations in the GUI. Used by Segmentation.py to add new samples to the spreadsheet and by the PixLab.py gui to track sample annotations.

#### Methods

##### locate_sample(spreadsheet_path, name_query, sheet_num)
```python
def locate_sample(spreadsheet_path, name_query, sheet_num):
    """
    Finds the spreadsheet coordinates of a given query string

    Args:
        String spreadsheet_path: full path to spreadsheet
        String name_query: String to locate
        int sheet_num: 0 by default
    Returns:
        int row: returns None iff the query string doesnt exist in spreadsheet_path
        int col: returns None iff the query string doesnt exist in spreadsheet_path
    """
```

##### get_cell(row, col, spreadsheet_path, sheet_num)
```python
def get_cell(row, col, spreadsheet_path, sheet_num):
    """
    Returns the value of a given spreadsheet index

    Args:
        int row: spreadsheet coordinates (1-indexed)
        int col: spreadsheet coordinates (1-indexed)
        String spreadsheet_path: full path to spreadsheet
        int sheet_num: 0 by default
    Returns:
        String val: The value of the cell. String by default
    """
```

##### change_cell(row, col, spreadsheet_path, val, sheet_num)
```python
def change_cell(row, col, spreadsheet_path, val, sheet_num):
    """
     Changes the value of a given cell

    Args:
        int row: spreadsheet coordinates (1-indexed)
        int col: spreadsheet coordinates (1-indexed)
        String spreadsheet_path: full path to spreadsheet
        int or String val: vleu to be changed to
        int sheet_num: 0 by default
    Returns:
        None
    """
```

##### update_percent(Pcurr, Tcurr, segment_area, rating, isSubject)
```python
def update_percent(Pcurr, Tcurr, segment_area, rating, isSubject):
    """
     Updates the percent distributions of all constituents in Point_Counts.xlsx

    Args:
        double Pcurr: current percent of a given constituent
        double Tcurr: current total number of mapped pixels
        double segment_area: the number of pixels of the polygon to be added
        int rating: alteration score
        bool isSubject: True if the constituent == the constituent of the added polygon
    Returns:
        None
    """
```

##### add_new_sample(path, name)
```python
def add_new_sample(path, name):
    """
     Adds a new row header (sample) to the first available spot

    Args:
        String path: Spreadsheet path
        String name: name of the sample to be added
    Returns:
        None
    """
```

##### find_open_cell(filename)
```python
def find_open_cell(filename):
    """
     Finds an open column in order to add new constituents

    Args:
        String filename: Spreadsheet path
    Returns:
        None
    """
```

##### set_zeros(filename, col)
```python
def find_open_cell(filename):
    """
     Adds a new costituent column header

    Args:
        String filename: Spreadsheet path
        int col: open column to add
    Returns:
        None
    """
```
##### update_spreadsheet(polygon_area, constituent, image, altertion_score, path)
```python
def update_spreadsheet(polygon_area, constituent, image, altertion_score, path):
    """
     Calls all other methods in the library to update Point_Counts.xlsx upon saving an annotation in PixLab

    Args:
        Double polygon_area: number of pixels in the polygon
        String constituent: label of the annotation/polygon
        String image: path to the image being annotated
        int altertion_score: alteration score
        String path; spreadsheet path
    Returns:
        None
    """
```
___
### SetupAPI <a name="SetupAPI"></a>

#### Description
NOT AN API I DON'T KNOW WHY I CALLED IT AN API THERE AREN'T EVEN ANY CLASSES ITS JUST A LIBRARY BUT I DON'T WANT TO CHANGE THE NAME NOW. Uses openpyxl, PIL and matplotlib to perform actions in Setup.py and Segmentation.py. Handles image segmentation and all of the folder/file manipulation in Setup.py. Not called by PixLab.py at all.

#### Methods

##### grab_random_image(folder_path, helperfilepath)
```python
def grab_random_image(folder_path, helperfilepath):
    """
    grabs a random, non-segmented image from the input folder path

    Args:
        String folder_path: path to the sample_image subfolder in the project directory
        String helperfilepath: path to FS_helper.txt to ensure no previously segmented images are returned
        int sheet_num: 0 by default
    Returns:
        String: path to a sample image
    """
```

##### create_subdirectories(home, path)
```python
def create_subdirectories(home, path):
    """
    A single-use function to create subdirectories and helper files in the project directory

    Args:
        String home: path to the code directory
        String path: path to project directory
    Returns:
        none
    """
```

##### create_spreadsheet(home)
```python
def create_spreadsheet(home):
    """
    Just to avoid creating subdirs and files

    Args:
        String home: path to the project directory (variable naming is horrible consistent im sorry)
    Returns:
        none
    """
```

##### slice_segments(in_image, imName, external_drive_path)
```python
def slice_segments(in_image, imName, external_drive_path):
    """
    Segments an image in sample_images into multiple smaller cells and saves the cells into a folder in /Unlabeled. (project/training_data/All_data/Unlabeled/{image_name})

    Args:
        String in_image: full path to the image
        String imName: Name of the image without path or extension
        external_drive_path: path to the project directory
    Returns:
        none
    """
```

##### slice_segments(in_image, imName, external_drive_path)
```python
def slice_segments(in_image, imName, external_drive_path):
    """
    Segments an image in sample_images into multiple smaller cells and saves the cells into a folder in /Unlabeled. (project/training_data/All_data/Unlabeled/{image_name})

    Args:
        String in_image: full path to the image
        String imName: Name of the image without path or extension
        external_drive_path: path to the project directory
    Returns:
        none
    """
```

Lines 175-385 define classes for cutting up images with a quadtree energy-gradient algorithm. This process is left over from a previous iteration of the GUI. Essentially its only function is to split suqare images into several smaller square sections. I've kept it in because sometimes it works well if we want smaller image segments. However, a better, more updated approach would involve just 1 method which splits the image into n suqare cells of size m. This can be done in like 30 lines of code and should be a change I consider making in future updates however, for now I'm just keeping what works.
___

### JsonEncoder <a name="JsonEncoder"></a>

#### Description
Library of methods for tracking annotations in JSON files for detectron2. JSON file is effectivly a hashmap storing annotation attributes which are also themselves hashmaps and, in some cases hashmaps of hashmaps. The configuration mirrors that of "labelme" and thus is not already in COCO format. Converting to COCO is handled by Train_Custom_Dataset.py (see MLtools) however, this method is error prone. I've been dreaming up solutions to just saving JSON files in COCO format but all require substantial code changes and in its current fragile state, I dont want to release an update which makes PixLab dysfunctional.

#### Methods

##### create_json_string(label, points, imagePath, imageHeight, imageWidth)
```python
def create_json_string(label, points, imagePath, imageHeight, imageWidth):
    """
    Creates a JSON file and intializes the first shape

    Args:
        String label: label/constituent name of the annotation
        String points: see getPointsJSON() in Polygon.py
        int imageHeight, imageWidth: Actual image dimensionas before tkinter resize (self.OW and self.OH of class PaintApp in Segment.py)
    Returns:
        JSON file contents on init
    """
```

##### add_shape_to_json(json_string, label, points)
```python
def add_shape_to_json(json_string, label, points):
    """
    Effectivly put(). Adds a shape to the Shapes attribute of JSON

    Args:
        String json_string: Information regarding the annotation
        String points: see getPointsJSON() in Polygon.py
        String label: label/constituent name of the annotation
    Returns:
        All of the shapes attributes
    """
```

##### get_shapes(img_path)
```python
def get_shapes(img_path):
    """
    Takes an image path, searches for its JSOn file and if it exists, returns all of the shapes attributes

    Args:
        String img_path: full image path
    Returns:
        All of the shapes attributes
    """
```

##### changeSource(img_path, jsonfile)
```python
def changeSource(img_path, jsonfile):
    """
    Used in Train_Custom_Dataset.py to update the image path in the JSON file when /labeled is moved to a new folder to be read by detectron2

    Args:
        String img_path: full new image path
        String jsonfile: json file path
    Returns:
        None
    """
```
___
## MLtools <a name="MLtools"></a>
MLtools consists of two runable python files with detectron2 and torchvision dependencies

### Train_Custom_Dataset <a name="Train_Custom_Dataset"></a>

#### Description
Train_Custom_Dataset.py takes annotations from /labeled, converts json --> COCO formatted annotations, registers a dataset with detectron2, trains an instance segmentation model (COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x), Then saves the .yaml and .pth files to /output in order to make inferences on new images. Train_Custom_Dataset.py has a couple tunable parameters which can alter model performace. There parameters will be listed here with links to the methods documenting these parameters more in depth. I've contemplated giving the user the choice to tune parameters in the command line, however this seems too involved and I'd rather just run a default taining process that works well rather than have the user painstakingly tests parameters -- especially when training takes so long without cpu. The code for training uses detectron2's tutorial as a shell. This can be found [here](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html).

| param                 |  Explaination  |
| --------------------- | -------------- |
| cfg:                  | load an empty detectron2 configuration with get_cfg() on which we will build  |
| cfg.MODEL.DEVICE:     | cpu or gpu, cpu is slower and also the only option for mac.                     |
| cfg.merge_from_file() | The model_zoo is a subdirectory of detectron2 which hosts all of the untrained models. Instance segmentation works best for constituent outlining, but you can view all the options in the full model zoo: https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md |
| cfg.DATASETS.TRAIN    | Path to the training dataset /train                                          |
| cfg.DATASETS.TEST     | Path to the training dataset /test or () if not testing data exists          |
| cfg.DATALOADER.NUM_WORKERS | Relates to the number of parallel runs executed by the mutiprocessor, can cause errors, but ive found that 0 works fine |
| cfg.MODEL.WEIGHTS     | Initialize from model zoo. Getting the checkpoint of cfg.merge_from_file()   |
| cfg.SOLVER.IMS_PER_BATCH |  Number of samples passed through each step of the model at the same time. Higher number == faster training. 2 by default |
| cfg.SOLVER.BASE_LR    | the learning rate, how much the model is punished for a wrong prediction in training. 0.00025 by default. |
| cfg.SOLVER.MAX_ITER   | Higher number = more thourough training, though may lead to overtraining, smaller number = faster training but may casue undertraining. 300 by default 300 |
| cfg.SOLVER.STEPS      | How to decay the learning rate for each iteration. Maybe we want to punish the model less as it learns more?. 0 by default -- No learning rate decay |
| cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE | Default is 128. From ghost in [this forum](https://github.com/facebookresearch/detectron2/issues/1341): "parameter is used to sample a subset of proposals coming out of RPN to calculate cls and reg loss during training. Calculating loss on all RPN proposals isn't computationally efficient." |
| cfg.MODEL.ROI_HEADS.NUM_CLASSES | Number of classes. Default is len(labels)  where labels is the output of get_unique_labels() |

#### Methods

##### install_packages()
```python
def install_packages():
    """
    Ensures correct build for detectron2 and installs dependencies, some of which should already be preloaded in this repository (like /output)

    Args:
        None
    Returns:
        None
    """
```

##### update_json_image_path(json_file_path, new_image_path)
```python
def update_json_image_path(json_file_path, new_image_path):
    """
    Duplicate. See JsonEncode.changeSource(img_path, jsonfile)

    Args:
        String new_image_path: full new image path
        String json_file_path: json file path
    Returns:
        None
    """
```

##### get_unique_labels(directory_path)
```python
def get_unique_labels(directory_path):
    """
    Crawls all JSON files in /labeled for a all unique labels

    Args:
        String directory_path: path to a /labeled subdirectory
    Returns:
        String[]: Array of unique labels
    """
```

##### create_folders_and_move_files(new_dir_name, data_dir_path)
```python
def create_folders_and_move_files(new_dir_name, data_dir_path):
    """
    Creates a new subdirectory in project folder configured with /train and /test, and moves all images and json files from data_dir_path (/labeled) to /train of new_dir_name

    Args:
        String new_dir_name: path to a new folder named as given by user
        String data_dir_path: path to a /labeled folder named as given by user
    Returns:
        method_call get_unique_labels(new_dir_name/train): Array of unique labels forthe training data
        String new_main_folder_path: os.join(project directory, new_dir_name), full path to the new subdirectory
    """
```

##### register_dataset(labels, dataset_name, dataset_path)
```python
def register_dataset(labels, dataset_name, dataset_path):
    """
    Registers a COCO formatted version of annotations with detectron2

    Args:
        String[] labels: Array of unique labels
        String dataset_name: model name as given by user
        String dataset_path: path to the new subdirectory with the dataset

    Returns:
        Hashmap microfacies_metadata: Hashmap of dataset metadata as {name={datasetname}, thing_classes=[labels]}. See add_key_value_to_dict() <-- this is important and may be a source of bugs
        String (dataset_name + "_train"): Full path to the training folder
    """
```

##### train_and_save(metadata, training_supplies, labels, model_name)
```python
def train_and_save(metadata, training_supplies, labels, model_name):
    """
    Trains a detectron2 model on given training data

    Args:
        hashmap metadata: Contains {name="modelName", thing_classes=[labels]}, thing_classes is sometimes lost durring training, so metadata is saved to loadModelMetadata.txt
        String training_supplies: full path to the training dataset
        String[] labels: array of labels/classes on which to train model
        String model_name: Subset of training_supplies, just the parent dir or model name
    Returns:
        None
    """
```
___
### Run_Image_Analysis <a name="Run_Image_Analysis"></a>

#### Description
Run_Image_Analysis.py Takes in a sample image and runs an ML pipeline to analyze the constituents and background of the sample. Currently the code only runs detectron2 instance segmentation. However, a little sneak peak, and I didnt tell you this, but a future update will feature a pipeline.txt file and a local_model_zoo folder in the code folder where users can construct ML pipelines from the models they trainined or written themselves. The purpose of this would be to enable contituent classification and classification of background objects (matrices) for the most accurate image classification yet. This approach will also enable users to make the ML as simple or complicated as they would like./
Run_Image_Analysis.py also has some tunable parameters and other bells and whistles that can be changed or deleted

| param                 |  Explaination  |
| --------------------- | -------------- |
| image_type = Torch_Interface.get_image_type(image) | Runs a torchvision resnet image classification to determine the ML pipeline. This will be depricated in the sneaky upcomming update as it is a pretty bad solution to the problem of ML pipelines. To just ensure that detectron2 Instance segmentation is run every time, replace this with image_type = "Constituent"|
| 'thresh' in DETECT_OBJECTS() | Detectron2 testing threshold, effectivly the sensitivity of the model predictions, 0.1 by default. Higher number = higher sensitivity classification |

#### Methods

##### DETECT_OBJECTS(image, model_name, thresh)
```python
def DETECT_OBJECTS(image, model_name, thresh):
    """
    Runs a detectron2 DefaultPredictor on a custom configuration build on a pretrained model defines by model_name

    Args:
        String image: Full path to the sample image on which predictions will be made.
        String model_name: input by user in command line
        double thresh: testing threshold (see tunable parameters)
    Returns:
        None
    """
```

##### DETECT_BACKGROUND()
```python
def DETECT_BACKGROUND():
    """
    Currently unsupported. Background classification using linear combination, k-clustering or PCA

    Args:
        None
    Returns:
        None
    """
```
___

## Miscellaneous <a name="Miscellaneous"></a>
Further explainations of important code, messy/poorly written code, links to further reading and tutorials, troubleshooting, next steps with this code.

### Preclassifier <a name="Preclassifier"></a>
The preclassifier module is an initial attempt at sorting sample images with torchvision resnet18 to avoid poor constituent classification by running detectron2 on packstones or homogenous mustones with few interclasts. It works pretty well but only for the ODP866a data so its recomended that you disconnnect the preclassifier. See [Run_Image_Analysis](#Run_Image_Analysis) on how to disconnect the Preclassifier. As for tunable parameters, this torchvision code supports a lot less testing customization than detectron2 (and I also spent far less time on this than the other ML). I don't fully understand the preclassifier, here is my proposed soultion.

![For readme (4)](https://github.com/user-attachments/assets/1b9b7e9c-4bc8-4716-b7ae-f4c3fce078fa)

### Detectron2 Useful Info <a name="Detectron2_Usefull_Info"></a>

The detectron2 repo can be found [here](https://github.com/facebookresearch/detectron2/tree/main). Everything necesary to the project can be found either in this Readme or in their tutorial ipynb.

In order to avoid including a correct detectron2 build in the repo and wasting space, the code installs detectron2 and dependencies automatically to your device. Some modules, like detectron2 and torchvision are best installed and run through a conda environment. With anaconda installed simply run the following in the terminal/CL:

    # create environment geo with python version 3.9 installed on
    conda create -n geo python=3.9
    # activate environment, you can now install the different packages in your newly created environment.
    conda activate geo
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

And install detectron2

    git clone git@github.com:facebookresearch/detectron2.git # if this does not work, download detectron2 repo as zip file.
    cd detectron 2
    pip install -e .

Let's double check we can both import pytorch and detectron2 by running python with the following command:

    python

    import torch
    import detectron2
    # if both works, let's just exit
    exit()

Then when you download and want to run code in the PixLab repo, simple cd into the code folder and run

    conda activate geo

This is an obnoxious step, and, while it isn't strictly required to run the code in the repo, it will lead to fewer errors


### Documented Errors, Bugs, and, fixes <a name="Documented_Errors_Bugs_and_fixes"></a>

This will be updated in Production testing

# Citations <a name="Citations"></a>
This repository leverages a variety of external packages and repositories cited here

| Package               |  Use  |
| --------------------- | -------------- |
| [Colorama](https://pypi.org/project/colorama/) | For coloring text in output for readability |
| [matplotlib](https://pypi.org/project/matplotlib/) | For displaying image/graph outputs and some image manipulation with cv2 |
| [xlsxwriter](https://pypi.org/project/XlsxWriter/) | For creating an excel file to track manual annotations |
| [openpyxl](https://pypi.org/project/openpyxl/) | Handles all excel file manipulation in place of xlsxwriter |
| [PIL](https://pypi.org/project/pillow/) | Handles some image manipulation in lieu of cv2 |
| [pathlib](https://pypi.org/project/pathlib/) | Works with os for file movement and creation of subdirectories |
| shutil | Works with os and pathlib for file movement and creation of subdirectories |
| [opencv-python/cv2](https://pypi.org/project/opencv-python/) | For image manipulation, displaying outputs, array manipulation |
| tkinter | Frontend of PixLab.py is constructed from tkinter widgets |
| time | Standard python module for runtime calculations |
| math | Standard python module |
| [numpy](https://pypi.org/project/numpy/) | Standard python package. For array manipulation and some math functionality |
| random | Standard python module |
| os | Standard python module |
| [pytorch](https://pypi.org/project/pytorch/) | dependent of detectron2 |
| [detectron2](https://github.com/facebookresearch/detectron2) | with pytorch/torchvision, see bellow for full citation |

Because PixLab clones the Detectron2 repository rather then redistributing it or modifying it, it doesn't qualify as a Derivative work. Citation for Detectron2 (Wich also links to the repository's relevant license):

    @misc{wu2019detectron2,
      author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                     Wan-Yen Lo and Ross Girshick},
      title =        {Detectron2},
      howpublished = {\url{https://github.com/facebookresearch/detectron2}},
      year =         {2019}
    }

# How to Cite This Repository<a name="Citations2"></a>

See LICENSE.txt for guidlines. While not required, if you wish to cite further use:

    Rodriguez C. 2024, Jul 24. PixLab 1.0.0, Retrieved from [https://github.com/username/repository_name](https://github.com/ColteRodriguez/PixLab/tree/main)

