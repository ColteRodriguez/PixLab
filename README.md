# PixLab: Installation, Troubleshooting, and Documentation
With Dependencies!!

# NOTICES:
Sorry for the visual bugs, our worker bees (me) are working so very hard to debug!
Here's a (very unprofessional) code demo video as a placeholder for README -- nobody reads those anyways right?

[https://www.youtube.com/watch?v=27rF6Az2xL4](https://youtu.be/sTqYdOxsvGE)

## Updates (As of noon 07-02-2024):
- Fixed specific->generic path to spreadsheet in SheetAPI
- Added the Time Machine! View annotations for all facies on saved images to compare annotations with others or to ensure consistent annotating

## Updates (As of noon 07-08-2024):
- Fixed spreadsheet indexing so alteration score is acurate
- Added primary/background constituent tracking to account for all pixels (while only sneding foreground to training data)
- Other minor bug fixes


# FULL DOCUMENTATION

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
### AutoCompleteApp <a name="AutoCompleteApp"></a>

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
Library of methods for tracking annotations in JSON files for detectron2. The configuration mirrors that of "labelme" and thus is not already in COCO format. Converting to COCO is handled by Train_Custom_Dataset.py (see MLtools) however, this method is error prone. I've been dreaming up solutions to just saving JSON files in COCO format but all require substantial code changes and in its current fragile state, I dont want to release an update which makes PixLab dysfunctional.

#### Methods

##### create_json_string(label, points, imagePath, imageHeight, imageWidth)
```python
def create_json_string(label, points, imagePath, imageHeight, imageWidth):
    """
    Creates a JSON file 

    Args:
        String folder_path: path to the sample_image subfolder in the project directory
        String helperfilepath: path to FS_helper.txt to ensure no previously segmented images are returned
        int sheet_num: 0 by default
    Returns:
        String: path to a sample image
    """
```

