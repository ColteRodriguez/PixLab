# PixLab: Installation and Troubleshooting
With Dependencies!!

# NOTICES:
It is suuuuuper important to run Setup.py at least once before running the code on a new devide regardless of whether or not the project folder is already configured. See Setup.py most recent commit -- "auto-install packages"

## Also
It is entirely feasible to run the code without the external drive so long as you have a "Project" folder somewhere with the appropriate configuration (see video). Transfering data and running the app across other devices works the same, but would require zipping the "Project" folder and sending it to the device

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


# DOCUMENTATION

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
A tkinter dependecy udes to display options of annotations in the annotation window to keep naming consistent. Instance vars:

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
