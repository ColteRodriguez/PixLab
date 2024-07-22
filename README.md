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
2. [Functions](#functions)
   - [function1](#function1)
   - [function2](#function2)

---

## Classes

### Point2D

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

### Polygon

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

### Segment

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
