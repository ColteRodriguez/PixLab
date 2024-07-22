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
## Point2D
&nbsp; &nbsp; A standard Point2D class constructed from a double x and double y coordinate\
\
**x(self):**\
&nbsp;&nbsp; Standard instance method to return the x coordinate\
\
**y(self):** \
&nbsp;&nbsp; Standard instance method to return the y coordinate\
\
**isWithin(self, Point2D otherPoint):** \
&nbsp;&nbsp; Returns True if Euclidean distance to otherPoint is less than threshold (THRESHOLD defined in outer scope)\
\
**distance(self, otherPoint):**\
&nbsp;&nbsp; Helper method for isWithin(). Returns Euclidian distance to otherPoint\
\
**toString(self):**\
&nbsp;&nbsp; Standard toString instance method for unit testing in main()\
\

## Polygon
&nbsp; &nbsp; A standard Polygon class constructed from an array of Point2d objects. self.points[] functions like a stack\
\
**isComplete(self):**\
&nbsp;&nbsp; Returns True iff the last point == the first point\
\
**addPoint(self, Point2D point):** \
&nbsp;&nbsp; Push new Point2D object onto self.points[]. If the polygon has > 2 vertices and the new point is within (see isWithin() Point2D), the new point coordinates are set to be identical to the first point in self.points[] in order ot close the polygon shape.\
\
**getRecent(self):** \
&nbsp;&nbsp; Pops the most recently added point from self.points[], returns it, then pushes it back\
\
**removePoint(self):**\
&nbsp;&nbsp; Pops the most recently added point from self.points[] without pushing it back\
\
**getpoints(self):**\
&nbsp;&nbsp; returns self.points[]\
\
**getPointsScaled(self, double sW, double sH):**\
&nbsp;&nbsp; returns self.points[], where each point x, y is scaled by sW and sH respectivly. Used to convert point coordinates in GUI canvas to actual image coordinates\
\
**getTuplePointsScaled(self, sW, sH):**\
&nbsp;&nbsp; returns self.points[]\
