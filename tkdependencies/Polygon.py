import numpy as np
from tkdependencies import Point2D as pt

###### POLYGON OBJECT IS SO CRUDE WILL UPDATE WHEN I HAVE MORE TIME ##########

# Polygon shapes for segmentation
# This class doesnt handle twists :( Ill just call it a feature and 
# say that the user should know not to twist the polygon -- This def wont come back to bite me...
class Polygon():
    def __init__(self):
        self.points = []
        
    # Determine if the shape is a polygon
    def isComplete(self):
        # Min criteria for polygon
        if len(self.points)>2:
            return self.points[0] == self.points[-1]
        else:
            return False
    
    # Appends a Point2D obj to the shape
    def addPoint(self, point):
        # If the point is null or if this shape is complete
        if point == None:
            raise Exception("Can not append a null point")
        elif self.isComplete():
            print("Can not append new points to a completed polygon")
            return
            
        elif len(self.points) > 2 and point.isWithin(self.points[0]): # If the placed point is near the first point, complete the loop
            point = self.points[0]
            self.points.append(point)
            
        # Or just standard append    
        else:
            self.points.append(point)
    
    def getRecent(self):
        # recent = self.points.pop()    <-- Techinally more computationally sensitive
        return self.points[-1]
        
    def removePoint(self):
        return self.points.pop()
    
    def getPoints(self):
        return self.points
    
    def getPointsScaled(self, sW, sH):
        points_scaled = []
        for point in self.points:
            scaled_point = pt.Point2D(int(point.x/sW), int(point.y/sH))
            points_scaled.append(scaled_point)
        return points_scaled
    
    def getTuplePointsScaled(self, sW, sH):
        points_scaled = []
        for point in self.points:
            scaled_point = (int(point.x/sW), int(point.y/sH))
            points_scaled.append(scaled_point)
        print(points_scaled)
        return points_scaled
    
    # for json conversion 
    def toString(self):
        xString, yString = "\"all_points_x\":[", "\"all_points_y\":["
        
        for i in range(len(self.points)):
            point = self.points[i]
            x=str(point.x)
            y=str(point.y)
            
            if i == len(self.points) - 1:
                xString, yString = xString + x, yString + y
            else:
                xString, yString = xString + x + ',', yString + y + ','
            
        return xString + ']' + ',' + yString + ']'
    
    # for json conversion 
    def toStringScaled(self, scaleW, scaleH):
        xString, yString = "\"all_points_x\":[", "\"all_points_y\":["
        
        for i in range(len(self.points)):
            point = self.points[i]
            x=str(int(point.x / scaleW))
            y=str(int(point.y / scaleH))
            
            if i == len(self.points) - 1:
                xString, yString = xString + x, yString + y
            else:
                xString, yString = xString + x + ',', yString + y + ','
            
        return xString + ']' + ',' + yString + ']'
    
    # for json conversion 
    def getPointsJson(self, scaleW, scaleH):
        return_points = []
    
        for point in self.points:
            non_tuple = [int(point.x / scaleW), int(point.y / scaleW)]
            return_points.append(non_tuple)
            
        return return_points

    def getPointsIndiv(self):
        xs = []
        ys = []
        for point in self.points:
            xs.append(int(point.x))
            ys.append(int(point.y))
            
        return xs, ys
        
        
# Unit Testing 
def main():
    xs = [0,0,100,101,100,3]
    ys = [0,1,100,90,0,3]
    
    shape = Polygon()
    
    for x, y in zip(xs, ys):
        shape.addPoint(pt.Point2D(x, y))
        print(f"shape is complete: {shape.isComplete()}. shape is: {shape.toString()}")
        if x == 101:
            shape.removePoint()
            
    shape.addPoint(pt.Point2D(10, 30))
    
if __name__ == "__main__":
    main()