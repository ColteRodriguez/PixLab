import math

# Set the pixel tolerance for isWithin
TOLERANCE = 10

# Points to construct segmentations (polygons -- see class Polygon)
class Point2D():
    def __init__(self, x, y):
        # Corner case
        if x == None or x < 0 or y == None or y < 0:
            raise Exception("Object point can not be constructed from null or negative coordinates")
        self.x, self.y = x, y
        
    def x(self):
        return self.x;
        
    def y(self):
        return self.y;   
    
    def isWithin(self, otherPoint):
        if self.distance(otherPoint) <= TOLERANCE:
            return True
        else:
            return False
        
        
    def distance(self, otherPoint):
        return math.sqrt((self.x - otherPoint.x)**2 + (self.y - otherPoint.y)**2)
        
    def toString(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'
    
# Unit Testing 
def main():
    point1 = Point2D(0, 0)
    point2 = Point2D(10, 10)
    
    print(f"Distance from {point1.toString()} to {point2.toString()} is {point1.distance(point2)} pixels away from point 2. this is within the threshold: {point1.isWithin(point2)}")
    
    
if __name__ == "__main__":
    main()