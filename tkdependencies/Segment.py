from tkinter import Tk, Canvas
from PIL import Image, ImageTk, ImageDraw
from tkdependencies import Polygon as pg
from tkdependencies import Point2D as pt
from colorama import Fore
import numpy as np

class PaintApp:
    '''
    NOTES: All images are resampled to 600x600px to fit the canvas. Polygon objects are constructed relative to these coords as well. 
           I've done this just becasue its easier to avoid visual bugs.
           A reference to the original image is kept however, and Polygon.toStringScaled() as well as Polygon.getPointsScaled() return point coordinates in the 
           frame of the original image using a scaling factor. This can result in 1-2 pixels of uncertainty in any given polygon depending
           on the original image size. For example. Quadtree segmentation produces many square images. upscaling a 60x60px segment will result
           in less uncertainty than downscaling a 1000x1000px segment. Bottom line this is not a great solution and I'd rather just be able to 
           zoom the image, however tkinter does not support a zoom widget and the ones coded in-house all have visual bugs.
    '''
    
    def __init__(self, root, path):
        self.root = root
        self.CW, self.CH = 600, 600
        self.canvas = Canvas(root, width=self.CW, height=self.CW, bg="white")  # Adjust width and height as needed
        self.canvas.pack(fill="both", expand=True)
        self.imageO = Image.open(path) # The original Image
        self.OW, self.OH = self.imageO.size
       
        self.scaleW, self.scaleH = self.CW/self.OW, self.CH/self.OH   # How much we need to divide positions by to get the actual coords
        
        self.image = self.imageO.resize((600,600), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(self.image)
        self.ref = self.img # reference to ward off sneaky garbage collection
        self.canvas.create_image(0, 0, anchor="nw", image=self.img)  # Display image on canvas
        self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.img)
        
        self.overlay = Image.new("RGBA", (self.CW, self.CH))  # Create an RGBA image for transparency
        self.overlay_img = ImageTk.PhotoImage(self.overlay)
        self.canvas.create_image(0, 0, anchor='nw', image=self.overlay_img)
        self.overlay_id = self.canvas.create_image(0, 0, anchor='nw', image=self.overlay_img)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-2>", self.on_backspace)
        
        self.prev_x = None  # Store previous click x-coordinate
        self.prev_y = None  # Store previous click y-coordinate
        
        self.shapes = []
        self.current = None  # The polygon currently being formed
        self.current_polygon = None

    # Append and reset the shape
    def on_return(self):
        # Avoid incomplete shape corner case
        if self.current == None or not self.current.isComplete():
            return
        elif self.current.isComplete():
            self.shapes.append(self.current)
            a = self.current.toStringScaled(self.scaleW, self.scaleH)
            # Reset the current shape to none as well as the previous x and y
            self.current = None
            self.prev_x = None
            self.prev_y = None
            return a
    
    def on_backspace(self, event):
        # Store isCompleted before popping off stack
        
        was_completed = self.current.isComplete()
        if self.current != None:
            last_dot = self.current.removePoint()  # Remove last dot info from list
            self.delete_item(last_dot.x-3, last_dot.y-3, last_dot.x+3, last_dot.y+3)  # Delete dot based on coordinates
            
            if len(self.current.points) == 0: # Corner case if user wants to get rid of shape
                self.current = None
                self.prev_x = None
                self.prev_y = None
                return
            
            if len(self.current.points) >= 1:  # Check if at least 2 dots remain
                prev_dot = self.current.getRecent()  # Get coordinates of the first dot
                prev_dot_x, prev_dot_y = prev_dot.x, prev_dot.y
                self.prev_x = prev_dot_x  # Update previous coordinates. for consistency use the Polygon api not the local vars
                self.prev_y = prev_dot_y
                
                if was_completed:   # Special case for if the shape is complete and we want to un-complete it, repair anchor
                    self.draw_dot(last_dot.x, last_dot.y)  # Its okay to use last_dot here because we have auto-correction
                    self.draw_line(self.current.points[0].x, self.current.points[0].y, self.current.points[1].x, self.current.points[1].y)
                    
                    # Delete the mask layer to avoid visual bugs
                    self.fill_polygon(self.current, 0)
                
                        
    def on_click(self, event):
        x = event.x
        y = event.y

        # if there are no shapes on the board, create a shape
        if self.current == None:
            self.current = pg.Polygon()
            point = pt.Point2D(x, y)
            self.current.addPoint(point)
            self.draw_dot(x, y)  # Visuals
            
            self.prev_x = self.current.getRecent().x  # Update previous coordinates. for consistency use the Polygon api not the local vars
            self.prev_y = self.current.getRecent().y
            
        elif self.current.isComplete():  # Do nothing if shape is complete, waiting for on_return()
            return
        else:
            point = pt.Point2D(x, y)
            self.current.addPoint(point) # Add to the stack
            # print(self.current.getPoints())
            if self.prev_x and self.prev_y:  # Check if previous click exists
                self.draw_line(self.prev_x, self.prev_y, self.current.getRecent().x, self.current.getRecent().y)  # Draw line between previous and current dot
            self.prev_x = self.current.getRecent().x  # Update previous coordinates
            self.prev_y = self.current.getRecent().y

            # If the shape is complete after adding the current point, fill the polygon
            if self.current.isComplete():
                last_point = self.current.getRecent()
                x, y = last_point.x, last_point.y
                self.draw_dot(x, y)
                self.fill_polygon(self.current, 128)
            else:
                self.draw_dot(x, y)



    def draw_dot(self, x, y):
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")  # Draw 10x10 dot

    def draw_line(self, x1, y1, x2, y2):
        self.canvas.create_line(x1, y1, x2, y2, fill="black")  # Draw black line

    # May have some visual bugs because of canvas vs overlay buffering, but still owrks as intended
    def fill_polygon(self, polygon, a):
        # Retrieve the points from the polygon object
        points = []
        for point in polygon.getPoints():
            points.append((point.x, point.y))

        # Draw the filled polygon with transparency on the overlay image
        draw = ImageDraw.Draw(self.overlay, "RGBA")
        draw.polygon(points, fill=(0, 128, 20, a))  # RGBA color with transparency

        # Update the overlay image on the canvas
        self.overlay_img = ImageTk.PhotoImage(self.overlay)
        self.canvas.create_image(0, 0, anchor='nw', image=self.overlay_img)
        
    def killPolygon(self):
        xs, ys = self.current.getPointsIndiv()
        x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        
        self.fill_polygon(self.current, 0)
        self.current = None
        self.prev_x = None
        self.prev_y = None
        
    def delete_item(self, x1, y1, x2, y2):
        # Find items within the rectangle created by dot coordinates (excluding the image)
        items = self.canvas.find_overlapping(x1, y1, x2, y2)
        for item in items:
            if item != self.image_id:  # Check if item is not the image
                bbox = self.canvas.bbox(item)  # Get bounding box of the item
                if bbox:  # Check if bbox exists
                    self.canvas.delete(item)  # Delete the item
                    
        
