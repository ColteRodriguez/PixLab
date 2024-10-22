# External packages
import tkinter as tk
from tkinter import Tk, ttk, Button, Label, filedialog, Toplevel
from PIL import Image, ImageTk  # For loading images
from pathlib import Path
import os
from os import path
import shutil

# Custom packages
from tkdependencies import ScrollableList
import utils.SheetAPI as shipy
from tkdependencies import Segment as sg
from tkdependencies import Polygon as poly
from tkdependencies import Point2D
import utils.JsonEncoder as jcode
import tkdependencies.AutoCompleteApp as ApCollegeBoard

# Define colors
gray = "#EEEEEE"
light_gray = "#403f3f"

# Image paths (replace with your image file paths)
new_image_icon = "Cosmetics/file-explorer-folder-libraries-icon-18298.png"
save_polygon_icon = "Cosmetics/Save_poly_icon.png"
delete_polygon_icon = "Cosmetics/Red_X.svg.png"
save_image_icon = "Cosmetics/save-download-icon-10.png"
info_img = "Cosmetics/26162-200.png"
history_img = "Cosmetics/History_Icons.webp"

# Create the main window
root = tk.Tk()
root.title("PixLab")
root.geometry("1000x800")
root.configure(bg=light_gray)

# Create the navigation bar frame
nav_bar = tk.Frame(root, bg=gray, height=80, highlightbackground="black", highlightthickness=1)
nav_bar.pack(fill=tk.X, side=tk.TOP)  # Pack at the top

def open_sure_window2():
    global secondary_window, rating
    def yess():
        global image, image_frame, secondary_window, rating
        constituent = inputtxt.textbox.get()

        if constituent not in saved_constituents:
            saved_constituents.append(constituent)
            f = open(contituents_file, "a")
            f.write(constituent + ",")
            f.close()
                        
        shipy.update_spreadsheet((Image.open(image).width * Image.open(image).width) - total_mapped_area, constituent, image, rating, data_home + '/Point_Counts.xlsx')
        
        save_image()
        secondary_window.destroy()
        secondary_window = None
        choose = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff")]
        )
        image = choose
        app = None
        display_images()
        
    def noo():
        global image, image_frame, secondary_window
        # Close the secondary window if it exists
        secondary_window.destroy()
        secondary_window = None
        
    # Create a new window if it doesn't already exist or if it has been closed
    secondary_window = tk.Toplevel()
    secondary_window.title("Save Image?:")
    secondary_window.geometry("400x600")
    # Add a label to the secondary window
    label = tk.Label(secondary_window, text="It appears the sample isn't completely annotated! \n Please select a constituent that you have NOT yet \n marked for this image in order to gague a 'Background' primary \n constituent (Micrite, Mud, Pore Space). NOTE that the dominant constituent\n will NOT be counted as a polygon and thus will be invisible in \n the Time Machine, and will not contribute to training data. \n it is purely for tracking constituent percentages.")
    label.pack(pady=20, side=tk.TOP)
    
    inputtxt_text = tk.Label(secondary_window, text="Primary/Dominant Constituent: ")
    inputtxt_text.pack(side=tk.TOP)
    
    # TextBox and label Creation 
    inputtxt = ApCollegeBoard.AutoCompleteApp(secondary_window, saved_constituents)
    
    # Pack up the buttons
    buttons_frame = tk.Frame(secondary_window)
    buttons_frame.pack(side=tk.BOTTOM)
    
    score_text = tk.Label(buttons_frame, text="Alteration Score: ")
    score_text.pack(side=tk.TOP)
    
    def update_rating(value):
        global rating
        rating = value
        
    # Score buttons
    alterations = ['Unaltered', 'Some Alteration (1-30%)', 'Patchy or Moderate Alteration (30-60%)', 'Patchy Nonalteration (60-80%)', 'Completely Altered']
    # Add 4 regular buttons to the right
    for i in range(1, 6):
        button = tk.Button(buttons_frame, text=f"{alterations[i - 1]}", command=lambda value=i: update_rating(value), width = 25)
        button.pack(anchor='e', pady=2)
    
    yes = tk.Button(buttons_frame, text="Save image to '/Labeled'", command=yess)
    yes.pack(side=tk.TOP)
    
    secondary_window.mainloop()
    
def open_sure_window():
    global secondary_window
    def yess():
        global image, image_frame, secondary_window
        save_image()
        secondary_window.destroy()
        secondary_window = None
        choose = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff")]
        )
        image = choose
        app = None
        display_images()
        
    def noo():
        global image, image_frame, secondary_window
        # Close the secondary window if it exists
        secondary_window.destroy()
        secondary_window = None
        
    # Create a new window if it doesn't already exist or if it has been closed
    secondary_window = Tk()
    secondary_window.title("WARNING:")
    secondary_window.geometry("300x300")
    # Add a label to the secondary window
    label = tk.Label(secondary_window, text="Are you sure you want to start a new image? \n This action won't forefit current annotations, \n but the incomplete image will be marked \n as labeled.")
    label.pack(pady=20, side=tk.TOP)
    
    # Pack up the buttons
    buttons_frame = tk.Frame(secondary_window)
    buttons_frame.pack(side=tk.TOP)
    
    yes = tk.Button(buttons_frame, text="Yes. I understand this action will \n autosave the current image to '/Labeled'", command=yess)
    yes.pack(side=tk.TOP)
    no = tk.Button(buttons_frame, text="No. I want to continue anotating this image",command=noo)
    no.pack(side=tk.TOP)
    
    secondary_window.mainloop()
    
# Don't even ask why this isn't an instance method im so done
def get_poly_area(points):
    # Shoelace theorum
    n = len(points)  # Number of vertices
    if n < 3:
        return 0  # A polygon must have at least 3 vertices

    area = 0
    for i in range(n):
        j = (i + 1) % n  # The next vertex
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    area = abs(area) / 2.0
    return area
    
def save_polygon_window():
    global secondary_window, image, saved_constituents, constituents_file
    
    secondary_window = tk.Toplevel()
    secondary_window.title("Save Polygon")
    
    # Set up frames for styling
    text_frame1 = tk.Frame(secondary_window, bg='#323232', height = 100)
    text_frame1.pack(side=tk.TOP)
    
    wget_frame1 = tk.Frame(secondary_window, bg='#323232')
    wget_frame1.pack(pady=10, side=tk.TOP)
    
    # Function for getting external drivet 
    # from textbox and printing it  
    # at label widget 
    def polygonal_sends():
        global image, secondary_window, data_home, rating, saved_constituents, constituents_file, total_mapped_area
        constituent = inputtxt.textbox.get()

        if constituent not in saved_constituents:
            saved_constituents.append(constituent)
            f = open(contituents_file, "a")
            f.write(constituent + ",")
            f.close()
        
        print(get_poly_area(app.current.getPointsJson(app.scaleW, app.scaleH)))
        
        total_mapped_area+=get_poly_area(app.current.getPointsJson(app.scaleW, app.scaleH))
        shipy.update_spreadsheet(get_poly_area(app.current.getPointsJson(app.scaleW, app.scaleH)), constituent, image, rating, data_home + '/Point_Counts.xlsx')
        
        json_file = data_home + '/Training_data/All_data/Labeled/' + image[image.index("Img") + 8:][:-4] + '.json'
        
        # If the json file doesn't exist, create it
        if not path.exists(json_file):
            file = open(json_file, 'w')
            # Encode the parent meta
            file.write(jcode.create_json_string(constituent, app.current.getPointsJson(app.scaleW, app.scaleH), image, app.OW, app.OH))
            file.close()
        else: # If the file does exist
            file = open(json_file, 'r')
            json_string = file.read()
            file.close()
            with open(json_file, 'w') as file:
                new = jcode.add_shape_to_json(json_string, constituent, app.current.getPointsJson(app.scaleW, app.scaleH))
                file.write(new)
                
        f = open(helperfilepath, "a")
        Suspicious_Shape = len(app.current.points) < 5
        a = app.on_return()
        f.write(f"{constituent} shape of type 'Polygon': {a} saved to {json_file}. Suspicious == {str(Suspicious_Shape)} \n")
        f.close()
        secondary_window.destroy()
        secondary_window = None

    
    inputtxt_text = tk.Label(text_frame1, text="Constituent: ")
    inputtxt_text.pack(side=tk.LEFT)
    
    # TextBox and label Creation 
    inputtxt = ApCollegeBoard.AutoCompleteApp(text_frame1, saved_constituents)
    
    score_text = tk.Label(wget_frame1, text="Alteration Score: ")
    score_text.pack(side=tk.LEFT)
    
    def update_rating(value):
        global rating
        rating = value
        
    # Score buttons
    alterations = ['Unaltered', 'Some Alteration (1-30%)', 'Patchy or Moderate Alteration (30-60%)', 'Patchy Nonalteration (60-80%)', 'Completely Altered']
    # Add 4 regular buttons to the right
    for i in range(1, 6):
        button = tk.Button(wget_frame1, text=f"{alterations[i - 1]}", command=lambda value=i: update_rating(value), width = 25)
        button.pack(anchor='e', pady=2)

    # Button Creation 
    printButton = tk.Button(secondary_window, text = "Save", command = polygonal_sends) 
    printButton.pack(side=tk.BOTTOM)
    
    secondary_window.mainloop()
    
def history_window():
    global secondary_window, past_image, history_viewer
    secondary_window = tk.Toplevel()
    past_image = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff")]
        )
    width, height = Image.open(past_image).size
    SW, SH = 600/width, 600/height
    
    def exit_history():
        global secondary_window
        if secondary_window is not None:
            secondary_window.destroy()
            secondary_window, history_viewer, past_image = None, None, None

    '''
    bruh this code is so scuffed ik. Only god and myself 
    knew what was happening when I wrote this.
    '''
    def up(event):
        global history_viewer
        nonlocal SW, SH, text2, objects, i, viewer_frame
        if i < len(objects):
            i+=1
            print(i)
            objects=list(shapes_dict.items())[i]
            viewer_frame.pack_forget()
            viewer_frame.destroy()
            viewer_frame = tk.Frame(secondary_window)
            viewer_frame.pack(side=tk.TOP)
            history_viewer = sg.PaintApp(viewer_frame, past_image)
            # fill each polygon in the window
            for list_of_points in objects[1]:
                polygon = poly.Polygon()
                for coord in list_of_points:
                    polygon.addPoint(Point2D.Point2D(coord[0]*SW, coord[1]*SH))
                history_viewer.fill_polygon(polygon, 128)
            text2.pack_forget()
            text2.destroy()
            text2 = tk.Label(secondary_window, text=f"Currently Viewing: {objects[0]}")
            text2.pack(side=tk.TOP)
             
    def down(event):
        global history_viewer
        nonlocal SW, SH, text2, objects, i, viewer_frame
        if i != 0:
            i-=1
            print(i)
            objects=list(shapes_dict.items())[i]
            viewer_frame.pack_forget()
            viewer_frame.destroy()
            viewer_frame = tk.Frame(secondary_window)
            viewer_frame.pack(side=tk.TOP)
            history_viewer = sg.PaintApp(viewer_frame, past_image)
            # fill each polygon in the window
            for list_of_points in objects[1]:
                polygon = poly.Polygon()
                for coord in list_of_points:
                    polygon.addPoint(Point2D.Point2D(coord[0]*SW, coord[1]*SH))
                history_viewer.fill_polygon(polygon, 128)
            text2.pack_forget()
            text2.destroy()
            text2 = tk.Label(secondary_window, text=f"Currently Viewing: {objects[0]}")
            text2.pack(side=tk.TOP)
            
    # get the coords of all polys
    shapes_dict = jcode.get_shapes(past_image)
    
    # init to the fist constit
    i = 0
    objects = list(shapes_dict.items())[i]
    
    # Enable switching objects with keys
    secondary_window.bind('<Up>', up)
    secondary_window.bind('<Down>', down)
    
    text = tk.Label(secondary_window, text="Use Up/Down arrows to toggle through Constituents")
    text.pack(side=tk.TOP)
    text2 = tk.Label(secondary_window, text=f"Currently Viewing: {objects[0]}")
    text2.pack(side=tk.TOP)
    
    viewer_frame = tk.Frame(secondary_window)
    viewer_frame.pack(side=tk.TOP)
    history_viewer = sg.PaintApp(viewer_frame, past_image)
    # fill each polygon in the window
    for list_of_points in objects[1]:
        polygon = poly.Polygon()
        for coord in list_of_points:
            polygon.addPoint(Point2D.Point2D(coord[0]*SW, coord[1]*SH))
        history_viewer.fill_polygon(polygon, 128)
    
    done_button = tk.Button(secondary_window, text="Done", command=exit_history)
    done_button.pack(side=tk.BOTTOM)
    
    secondary_window.mainloop()
    

def help_window():
    global secondary_window
    secondary_window = tk.Toplevel()
    secondary_window.title("WARNING:")
    secondary_window.geometry("200x200")
    # Add a label to the secondary window
    label = tk.Label(secondary_window, text="Left click to add vertex \n to polygon. Right click \n to delete vertex. ")
    label.pack(pady=20, side=tk.TOP)
    
# Open QGIS: dumpsterfire ed.
def open_secondary_window(frame, filename):
    global app
    app = sg.PaintApp(frame, filename)


def display_images():
    global image, image_frame, total_mapped_area
    
    total_mapped_area = 0
    
    # Delete any current wgets
    if image is not None and (image_frame is not None):
        image_frame.pack_forget()
        image_frame.destroy()
        
    segment = image

    image_frame = tk.Frame(root)
    image_frame.pack()
    
    f = open(helperfilepath, "a")
    f.write(f"New Annotation for {image} \n")
    f.close()
    
    open_secondary_window(image_frame, segment)
    
# Define button functions (replace with actual functionality)
def new_image():
    global image, image_frame, app
    
    # If init
    if image_frame == None or (image_frame and len(app.shapes) == 0):
        choose = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff")]
        )
        image = choose
        display_images()
    else:
        open_sure_window()
    
def time_machine():
    history_window()

def save_polygon():
    print("Save Polygon Button Clicked")
    save_polygon_window()
        


def delete_polygon():
    print("Delete Polygon Button Clicked")
    # get the bounding box for the recent polygon
    app.killPolygon()

def save_image():
    global ThisSession, image, data_home
    
    ThisSession.add_label(image)
    # Move the file to the respective classification
    source_path = image
    destination_path = data_home + '/Training_data/All_data/Labeled'
    shutil.move(source_path, destination_path)
    
    f = open(helperfilepath, "a")
    f.write(f"Annotations for {image} saved and moved to subdirectory '/Training_data/All_data/Labeled' \n")
    f.close()



image = None
image_frame = None
app = None
sure = False
secondary_window=None
data_home = input("Project Directory is: ")
rating = None
saved_constituents = []
total_mapped_area = 0

history_viewer = None
past_image = None

contituents_file = data_home + "/Constituents.txt"
if not os.path.exists(contituents_file):
    f = open(contituents_file, "w")
    f.write(f"")
    f.close()
else:
    f = open(contituents_file, "r")
    saved_constituents = f.read().split(',')
    f.close()
    
print(saved_constituents)
            

# Load images (replace with error handling if images not found)
new_image_img = ImageTk.PhotoImage(Image.open(new_image_icon).resize((40, 40)))
save_polygon_img = ImageTk.PhotoImage(Image.open(save_polygon_icon).resize((50, 50)))
delete_polygon_img = ImageTk.PhotoImage(Image.open(delete_polygon_icon).resize((40, 40)))
save_image_img = ImageTk.PhotoImage(Image.open(save_image_icon).resize((40, 40)))
info_image_img = ImageTk.PhotoImage(Image.open(info_img).resize((40, 40)))
history_img_ref = ImageTk.PhotoImage(Image.open(history_img).resize((40, 40)))

# Create buttons with text and image
button_new_image = tk.Button(nav_bar, text="New Image", image=new_image_img, compound=tk.TOP, command=new_image, borderwidth=0, highlightthickness=0)
button_new_image.pack(side=tk.LEFT, padx=15)

button_save_polygon = tk.Button(nav_bar, text="Save Polygon", image=save_polygon_img, compound=tk.TOP, command=save_polygon, borderwidth=0, highlightthickness=0)
button_save_polygon.pack(side=tk.LEFT, padx=15)

button_delete_polygon = tk.Button(nav_bar, text="Delete Polygon", image=delete_polygon_img, compound=tk.TOP, command=delete_polygon, borderwidth=0, highlightthickness=0)
button_delete_polygon.pack(side=tk.LEFT, padx=15)

button_save_image = tk.Button(nav_bar, text="Save Image", image=save_image_img, compound=tk.TOP, command=open_sure_window2, borderwidth=0, highlightthickness=0)
button_save_image.pack(side=tk.RIGHT, padx=15)

button_save_image2 = tk.Button(nav_bar, text="Help", image=info_image_img, compound=tk.TOP, command=help_window, borderwidth=0, highlightthickness=0)
button_save_image2.pack(side=tk.LEFT, padx=15)

button_history = tk.Button(nav_bar, text="Time Machine", image=history_img_ref, compound=tk.TOP, command=time_machine, borderwidth=0, highlightthickness=0)
button_history.pack(side=tk.LEFT, padx=15)

# Create the sidebar frame
sidebar = tk.Frame(root, bg=gray, width=250, highlightbackground="black", highlightthickness=1)
sidebar.pack(fill=tk.Y, side=tk.RIGHT, pady=5)  # Pack at the right side
sidebar_top = tk.Frame(sidebar, bg=light_gray, width=250, height=20, highlightbackground="black", highlightthickness=1)
sidebar_top_label = tk.Label(sidebar_top, text="This Session", width=20, bg='#787878', fg='black')
sidebar_top.pack(side=tk.TOP)
sidebar_top_label.pack()
ThisSession = ScrollableList.ScrollableListApp(root, sidebar)

from datetime import datetime

helperfilepath = data_home + "/PixLab_Session_Info.txt"
if not os.path.exists(helperfilepath):
    f = open(helperfilepath, "w")
    f.write(f"_init_ at {datetime.now()} \n")
    f.close()
else:
    f = open(helperfilepath, "a")
    f.write(f"New Session at {datetime.now()} \n")
    f.close()
    
root.mainloop()
