from colorama import Fore                               # For making console output pretty
import xlsxwriter                                       # Theres literally no reason to use this over openpyxl but its here
import openpyxl

# These should come preloaded with python
import math                                             # For crunching the numbers
import numpy as np                                      # For array manipulation
import random                                           # Purely random num generation
import os
import cv2
from PIL import Image, ImageOps, ImageTk

# matplotlib for displaying the images 
from matplotlib import pyplot as plt
import matplotlib.patches as patches



def grab_random_image(folder_path, helperfilepath):
    # List all files in the specified directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Import the helper file
    f = open(helperfilepath, "r")
    segmented_images = str(f.read())
    f.close()
    
    # Check if the directory is empty
    if not files:
        return None
    
    # Choose a random file from the list
    i = 0
    while(True):
        random_file = files[i]
        if random_file in segmented_images or 'DS' in random_file or '._' in random_file:
            i+=1
            continue
        else:
            image_path = os.path.join(folder_path, random_file)
            i+=1
            break
        if i == len(files - 1):
            print("All Files Segmented")
            break
            
    
    return image_path

def create_subdirectories(home, path):
    training_data_directory = path + '/Training_data'

    # Create the subdir for the image
    image_subdir = training_data_directory + '/All_data'
    import os
    if os.path.exists(image_subdir):
        print(Fore.GREEN + "Image Subdirectories Already Exist! \n...Initializing a spreadsheet to track point counts...\n")

    elif not os.path.exists(image_subdir):
        os.makedirs(image_subdir)
        
        # Create the constituent subdirs for the image
        dirs = [image_subdir + '/Labeled', image_subdir + '/Unlabeled']
        for directory in dirs: 
            if not os.path.exists(directory): 
                os.makedirs(directory) 
                
    if path == home:
        print(Fore.RED + "Warning: Subdirectories have been created. Image segmentation will generate >1000 images. It is recomended that path != the home directory. \n...Initializing a spreadsheet to track point counts...\n")

        
def create_spreadsheet(home):
    def grab_all_image_names(folder_path):
        # List all files in the specified directory
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Check if the directory is empty
        if not files:
            return None

        # Return all image names
        image_names = []
        for file in files:
            if (not '._' in file):
                image_names.append(file[0:7])
                
        return image_names


    spreadsheet_directory = home
    images_directory = spreadsheet_directory + '/Sample_Images'

    row_names = grab_all_image_names(images_directory)
    col_names_const = ['Total_mapped_area', '%Mapped', 'Alteration']

    # Create a workbook and add a worksheet. Don't replace if a spreadsheet already exists!
    spreadsheet_path = spreadsheet_directory + '/Point_Counts.xlsx'
    if (not os.path.exists(spreadsheet_path)):
        workbook = xlsxwriter.Workbook(spreadsheet_path)
        worksheet = workbook.add_worksheet()
        workbook.add_worksheet()

        row = 1
        col = 1
        # Iterate over the data and write it out row by row.
        # Yea this is such a waste of codespace but it works, sue me
        '''
        for sample_name in row_names:
            # Filter out ds.store meta
            if 'DS' not in sample_name:
                worksheet.write(row, 0, sample_name)
                worksheet2.write(row, 0, sample_name)
            row += 1
        '''
        # Add the simple metrics
        for constituent in col_names_const:
            worksheet.write(0, col, constituent)
            col += 1
        '''
        for i in range(len(row_names)):
            for j in range(len(col_names_const)):
                r, c = i+1, j+1
                worksheet.write(r, c, 0)

        for i in range(len(row_names)):
            for j in range(len(col_names_sat)):
                if col_names_sat[j] == ' ':
                    continue
                r, c = i+1, j+1
                worksheet2.write(r, c, 0)
        '''

        workbook.close() 
        print(Fore.GREEN + "Point_Counts.xlsx Created. Setup Complete.")
    else:
        print(Fore.GREEN + "Point_Counts.xlsx Already exists. Setup Complete.")
    

######### For Image segmentation #########

# Saving the segments of the quadtree to /Unlabeled
def slice_segments(in_image, imName, external_drive_path):
    image = cv2.imread(random_image)
    image_informal_name = imName[0:7]
    
    meta_data_filename = str(external_drive_path + "/Training_data" + "/" + image_informal_name + "_meta.txt")
    unlabeled_segments_path = str(external_drive_path + "/Training_data/All_data/Unlabeled/")
    
    # Keep some image metadata for later
    if not os.path.exists(meta_data_filename):  
        f = open(meta_data_filename, "x")
        f.write(str(image.shape[0] * image.shape[1]))
        f.close()

    # partition (redundant)
    qtIm = QTree(int(input("Color threshold value: ")), int(input("Minimum Cell Size is: ")), image)
    qtIm.subdivide()

    # Traverse the tree
    segments = qtIm.readable_children()

    print(Fore.RED + "WARNING: " + str(len(segments)) + " image segments " + " will be saved to " + external_drive_path + "/Training_data/All_data" + "/Unlabeled")
    kill_process = input("Type 'quit' to stop or anything else to continue: ")
    if kill_process == 'quit':
        return
    else:
        for segment, i in zip(segments, range(0, len(segments))):  
            # Save the segments to unlabeled
            im = Image.fromarray(np.array(segment))
            name = str(unlabeled_segments_path + imName[0:7] + '_' + str(i) + '.png')
            im.save(unlabeled_segments_path + '/' + imName[0:7] + '_' + str(i) + '.png')
            

            
# A good dev would put this into different files but...
# Useful for displaying outputs
def printI(img):
    fig= plt.figure(figsize=(20, 20))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.draw()
    plt.pause(0.001)  # Pause to allow the image to display
    
    
def printI2(i1, i2):
    fig= plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(cv2.cvtColor(i1, cv2.COLOR_BGR2RGB))
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(cv2.cvtColor(i2, cv2.COLOR_BGR2RGB))
    
# Beginning of quad-tree implementation
class Node():
    # JRtechs uses implicit pointers by representing the quad-tree as an array. Computationally this is slow and lookup can be 
    # improved to Ω(log4(n)) with a linked implementation however, im unsure how the python backend favors these two cases. both are probably Θ(log4(n)) avg case
    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.children = []
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_points(self):
        return self.points
    
    def get_points(self, img):
        return img[self.x0:self.x0 + self.get_width(), self.y0:self.y0+self.get_height()]
    
    # Energy method used for determining partitions based on color similarity
    def get_error(self, img):
        pixels = self.get_points(img)
        b_avg = np.mean(pixels[:,:,0])
        b_mse = np.square(np.subtract(pixels[:,:,0], b_avg)).mean()
    
        g_avg = np.mean(pixels[:,:,1])
        g_mse = np.square(np.subtract(pixels[:,:,1], g_avg)).mean()
        
        r_avg = np.mean(pixels[:,:,2])
        r_mse = np.square(np.subtract(pixels[:,:,2], r_avg)).mean()
        
        e = r_mse * 0.2989 + g_mse * 0.5870 + b_mse * 0.1140
        
        return (e * img.shape[0]* img.shape[1])/90000000
    
# Quadtree implementation
class QTree():
    def __init__(self, stdThreshold, minPixelSize, img):
        self.threshold = stdThreshold                             # Determines the tolerance for color differences
        self.min_size = minPixelSize                              # Minimum segment size
        self.minPixelSize = minPixelSize
        self.img = img                                            # Image instance
        self.root = Node(0, 0, img.shape[0], img.shape[1])        # Root of the qtree
        self.rgb_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Image copy in rgb for splicing

    def get_points(self):
        return img[self.root.x0:self.root.x0 + self.root.get_width(), self.root.y0:self.root.y0+self.root.get_height()]
    
    def subdivide(self):
        recursive_subdivide(self.root, self.threshold, self.minPixelSize, self.img)
    
    # Display the qtree
    def graph_tree(self):
        fig = plt.figure(figsize=(10, 10))
        plt.title("Quadtree")
        c = find_children(self.root)
        print("Number of segments: %d" %len(c))
        for n in c:
            plt.gcf().gca().add_patch(patches.Rectangle((n.y0, n.x0), n.height, n.width, fill=False))
        plt.gcf().gca().set_xlim(0,img.shape[1])
        plt.gcf().gca().set_ylim(img.shape[0], 0)
        plt.axis('equal')
        plt.show()
        return
    
    # Returns the rbg values for the indices of each segment, an array of 2d arrays
    def readable_children(self):
        c = find_children(self.root)
        segments = []
        
        # Iterate through each node and get the image segment
        # DO NOT use extract_subarray() for this!
        for n in c:
            segment = n.get_points(self.rgb_copy)             
            segments.append(segment)
        return segments
    
    # Used for meta and tracking %Mapped
    def num_children(self):
        return len(find_children(self.root))

    # Left this in from JRtechs but we dont need it. Returns the color grouped image whereby each segment
    # is the mean color of each pixel in the segment. Fun to play around with though
    def render_img(self, thickness = 1, color = (0,0,255)):
        imgc = self.img.copy()
        c = find_children(self.root)
        for n in c:
            pixels = n.get_points(self.img)
            # grb
            gAvg = math.floor(np.mean(pixels[:,:,0]))
            rAvg = math.floor(np.mean(pixels[:,:,1]))
            bAvg = math.floor(np.mean(pixels[:,:,2]))

            # imgc[n.x0:n.x0 + n.get_width(), n.y0:n.y0+n.get_height(), 0] = gAvg
            # imgc[n.x0:n.x0 + n.get_width(), n.y0:n.y0+n.get_height(), 1] = rAvg
            # imgc[n.x0:n.x0 + n.get_width(), n.y0:n.y0+n.get_height(), 2] = bAvg

        if thickness > 0:
            for n in c:
                # Draw a rectangle
                imgc = cv2.rectangle(imgc, (n.y0, n.x0), (n.y0+n.get_height(), n.x0+n.get_width()), color, thickness)
        return imgc
    
# Function to perform the quadtree partitioning, created 4 children for parent node
def recursive_subdivide(node, k, minPixelSize, img):

    if node.get_error(img)<=k:
        return
    w_1 = int(math.floor(node.width/2))
    w_2 = int(math.ceil(node.width/2))
    h_1 = int(math.floor(node.height/2))
    h_2 = int(math.ceil(node.height/2))


    if w_1 <= minPixelSize or h_1 <= minPixelSize:
        return
    x1 = Node(node.x0, node.y0, w_1, h_1) # top left
    recursive_subdivide(x1, k, minPixelSize, img)

    x2 = Node(node.x0, node.y0+h_1, w_1, h_2) # btm left
    recursive_subdivide(x2, k, minPixelSize, img)

    x3 = Node(node.x0 + w_1, node.y0, w_2, h_1)# top right
    recursive_subdivide(x3, k, minPixelSize, img)

    x4 = Node(node.x0+w_1, node.y0+h_1, w_2, h_2) # btm right
    recursive_subdivide(x4, k, minPixelSize, img)

    node.children = [x1, x2, x3, x4]
    
# Recurse through the qtree
def find_children(node):
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += (find_children(child))
    return children

# Splice a segment from the image. NGL I think np already has a function for this
def extract_subarray(array, start_row, start_col, height, width):
    # Get the number of rows and columns in the original array
    num_rows = len(array)
    num_cols = len(array[0])

    # Ensure the subarray boundaries are within the original array's boundaries
    end_row = min(start_row + height, num_rows)
    end_col = min(start_col + width, num_cols)

    # Extract the subarray using list comprehension
    subarray = [row[start_col:end_col] for row in array[start_row:end_row]]

    return subarray


# More for displaying images
def concat_images(img1, img2, boarder=5, color=(255,255,255)):
    img1_boarder = cv2.copyMakeBorder(
                 img1, 
                 boarder, #top
                 boarder, #btn
                 boarder, #left
                 boarder, #right
                 cv2.BORDER_CONSTANT, 
                 value=color
              )
    img2_boarder = cv2.copyMakeBorder(
                 img2, 
                 boarder, #top
                 boarder, #btn
                 0, #left
                 boarder, #right
                 cv2.BORDER_CONSTANT, 
                 value=color
              )
    return np.concatenate((img1_boarder, img2_boarder), axis=1)


def displayQuadTree(img_name, threshold, minCell, img_boarder, line_boarder, line_color=(0,0,255)):
    imgT= cv2.imread(img_name)
    qt = QTree(threshold, minCell, imgT) 
    qt.subdivide()
    qtImg= qt.render_img(thickness=line_boarder, color=line_color)
    file_name = "output/" + img_name.split("/")[-1]
    # cv2.imwrite(file_name,qtImg)
    file_name_2 = "output/diptych-" + img_name[-6] + img_name[-5] + ".jpg"
    hConcat = concat_images(imgT, qtImg, boarder=img_boarder, color=(255,255,255))
    # cv2.imwrite(file_name_2,hConcat)
    print("Number of segments: %d" %qt.num_children())
    printI(hConcat)

# displayQuadTree(random_image, threshold=5, minCell=40, img_boarder=70, line_color=(0,0,0), line_boarder = 1)
    
# Saving the segments of the quadtree to /Unlabeled
def slice_segments(in_image, imName, home_data, thresh, mincell, helperfilepath):
    image = cv2.imread(in_image)
    image_informal_name = imName[0:7]
    
    meta_data_filename = str(home_data + "/Training_data" + "/" + image_informal_name + "_meta.txt")
    
    # Folder for the image segments
    unlabeled_segments_path = str(home_data + "/Training_data/All_data/Unlabeled/")
    labeled_segments_path = str(home_data + "/Training_data/All_data/Labeled/")
    img_dir = os.path.join(unlabeled_segments_path, image_informal_name)
    img_dir_lab = os.path.join(labeled_segments_path, image_informal_name)
    os.mkdir(img_dir)
    os.mkdir(img_dir_lab)
                       
    # Keep some image metadata for later
    if not os.path.exists(meta_data_filename):  
        f = open(meta_data_filename, "x")
        f.write(str(image.shape[0] * image.shape[1]))
        f.close()

    # partition (redundant)
    qtIm = QTree(int(thresh), int(mincell), image)
    qtIm.subdivide()

    # Traverse the tree
    segments = qtIm.readable_children()

    print(Fore.RED + "WARNING: " + str(len(segments)) + " image segments " + " will be saved to " + img_dir)
    kill_process = input(Fore.RED + "Terminate? (Y/N): ")
    if kill_process == 'y' or kill_process == 'Y':
        return
    else:
        for segment, i in zip(segments, range(0, len(segments))):  
            # Save the segments to unlabeled
            im = Image.fromarray(np.array(segment))
            name = str(img_dir + imName[0:7] + '_' + str(i) + '.png')
            im.save(img_dir + '/' + imName[0:7] + '_' + str(i) + '.png')
            
        f = open(helperfilepath, "a")
        f.write(str(imName + ", "))
        f.close()
        print(Fore.GREEN + f"Segmentation sucessful! Ulabled images are in {img_dir}")
    
