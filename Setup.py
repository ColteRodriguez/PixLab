import SetupAPI as spipy

from colorama import Fore                               # For making console output pretty
import matplotlib.pyplot as plt                         # For quadtree demos
import xlsxwriter                                       # Theres literally no reason to use this over openpyxl but its here
import openpyxl
from PIL import Image, ImageOps, ImageTk                # For image manipulation when cv has problems
from pathlib import Path                                # for file manipulation
import shutil                                           # For moving files
import cv2                                              # For image manipulation when PIL has problems

# These should come preloaded with python
import math                                             # For crunching the numbers
import numpy as np                                      # For array manipulation
import random                                           # Purely random num generation
import os

# Define home and external drive paths
home = str(Path(os.getcwd()).parent.parent)
external_drive_path = input("Will you be storing data on an external drive? (Y/N)")

if external_drive_path == 'Y' or external_drive_path == 'y':
    external_drive_path = input("External drive path is: ")
else:
    external_drive_path = home  
home_data = external_drive_path

# Save the project folder locally
navigation = str(external_drive_path + "/NAV_helper.txt")
f = open(navigation, "w")
f.write(str(external_drive_path))
f.close()

# Store the segmented images in a helper file
helperfilepath = str(external_drive_path + "/FS_helper.txt")
if os.path.exists(helperfilepath):
    print(Fore.GREEN + "FS_helper exists. \n ...Initializing subdirectories...\n")
else:
    f = open(helperfilepath, "w")
    f.close()
    print(Fore.GREEN + "FS_helper created. \n...Initializing subdirectories...\n")

# Create Subdirectories if they dont exist
spipy.create_subdirectories(home, home_data)

# Create the Spreadsheet if it doesn't exist
spipy.create_spreadsheet(home_data)

