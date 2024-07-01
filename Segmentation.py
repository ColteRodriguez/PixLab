import utils.SetupAPI as spipy
import time
from pathlib import Path
import os
from colorama import Fore
from utils.SheetAPI import add_new_sample

# Get the home directory
home = input("External Drive Path: ")

home_data = home
helperfilepath = str(home_data + "/FS_helper.txt")

spreadsheetpath = str(home_data + "/Point_Counts.xlsx")

# This will be depricated soon, this approach is just easier for test cases
images_directory = home_data + '/Sample_Images'
# random_image = grab_random_image(home + "/Sample_Images")
random_image = spipy.grab_random_image(images_directory, helperfilepath)
image_name = random_image[-12:]
print(Fore.WHITE + image_name + " Will be segmented \n ...proceeding to segmentation preview...\n")

while(True):  
    threshold = int(input("Color-energy threshold value: "))
    minCell = int(input("Minimum cell size (in px): "))
    spipy.displayQuadTree(random_image, threshold=threshold, minCell=minCell, img_boarder=70, line_color=(0,0,0), line_boarder = 1)
    sat = input("satisfied with this segmentation? (Y/N)")
    if sat == 'Y' or sat == 'y':
        print(f'...Segmenting {image_name}...')
        break
    else:
        print("I'm sorry to hear that. Trying new inputs")
        continue
        
# Ensure no other actions to give the alg time to work
for i in range(20):
    print(Fore.WHITE + '|', end='')
    time.sleep(0.2)
print('\n')

spipy.slice_segments(random_image, image_name, home_data, threshold, minCell, helperfilepath)

add_new_sample(spreadsheetpath, image_name[0:7])
