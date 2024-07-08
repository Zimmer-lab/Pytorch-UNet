import os
import shutil
from tkinter import Tk, Label, Button, Entry, Checkbutton, IntVar, StringVar, filedialog
from PIL import Image
import numpy as np

# Function to binarize mask images
def binarize_image(image_path, output_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    binarized_array = np.where(image_array != 0, 255, 0).astype(np.uint8)
    binarized_image = Image.fromarray(binarized_array)
    binarized_image.save(output_path)

# Function to process files
def process_files():
    source_folder = folder_path.get()
    binarize = binarize_var.get()
    
    if not source_folder:
        print("Source folder path is required")
        return
    
    imgs_folder = os.path.join(source_folder, 'imgs')
    masks_folder = os.path.join(source_folder, 'masks')
    
    os.makedirs(imgs_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        
        if filename.endswith('_mask.png'):
            if binarize:
                binarize_image(file_path, os.path.join(masks_folder, filename))
            else:
                shutil.move(file_path, os.path.join(masks_folder, filename))
        elif filename.endswith('.jpg'):
            shutil.move(file_path, os.path.join(imgs_folder, filename))
    
    print('Processing complete.')

# Function to browse folder
def browse_folder():
    folder_selected = filedialog.askdirectory()
    folder_path.set(folder_selected)

# Initialize the main window
root = Tk()
root.title("Image and Mask Separator")

# Create GUI components
Label(root, text="Source Folder:").grid(row=0, column=0, padx=10, pady=10)
folder_path = StringVar()
Entry(root, textvariable=folder_path, width=50).grid(row=0, column=1, padx=10, pady=10)

Button(root, text="Browse", command=browse_folder).grid(row=0, column=2, padx=10, pady=10)

binarize_var = IntVar()
Checkbutton(root, text="Binarize Masks", variable=binarize_var).grid(row=1, column=1, padx=10, pady=10)

Button(root, text="Process", command=process_files).grid(row=2, column=1, padx=10, pady=10)

# Run the GUI event loop
root.mainloop()
