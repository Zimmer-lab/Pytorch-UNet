"""
This script provides a graphical user interface (GUI) for converting all images in a selected directory to RGB PNG images.
The converted images will be saved in an output folder created inside the selected folder of images.
"""

import os
from PIL import Image
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def convert_images_to_rgb_png(input_dir):
    """
    Converts all images in the specified directory to RGB PNG images.

    Args:
    input_dir (str): The directory containing images to be converted.

    Returns:
    None
    """
    input_dir = Path(input_dir)
    output_dir = input_dir / 'output'
    
    # Check if the input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return
    
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.glob('*'))
    if not image_files:
        print(f"No image files found in {input_dir}.")
        return

    print(f"Found {len(image_files)} image files. Converting...")
    
    # List all image files in the directory
    for file_path in image_files:
        try:
            # Open the image file
            img = Image.open(file_path)
            
            # Convert the image to RGB if not already in RGB mode
            if img.mode != 'RGB':
                rgb_img = img.convert('RGB')
            else:
                rgb_img = img
            
            # Save the RGB image as PNG to the output directory
            output_file_path = output_dir / (file_path.stem + '.png')
            rgb_img.save(output_file_path, format='PNG')
            print(f"Converted to RGB PNG and saved {output_file_path}")
        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")

    print(f"All images in {input_dir} have been converted to RGB PNG and saved in {output_dir}")
    messagebox.showinfo("Conversion Complete", f"All images have been converted to RGB PNG and saved in {output_dir}")

def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        convert_images_to_rgb_png(folder_selected)

# Set up the GUI
root = tk.Tk()
root.title("Image to RGB PNG Converter")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Select a folder containing images:")
label.pack(pady=5)

button = tk.Button(frame, text="Select Folder", command=select_folder)
button.pack(pady=5)

root.mainloop()
