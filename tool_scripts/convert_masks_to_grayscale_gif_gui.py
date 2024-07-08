"""
This script provides a graphical user interface (GUI) for converting all PNG images in a selected directory to black and white GIFs.
The converted GIFs will be saved in an output folder created inside the selected folder of images.
"""

import os
from PIL import Image
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def convert_to_black_and_white_gif(input_dir):
    """
    Converts all PNG images in the specified directory to black and white GIFs.

    Args:
    input_dir (str): The directory containing PNG images to be converted.

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

    png_files = list(input_dir.glob('*.png'))
    if not png_files:
        print(f"No PNG files found in {input_dir}.")
        return

    print(f"Found {len(png_files)} PNG files. Converting...")

    # List all PNG files in the directory
    for file_path in png_files:
        try:
            # Open the image file
            img = Image.open(file_path)

            # Convert the image to grayscale (if not already in that mode)
            if img.mode != 'L':
                img = img.convert('L')

            # Convert the grayscale image to a black and white indexed GIF
            img = img.convert('1')  # This converts the image to a two-color (black and white) palette

            # Save the black and white image as a GIF to the output directory
            output_file_path = output_dir / file_path.with_suffix('.gif').name
            img.save(output_file_path, format='GIF')
            print(f"Converted to black and white GIF and saved {output_file_path}")
        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")

    print(f"All PNG files in {input_dir} have been converted to black and white GIFs and saved in {output_dir}")
    messagebox.showinfo("Conversion Complete", f"All PNG files have been converted to black and white GIFs and saved in {output_dir}")

def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        convert_to_black_and_white_gif(folder_selected)

# Set up the GUI
root = tk.Tk()
root.title("PNG to Black and White GIF Converter")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Select a folder containing PNG images:")
label.pack(pady=5)

button = tk.Button(frame, text="Select Folder", command=select_folder)
button.pack(pady=5)

root.mainloop()
