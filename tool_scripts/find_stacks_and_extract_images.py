"""
Script to process TIFF stacks from a source folder and export every nth element as PNG images.

Description:
This script reads TIFF image stacks from a specified source folder and its subfolders, 
extracts every nth element from each stack, and saves the extracted images as PNG files 
in a specified output directory.

Usage:
python process_stacks.py --source_folder /path/to/source --names raw_stack.btf --nth_element 15

Arguments:
--source_folder: Path to the source folder containing the TIFF stacks.
--names: List of names of the TIFF stacks to search for in the source folder and its subfolders.
--nth_element: The interval for selecting elements from the stack (default is 15).

Example:
python process_stacks.py --source_folder /path/to/source --names raw_stack.btf --nth_element 15
"""

import numpy as np
import tifffile as tiff
import os
from PIL import Image
import argparse

def find_files_in_subfolders(source_folder, names):
    filepaths = {name: [] for name in names}  # Initialize a dictionary with lists to store file paths
    for root, dirs, files in os.walk(source_folder):
        for name in names:
            for file in files:
                if file == name:  # Check for the exact filename match
                    filepath = os.path.join(root, file)
                    filepaths[name].append(filepath)
                    print(f"Found file: {filepath}")  # Debug print to confirm file discovery
    return filepaths

def process_stacks(image_stack_filenames, nth_element, output_folder):
    # Create export folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Print all found bigtiff stacks
    for stack_type, filepaths in image_stack_filenames.items():
        print(f"Found {len(filepaths)} '{stack_type}' stacks: {filepaths}")

    # Process each stack and store every nth element as a NumPy array
    stack_counter = 1
    for stack_type, filepaths in image_stack_filenames.items():
        for filepath in filepaths:
            selected_elements = []  # List to store every nth element from the current stack

            # Read and collect every nth element from the stack
            with tiff.TiffFile(filepath) as tif:
                total_slices_or_bands = len(tif.pages)

                for i in range(0, total_slices_or_bands, nth_element):  # Loop in steps of nth_element
                    slice_or_band = tif.asarray(key=i)
                    selected_elements.append(slice_or_band)

            # Convert the list of selected elements into a NumPy array
            output_arrays = np.array(selected_elements)

            # Create a numbered subfolder for this specific stack
            stack_folder = os.path.join(output_folder, f"stack_{stack_counter}")
            os.makedirs(stack_folder, exist_ok=True)
            stack_counter += 1

            # Iterate through each image array in the list
            for idx, array in enumerate(output_arrays):
                # Convert the array to an image using PIL (if it's not already an image)
                image = Image.fromarray(array)

                # Convert the image to RGB mode if it's not already in that mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Construct the filename using the index (element position) and base name
                filename = f"{stack_type}_{idx}.png"
                file_path = os.path.join(stack_folder, filename)
                # Save the image under the constructed filename
                image.save(file_path, format='PNG')
                
                print(f"Exported {filename} successfully to {stack_folder}.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process TIFF stacks from a source folder and export every nth element as PNG images.')
    parser.add_argument('--source_folder', type=str, help='Path to the source folder containing the TIFF stacks', required=True)
    parser.add_argument('--names', type=str, nargs='+', help='Names of the TIFF stacks to search for', required=True)
    parser.add_argument('--nth_element', type=int, help='The interval for selecting elements', default=15)

    args = parser.parse_args()

    # Find files in subfolders
    image_stack_filenames = find_files_in_subfolders(args.source_folder, args.names)
    
    if not any(image_stack_filenames.values()):
        print("No files found with the specified names in the source folder.")
        return

    # Set output folder inside the source folder
    output_folder = os.path.join(args.source_folder, "export_unet")

    # Call the processing function
    process_stacks(image_stack_filenames, args.nth_element, output_folder)

if __name__ == '__main__':
    main()
