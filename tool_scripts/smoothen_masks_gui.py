import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry, Button

"""
This script smooths the edges of binary mask images using OpenCV and provides a GUI to select the source folder, smoothing factor, and lower threshold.
Processed images are saved in a new folder named 'smoothed_masks' within the source folder.

Usage:
1. Run the script.
2. Use the GUI to select the source folder containing binary mask images.
3. Enter the smoothing factor (must be an odd number).
4. Enter the lower threshold value (default is 50).
5. Click the "Process Masks" button.
6. The smoothed images will be saved in 'smoothed_masks' within the source folder.
"""

def smooth_edges(mask, smooth_factor, lower_threshold):
    # Apply GaussianBlur to smooth the edges of the binary mask
    smoothed_mask = cv2.GaussianBlur(mask, (smooth_factor, smooth_factor), 0)
    _, binary_smoothed_mask = cv2.threshold(smoothed_mask, lower_threshold, 255, cv2.THRESH_BINARY)
    return binary_smoothed_mask

def process_masks(source_folder, smooth_factor, lower_threshold):
    # Create output folder within the source folder
    output_folder = os.path.join(source_folder, 'smoothed_masks')
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            file_path = os.path.join(source_folder, filename)
            print(f"Processing: {file_path}")
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                smoothed_mask = smooth_edges(mask, smooth_factor, lower_threshold)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, smoothed_mask)
                print(f"Saved: {output_path}")
            else:
                print(f"Could not read image: {file_path}")

    messagebox.showinfo("Process Completed", "All masks have been processed and saved in 'smoothed_masks' folder.")
    print("All masks have been processed and saved in 'smoothed_masks' folder.")

def select_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        folder_label.config(text=f"Selected Folder: {folder_selected}")
        process_button.config(state=tk.NORMAL)
        print(f"Selected folder: {folder_selected}")

def start_processing():
    try:
        smooth_factor = int(smooth_factor_entry.get())
        if smooth_factor % 2 == 0:
            raise ValueError("Smoothing factor must be an odd number.")
        lower_threshold = int(lower_threshold_entry.get())
        source_folder = folder_label.cget("text").replace("Selected Folder: ", "")
        print(f"Starting processing with smoothing factor: {smooth_factor} and lower threshold: {lower_threshold}")
        process_masks(source_folder, smooth_factor, lower_threshold)
    except ValueError as e:
        messagebox.showerror("Error", str(e))
        print(f"Error: {str(e)}")

# Create GUI using tkinter
root = tk.Tk()
root.title("Binary Mask Smoothing")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

select_button = tk.Button(frame, text="Select Folder with Binary Masks", command=select_folder)
select_button.grid(row=0, column=0, columnspan=2, pady=5)

folder_label = Label(frame, text="Selected Folder: None")
folder_label.grid(row=1, column=0, columnspan=2, pady=5)

smooth_factor_label = Label(frame, text="Smoothing Factor (odd number):")
smooth_factor_label.grid(row=2, column=0, pady=5)

smooth_factor_entry = Entry(frame)
smooth_factor_entry.grid(row=2, column=1, pady=5)

lower_threshold_label = Label(frame, text="Lower Threshold:")
lower_threshold_label.grid(row=3, column=0, pady=5)

lower_threshold_entry = Entry(frame)
lower_threshold_entry.grid(row=3, column=1, pady=5)
lower_threshold_entry.insert(0, "50")  # Set default value to 50

process_button = Button(frame, text="Process Masks", command=start_processing, state=tk.DISABLED)
process_button.grid(row=4, column=0, columnspan=2, pady=10)

root.mainloop()
