import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from natsort import natsorted
from imutils.scopereader import MicroscopeDataReader
import dask.array as da
from skimage.morphology import binary_erosion

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--input_file_path', '-if', metavar='INPUT_FILE_PATH', help='Path to a single input file')
    parser.add_argument('--output_file_path', '-of', metavar='OUTPUT_FILE_PATH', help='Path to a single output file')
    parser.add_argument('--filter_mask', '-f_m', metavar='FILTER_MASK', default = 0, help='set 1 if you want to filter predicted mask for only biggest object')

    return parser.parse_args()

# Function to convert an image to RGB
def to_rgb(img):
    if img is None:
        raise ValueError("Input image is None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a numpy array but got {type(img)}")

    if len(img.shape) == 2:  # Grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:  # Already RGB
        return img
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

def convert_array_to_pil_image(array):
    """Convert a NumPy array to a PIL image."""
    # Ensure the input is a NumPy array
    if not isinstance(array, np.ndarray):
        raise TypeError("Input should be a NumPy array")

    # Convert to RGB mode
    pil_img = Image.fromarray(array).convert('RGB')

    return pil_img

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return np.array(out)


def filter_output_image(mask):
    """
    Isolate the largest connected component in a binary image.

    Parameters:
    - mask (numpy.ndarray): A binary image where the objects are 255 and the background is 0.

    Returns:
    - numpy.ndarray: A binary image with only the largest connected component retained.
    """
    if not isinstance(mask, np.ndarray) or mask.dtype not in [np.uint8, np.bool_]:
        raise ValueError("Input must be a binary image of type np.uint8 or np.bool")

    print("filtering mask for biggest object (worm)")

    # Find all unique elements
    num_labels, labels = cv2.connectedComponents(mask)

    # If no elements found return original mask
    if num_labels == 1:
        return mask

    # Count the pixels for each component, find the largest one, exclude background label 0
    largest_component = np.argmax(np.bincount(labels.flat)[1:]) + 1

    # Create a new binary mask where only the largest component is white
    filtered_image = (labels == largest_component).astype(np.uint8) * 255

    return filtered_image

def main(arg_list=None):
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    print(torch.cuda.is_available())
    print(torch.version.cuda)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    reader_obj = MicroscopeDataReader(args.input_file_path, as_raw_tiff=True, raw_tiff_num_slices=1)
    tif = da.squeeze(reader_obj.dask_array)

    with tiff.TiffWriter(args.output_file_path, bigtiff=True) as tif_writer:
        for i, img in enumerate(tif):
            print(f"\nImage {i} - Initial `img` type: {type(img)}")
            img = np.array(img)
            print(f"\nImage {i} - Initial `img` type: {type(img)}")

            try:
                img = to_rgb(img)
                print(f"Image {i} - After `to_rgb` conversion: {type(img)} with shape {img.shape}")

            except (ValueError, TypeError) as e:
                logging.error(f"Skipping image at index {i} due to conversion error: {e}")
                continue

            img = convert_array_to_pil_image(img)
            print(f"Image {i} - After `convert_array_to_pil_image`: {type(img)}")

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            mask_final = mask_to_image(mask, mask_values)

            if str(args.filter_mask) == "1":
                mask_final = filter_output_image(mask_final)

            # Write the mask to the TIFF writer
            tif_writer.write(mask_final, contiguous=True)


if __name__ == '__main__':
    main(sys.argv[1:])  # exclude the script name from the args when called from shell