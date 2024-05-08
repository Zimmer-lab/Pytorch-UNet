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

def convert_array_to_pil_image(full_img):
    """Convert a NumPy array to a PIL image."""
    if isinstance(full_img, np.ndarray):
        if full_img.ndim == 2:
            # Grayscale image
            pil_img = Image.fromarray(full_img, mode='L')
        elif full_img.ndim == 3:
            if full_img.shape[2] == 3:
                # RGB image
                pil_img = Image.fromarray(full_img, mode='RGB')
            elif full_img.shape[2] == 4:
                # RGBA image
                pil_img = Image.fromarray(full_img, mode='RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {full_img.shape[2]}")
        else:
            raise ValueError(f"Unsupported image array shape: {full_img.shape}")
    else:
        pil_img = full_img

    return pil_img

def save_intermediate_image(img_array, filename):
    img_pil = Image.fromarray(img_array)
    img_pil.save(filename)

def main(arg_list=None):
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    output_dir = os.path.dirname(args.output_file_path)

    reader_obj = MicroscopeDataReader(args.input_file_path, as_raw_tiff=True, raw_tiff_num_slices=1)
    tif = da.squeeze(reader_obj.dask_array)
    with tiff.TiffWriter(args.output_file_path, bigtiff=True) as tif_writer:
        for i, img in enumerate(tif):
            print(f"\nImage {i} - Initial `img` type: {type(img)}")
            img = np.array(img)
            print(f"\nImage {i} - Initial `img` type: {type(img)}")

            initial_png_path = os.path.join(output_dir, f'image_{i}_initial.png')
            save_intermediate_image(img, initial_png_path)

            try:
                img = to_rgb(img)
                print(f"Image {i} - After `to_rgb` conversion: {type(img)} with shape {img.shape}")

                rgb_png_path = os.path.join(output_dir, f'image_{i}_rgb.png')
                save_intermediate_image(img, rgb_png_path)

            except (ValueError, TypeError) as e:
                logging.error(f"Skipping image at index {i} due to conversion error: {e}")
                continue

            img = convert_array_to_pil_image(img)
            print(f"Image {i} - After `convert_array_to_pil_image`: {type(img)}")

            pil_png_path = os.path.join(output_dir, f'image_{i}_pil.png')
            img.save(pil_png_path)

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            print(f"Image {i} - After prediction: {type(mask)} with shape {mask.shape}")

            # Ensure the mask is in the correct format (2D, np.uint8)
            mask = mask.astype(np.uint8)

            # Convert mask to a PIL image for saving
            pil_mask = Image.fromarray(mask)

            pil_mask_path = os.path.join(output_dir, f'image_{i}_mask.png')
            pil_mask.save(pil_mask_path)
            print(f"Image {i} - Saved predicted mask image as {pil_mask_path}")

            # Optionally, save the mask as grayscale
            pil_mask_gray_path = os.path.join(output_dir, f'image_{i}_mask_gray.png')
            pil_mask.convert('L').save(pil_mask_gray_path)
            print(f"Image {i} - Saved grayscale mask image as {pil_mask_gray_path}")

            # Write the mask to the TIFF writer
            tif_writer.write(mask, contiguous=True)


if __name__ == '__main__':
    main(sys.argv[1:])  # exclude the script name from the args when called from shell