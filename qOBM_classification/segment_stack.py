import os
import sys
import argparse
import pickle
import scipy
import numpy as np
import tempfile
from PIL import Image
from functools import partial
from tqdm import tqdm
import cv2

from qOBM_classification.mask_generation import get_mask_generator, filter_mask_area_roundness
from qOBM_classification.utils import phase2GRAY, rlencode, rldecode, gaussian_kernel

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Segment qOBM phase stack images")
    parser.add_argument("-i", "--input", dest="input",
                        help="Source directory of phase stack in .npy format", type=str)
    parser.add_argument("-o", "--output", dest="output",
                        help="Destination directory for segmentation outputs", type=str)
    parser.add_argument("-cp", "--checkpoint_path", dest="checkpoint_path", 
                        help="Path to SAM checkpoint", type=str)
    parser.add_argument("-v", "--verbose", dest="verbose", 
                        help="Enable verbose mode",
                        default = False, action=argparse.BooleanOptionalAction)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def segment(input_dir, output_dir, checkpoint_path, verbose):
    if verbose:
        print("Starting segmentation")
        print("Input directory: %s"%input_dir)
        print("Output directory: %s"%output_dir)
        print("SAM checkpoint: %s"%checkpoint_path)
    
    if not os.path.isdir(input_dir):
        raise ValueError("Provided input directory is not valid")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        raise ValueError("Provided output directory is not valid")
    
    mask_generator = get_mask_generator(checkpoint_path)
    
    mask_filterer = partial(filter_mask_area_roundness, min_area=3000, max_area=15000, min_roundness=0.85)
    mask_filterer_loose = partial(filter_mask_area_roundness, min_area=3000, max_area=15000, min_roundness=0.5)
    kernel = gaussian_kernel(sigma=10, size=17)
    
    if verbose:
        print("Loaded pretrained models")
    
    file_list = list(filter(lambda x: x.endswith(".npy"), os.listdir(input_dir)))
    num_files = len(file_list)
    
    if verbose:
        print("Found %d files"%num_files)
            
    for i, file_path in enumerate(file_list):
        full_file_path = os.path.join(input_dir, file_path)
        stack = np.load(full_file_path)
        dim1, dim2, stack_height = stack.shape
        
        if verbose:
            print("Processing %d/%d: %s with %d images"%(i+1, num_files, file_path, stack_height))

        mask_aggregate = np.zeros((dim1, dim2))
        rles = []
        for h in tqdm(range(stack_height), disable=not verbose, total=stack_height, desc="Generating masks"):
            phase = stack[:,:,h]
            gray = phase2GRAY(phase, vmin=-0.2, vmax=0.5)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            masks = mask_generator.generate(image)

            masks_filtered = list(filter(mask_filterer, masks))
            mask_segmentations = [mask["segmentation"] for mask in masks_filtered]
            reduced_mask = np.logical_or.reduce(mask_segmentations).astype(int)
            mask_aggregate += reduced_mask

            masks_filtered_loose = list(filter(mask_filterer_loose, masks))
            mask_segmentations_loose = np.array([mask["segmentation"] for mask in masks_filtered_loose]).astype(np.uint8)
            loose_rle = list(map(lambda x: rlencode(x.reshape(-1)), mask_segmentations_loose))
            rles.append(loose_rle)
            
        flat_tracked_cells = (mask_aggregate > stack_height * 0.5).astype(np.uint8).reshape(-1)
        stack_masks = []
        for rle_list in tqdm(rles, disable=not verbose, total=stack_height, desc="Refining masks"):
            mask_segmentations = []
            for rle in rle_list:
                flat_mask = rldecode(*rle)
                if (flat_mask * flat_tracked_cells).sum() > flat_mask.sum() * 0.8:
                    mask_segmentations.append(flat_mask.reshape((dim1,dim2)))
            stack_masks.append(np.logical_or.reduce(mask_segmentations).astype(np.uint8))

        stacked_masks = np.stack(stack_masks, axis=2)
        smoothed_stacked_masks = scipy.ndimage.convolve1d(stacked_masks.astype(float), weights=kernel, axis=2, mode="nearest")
        stacked_output = (smoothed_stacked_masks > 0.5).astype(np.uint8)

        segmentation_output_path = os.path.join(output_dir, file_path)
        np.save(segmentation_output_path, stacked_output)
        
if __name__ == "__main__" :
    args = parse_args()
    segment(args.input, args.output, args.checkpoint_path, args.verbose)