import os
import sys
import argparse
import pickle
import numpy as np
import tempfile
from PIL import Image

from qOBM_classification.mask_generation import get_mask_generator, get_all_viable_masks
from qOBM_classification.utils import phase2GRAY

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
    
    if verbose:
        print("Loaded pretrained models")
    
    file_list = list(filter(lambda x: x.endswith(".npy"), os.listdir(input_dir)))
    num_files = len(file_list)
    
    if verbose:
        print("Found %d files"%num_files)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        for i, file_path in enumerate(file_list):
            full_file_path = os.path.join(input_dir, file_path)
            stack = np.load(full_file_path)
            stack_height = stack.shape[2]

            if verbose:
                print("Processing %d/%d: %s with %d images"%(i+1, num_files, file_path, stack_height))

            image_file_paths = []
            for h in range(stack_height):

                # prepare files
                image_file_path = os.path.join(temp_dir, f"{h}.png")
                phase = stack[:,:,h]
                gray = phase2GRAY(phase, vmin=-0.2, vmax=0.5)
                img = Image.fromarray(gray , 'L')
                img.save(image_file_path, format='PNG')
                image_file_paths.append(image_file_path)

            # predict
            all_masks = get_all_viable_masks(mask_generator, image_file_paths, None,
                                             min_area=3000, max_area=15000, min_roundness=0.85, 
                                             progress=verbose, reduce_masks=True)

            output_masks = []
            for image_file_path in image_file_paths:
                reduced_mask = all_masks[image_file_path]
                output_masks.append(reduced_mask)

            # write output
            stacked_output = np.stack(output_masks, axis=2).astype(np.uint8)
            segmentation_output_path = os.path.join(output_dir, file_path)
            np.save(segmentation_output_path, stacked_output)
        
if __name__ == "__main__" :
    args = parse_args()
    segment(args.input, args.output, args.checkpoint_path, args.verbose)