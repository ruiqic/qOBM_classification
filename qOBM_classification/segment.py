import os
import argparse
import pickle
import numpy as np
from qOBM_classification.mask_generation import (get_mask_generator, generate_masks, 
                                                 filter_mask_area_roundness, get_all_viable_masks)
from qOBM_classification.feature_extraction import (add_all_manual_features, add_all_masks_boxes, 
                                                    get_manual_feature_array, get_class_array, add_class_label,
                                                    get_all_keys_idxs, add_all_phasor_boxes)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Segment qOBM phasor images")
    parser.add_argument("-i", "--input", dest="input",
                        help="Source directory of phasor images in .npy format", type=str)
    parser.add_argument("-o", "--output", dest="output",
                        help="Destination directory for segmentation outputs", type=str)
    parser.add_argument("-cp", "--checkpoint_path", dest="checkpoint_path", 
                        help="Path to SAM checkpoint", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def segment(input_dir, output_dir, checkpoint_path):
    
    
    
    # load npy into image and npy
    
    # load pretrained mask generator and random forest
    
    # predict 
    
    # write output