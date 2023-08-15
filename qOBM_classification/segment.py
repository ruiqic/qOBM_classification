import os
import sys
import argparse
import pickle
import numpy as np
from io import BytesIO
from qOBM_classification.mask_generation import (get_mask_generator, generate_masks, 
                                                 filter_mask_area_roundness, get_all_viable_masks)
from qOBM_classification.feature_extraction import (add_all_manual_features, add_all_masks_boxes, 
                                                    get_manual_feature_array, get_class_array, add_class_label,
                                                    get_all_keys_idxs, add_all_phasor_boxes)
from qOBM_classification.utils import numpy_to_png
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
    
    
    with open('model_checkpoints/random_forest_v1.pkl', 'rb') as handle:
        rf_data = pickle.load(handle)
    rf_model = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])
    rf_model.fit(rf_data["X"], rf_data["y"])
    mask_generator = get_mask_generator(checkpoint_path)
    
    if verbose:
        print("Loaded pretrained models")
    
    phasor_list = list(filter(lambda x: x.endswith(".npy"), os.listdir(input_dir)))
    num_images = len(phasor_list)
    
    if verbose:
        print("Found %d images"%num_images)
        
    for i, phasor_path in enumerate(phasor_list):
        if verbose:
            print("Processing %d/%d: %s"%(i+1, num_images, phasor_path))
        
        # prepare files
        phasor_file_path = os.path.join(input_dir, phasor_path)
        image_file_path = BytesIO()
        numpy_to_png(phasor_file_path, image_file_path)
        
        # predict
        all_masks = get_all_viable_masks(mask_generator, [image_file_path], [phasor_file_path],
                                         min_area=3000, max_area=15000, min_roundness=0.85, progress=False)
        add_all_manual_features(all_masks)
        keys_idxs = get_all_keys_idxs(all_masks)
        X_test = get_manual_feature_array(all_masks, keys_idxs)
        y_pred = rf_model.predict(X_test)
        add_class_label(all_masks, keys_idxs, y_pred)
        
        # write output
        segmentation_output_path = os.path.join(output_dir, phasor_path)
        (_, v), = all_masks.items()
        out_masks = []
        for mask in v["masks"]:
            c = mask["class"]
            if c == 0:
                continue
            out_mask = mask["segmentation"].astype(int) * c
            out_masks.append(out_mask)
        out_masks = np.array(out_masks)
        np.save(segmentation_output_path, out_masks)
        
if __name__ == "__main__" :
    args = parse_args()
    segment(args.input, args.output, args.checkpoint_path, args.verbose)