import numpy as np
import scipy
import skimage
from monai.transforms import Compose, EnsureChannelFirst, BorderPad

def intensity_std(regionmask, intensity_image):
    return np.std(intensity_image[regionmask])
def intensity_range(regionmask, intensity_image):
    return np.ptp(intensity_image[regionmask])
def intensity_kurtosis(regionmask, intensity_image):
    return scipy.stats.kurtosis(intensity_image[regionmask])
def intensity_skew(regionmask, intensity_image):
    return scipy.stats.skew(intensity_image[regionmask])

def extract_manual_features(mask, phasor):
    """
    mask : dict
        with 'segmentation', 'stability_score', and 'predicted_iou' keys
    phasor : (H,W,3) array
    
    returns : 1D array
        array of manual features for the mask
    """
    label = skimage.measure.label(mask["segmentation"])
    regions = skimage.measure.regionprops(label, intensity_image=phasor, 
                                          extra_properties=(intensity_std, intensity_range, 
                                                            intensity_kurtosis, intensity_skew))
    region = regions[0]
    
    dp = [mask["stability_score"], mask["predicted_iou"]]
    dp += list(region.intensity_mean)
    dp += list(region.intensity_std)
    dp += list(region.intensity_range)
    dp += list(region.intensity_kurtosis)
    dp += list(region.intensity_skew)
    dp.append(np.log(region.area))
    dp.append(np.log(region.perimeter))
    dp.append(np.log(region.axis_major_length))
    dp.append(np.log(region.axis_minor_length))
    dp.append(region.solidity)
    dp.append(region.eccentricity)
    dp.append((4 * np.pi * region.area) / (region.perimeter **2))
    
    return np.array(dp)
    
def extract_mask_box(mask, image, bbox_extend):
    """
    mask : dict
        with 'segmentation' and 'bbox' keys
    image : (H,W,3) uint8 array
        grayscale image
    bbox_extend : int
        extend the bounding box on all sides by this amount
    
    returns : tuple
        (image_box : (H1,W1,3), segmentation_mask_box (H1,W1))
    """
    x,y,w,h = mask["bbox"]
    y0 = int(max(0,y-bbox_extend))
    y1 = int(min(2048,y+h+bbox_extend))
    x0 = int(max(0,x-bbox_extend))
    x1 = int(min(2048,x+w+bbox_extend))
    
    image_box = image[y0:y1,x0:x1,:]
    mask_box = mask["segmentation"][y0:y1,x0:x1]
    
    return image_box, mask_box

def extract_phasor_box(mask, padded_phasor, bbox_extend):
    """
    mask : dict
        with 'bbox' keys
    padded_phasor : (3,H,W) float array
        channel first phasor array with extended borders by bbox_extend
    bbox_extend : int
        extend the bounding box on all sides by this amount
    
    returns : array tuple
        phasor_box : (3,H1,W1)
    """
    transforms_mask = Compose([EnsureChannelFirst(channel_dim="no_channel"),
                               BorderPad(spatial_border=bbox_extend, mode="constant")])
    extended_mask = transforms_mask(mask["segmentation"])
    
    x,y,w,h = mask["bbox"]
    x = x + bbox_extend
    y = y + bbox_extend
    y0 = int(y-bbox_extend)
    y1 = int(y+h+bbox_extend)
    x0 = int(x-bbox_extend)
    x1 = int(x+w+bbox_extend)
    
    phasor_box = padded_phasor[:,y0:y1,x0:x1]
    extended_mask_box = extended_mask[:,y0:y1,x0:x1]
    
    return phasor_box, phasor_box*extended_mask_box


def add_all_manual_features(all_masks):
    """
    takes the output of mask_generation.get_all_viable_masks
    adds 'manual_features' key to the masks
    """
    for image_file_path in all_masks:
        image_data = all_masks[image_file_path]
        phasor = image_data["phasor"]
        masks = image_data["masks"]
        
        for mask in masks:
            manual_features = extract_manual_features(mask, phasor)
            mask["manual_features"] = manual_features
    
    
def add_all_masks_boxes(all_masks, bbox_extend=150):
    """
    takes the output of mask_generation.get_all_viable_masks
    adds 'image_box' and 'mask_box' keys to the masks
    """
    for image_file_path in all_masks:
        image_data = all_masks[image_file_path]
        image = image_data["image"]
        masks = image_data["masks"]
        
        for mask in masks:
            image_box, mask_box = extract_mask_box(mask, image, bbox_extend=bbox_extend)
            mask["image_box"] = image_box
            mask["mask_box"] = mask_box
            
def add_all_phasor_boxes(all_masks, bbox_extend=150):
    """
    takes the output of mask_generation.get_all_viable_masks
    adds 'phasor_box' key to the masks
    """
    Compose, EnsureChannelFirst, BorderPad
    transforms = Compose([EnsureChannelFirst(channel_dim=2),
                          BorderPad(spatial_border=bbox_extend, mode="reflect")])
    
    for image_file_path in all_masks:
        image_data = all_masks[image_file_path]
        phasor = image_data["phasor"]
        padded_phasor = transforms(phasor)
        masks = image_data["masks"]
        
        for mask in masks:
            phasor_box, phasor_box_no_background = extract_phasor_box(mask, padded_phasor, bbox_extend=bbox_extend)
            mask["phasor_box"] = phasor_box
            mask["phasor_box_no_background"] = phasor_box_no_background


def get_all_keys_idxs(all_masks):
    """
    takes the output of mask_generation.get_all_viable_masks
    
    returns list of (filename_key, mask_idx) pairs
    """
    all_mask_keys_idxs = []
    for key in all_masks:
        num_masks = len(all_masks[key]["masks"])
        for i in range(num_masks):
            all_mask_keys_idxs.append((key,i))
    return all_mask_keys_idxs
            
def get_manual_feature_array(all_masks, keys_idxs):
    """
    Given a list of keys_idxs, output a list of features
    """
    feature_array = []
    for key, idx in keys_idxs:
        mask = all_masks[key]["masks"][idx]
        if "manual_features" not in mask:
            raise ValueError("Queried mask has no manual features. Use feature_extraction.add_all_manual_features")
        feature_array.append(mask["manual_features"])
    return np.array(feature_array)

def get_class_array(all_masks, keys_idxs):
    """
    Given a list of keys_idxs, output a list of class labels
    """
    class_array = []
    for key, idx in keys_idxs:
        mask = all_masks[key]["masks"][idx]
        if "class" not in mask:
            raise ValueError("Queried mask has no class label. Use visualize.label_masks_class")
        class_array.append(mask["class"])
    return np.array(class_array)

def add_class_label(all_masks, keys_idxs, classes):
    """
    Given a list of keys_idxs and class labels, 
    add the labels to the masks.
    """
    assert(len(keys_idxs) == len(classes))
    for (key, idx), c in zip(keys_idxs,classes):
        mask = all_masks[key]["masks"][idx]
        mask["class"] = c