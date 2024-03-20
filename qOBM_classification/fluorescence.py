import numpy as np
from tqdm import tqdm
from functools import partial
from qOBM_classification.mask_generation import generate_masks, filter_mask_area_roundness, cv2_read_image

def filter_dapi(mask, dapi, threshold, bbox_extend=100):
    segmentation = mask["segmentation"]
    x,y,w,h = mask["bbox"]
    y0 = int(max(0,y-bbox_extend))
    y1 = int(min(2048,y+h+bbox_extend))
    x0 = int(max(0,x-bbox_extend))
    x1 = int(min(2048,x+w+bbox_extend))
    extended_bbox = np.zeros_like(segmentation)
    extended_bbox[y0:y1,x0:x1] = 1
    mask_mean_fluor = np.mean(dapi[segmentation])
    extended_mean_fluor = np.mean(dapi[extended_bbox==1])
    if mask_mean_fluor > threshold and extended_mean_fluor < threshold:
        return True
    else:
        return False


def get_all_live_dead_masks(mask_generator, image_file_paths, phasor_file_paths, dapi_file_paths,
                            min_area=3000, max_area=15000, min_roundness=0.85, stdev_threshold=3,
                            intersect_threshold=0.5,
                            progress=False):
    """
    image_file_paths : list of paths to png image
    phasor_file_paths : list of paths to npy phasor, same order as image_file_paths
    dapi_file_paths : list of paths to dapi png image
    
    returns : dict
        image_file path as key, value is dict with keys {'image', 'phasor', 'masks'}
            image : (H,W,3) grayscale array
            phasor : (H,W,3) array
            masks : list of filtered masks
    """
    
    all_masks = {}
    mask_filterer = partial(filter_mask_area_roundness, min_area=min_area, 
                            max_area=max_area, min_roundness=min_roundness)
    
    for image_file_path, phasor_file_path, dapi_file_path in tqdm(zip(image_file_paths, phasor_file_paths, dapi_file_paths), 
                                                  disable=not progress, total=len(image_file_paths)):
        image = cv2_read_image(image_file_path)
        phasor = np.load(phasor_file_path)
        
        # switch channels for images where channel 2 is phase
        if not (-0.2<phasor[:,:,0].mean()<0.2):
            phasor_copy = phasor.copy()
            phasor[:,:,0], phasor[:,:,2] = phasor_copy[:,:,2], phasor_copy[:,:,0]
        
        masks = generate_masks(mask_generator=mask_generator, image_file_path=image_file_path)
        masks_filtered = list(filter(mask_filterer, masks))
        
        dapi = cv2_read_image(dapi_file_path)
        mean_fluor = np.mean(dapi)
        std_fluor = np.std(dapi)
        threshold = mean_fluor + stdev_threshold*std_fluor
        dapi_masks = generate_masks(mask_generator=mask_generator, image_file_path=dapi_file_path)
        dapi_filterer = partial(filter_dapi, dapi=dapi, threshold=threshold, bbox_extend=100)
        dapi_masks_filtered = list(filter(dapi_filterer, dapi_masks))
        
        combined_dapi_mask = np.logical_or.reduce([m["segmentation"] for m in dapi_masks_filtered])
        
        for mask in masks_filtered:
            area = mask["area"]
            intersection = (mask["segmentation"]*combined_dapi_mask).sum()
            if intersection > area * intersect_threshold:
                mask["class"] = 1
            else:
                mask["class"] = 0
            
        
        
        all_masks[image_file_path] = {"image":image, "phasor":phasor, "masks":masks_filtered}
        
    return all_masks