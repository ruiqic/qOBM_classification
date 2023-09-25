import numpy as np
import cv2
import os
import skimage
from tqdm import tqdm
from io import BytesIO
from functools import partial
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def cv2_read_image(path_or_buf):
    if isinstance(path_or_buf, str) and os.path.isfile(path_or_buf):
        img_raw = cv2.imread(path_or_buf)
    elif isinstance(path_or_buf, BytesIO):
        # Get the BytesIO content as bytes
        image_bytes = path_or_buf.getvalue()

        # Convert the bytes to a NumPy array
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode the array using OpenCV 
        img_raw = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        raise ValueError("Invalid path to image")
    return img_raw
        

def get_mask_generator(vit_h_checkpoint, pred_iou_thresh=0.9, stability_score_thresh=0.96, device="cuda"):
    """
    Use sam_vit_h_4b8939.pth from 
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    """
    sam_checkpoint = vit_h_checkpoint
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64, pred_iou_thresh=pred_iou_thresh, 
                                               stability_score_thresh=stability_score_thresh)
    return mask_generator

def generate_masks(mask_generator, image_file_path):
    """
    image file in png format
    
    returns a list of dictionaries, each with keys:
    'segmentation': (H,W) bool array
      'area': int
      'bbox': list [x,y,w,h]
      'predicted_iou': float
      'point_coords': [[x,y]]
      'stability_score': float
      'crop_box': [0, 0, 2048, 2048]},
      
      to extract bbox phasor[y:y+h, x:x+w, :]
    """
    img_raw = cv2_read_image(image_file_path)
    masks = mask_generator.generate(img_raw)
    return masks
    

def filter_mask_area_roundness(mask, min_area=3000, max_area=15000, min_roundness=0.85):
    """
    mask : dict with keys 'area' and 'segmentation'
    
    returns : bool
        True if mask meets criteria, False otherwise
    """
    
    area = mask["area"]
    if area < min_area or area > max_area:
        return False
    
    label = skimage.measure.label(mask["segmentation"])
    regions = skimage.measure.regionprops(label)
    
    # check for disconnected regions
    if len(regions) != 1:
        return False
    
    region = regions[0]
    roundness = (4 * np.pi * region.area) / (region.perimeter **2)
    if roundness < min_roundness:
        return False
    return True

def get_all_viable_masks(mask_generator, image_file_paths, phasor_file_paths,
                         min_area=3000, max_area=15000, min_roundness=0.85,
                         progress=False):
    """
    image_file_paths : list of paths to png image
    phasor_file_paths : list of paths to npy phasor, same order as image_file_paths
    
    returns : dict
        image_file path as key, value is dict with keys {'image', 'phasor', 'masks'}
            image : (H,W,3) grayscale array
            phasor : (H,W,3) array
            masks : list of filtered masks
    """
    
    all_masks = {}
    mask_filterer = partial(filter_mask_area_roundness, min_area=min_area, 
                            max_area=max_area, min_roundness=min_roundness)
    
    for image_file_path, phasor_file_path in tqdm(zip(image_file_paths, phasor_file_paths), 
                                                  disable=not progress, total=len(image_file_paths)):
        image = cv2_read_image(image_file_path)
        phasor = np.load(phasor_file_path)
        
        # switch channels for images where channel 2 is phase
        if not (-0.2<phasor[:,:,0].mean()<0.2):
            phasor_copy = phasor.copy()
            phasor[:,:,0], phasor[:,:,2] = phasor_copy[:,:,2], phasor_copy[:,:,0]
        
        masks = generate_masks(mask_generator=mask_generator, image_file_path=image_file_path)
        masks_filtered = list(filter(mask_filterer, masks))
        all_masks[image_file_path] = {"image":image, "phasor":phasor, "masks":masks_filtered}
        
    return all_masks
    
