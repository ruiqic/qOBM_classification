import torch
from monai.transforms import (Compose, EnsureTyped, ScaleIntensityRanged,
                              NormalizeIntensityd, RandRotated, CenterSpatialCropd, 
                              RandFlipd, RandGaussianNoised, RandGaussianSmoothd, 
                              RandAdjustContrastd, CastToTyped, RandAffined)
from monai.data import CacheDataset, Dataset
from qOBM_classification.utils import get_foreground_stats

def make_phasor_transforms(device, stats_dict, label_float_type):
    if label_float_type:
        type_cast = CastToTyped(keys=["image", "label"], dtype=[torch.float, torch.float])
    else:
        type_cast = CastToTyped(keys=["image", "label"], dtype=[torch.float, torch.long])
    
    transforms = Compose([        
        
        
        # clip intensity
        #ScaleIntensityRanged(keys="image", a_min=stats_dict["min"], a_max=stats_dict["max"], 
        #                     b_min=[0,0,0], b_max=[1,1,1], clip=True, dtype=None),

        # normalization
        #NormalizeIntensityd(keys="image", subtrahend=((stats_dict["mean"]-stats_dict["min"])/stats_dict["range"]).tolist(), 
        #                    divisor=([stats_dict["std"]/stats_dict["range"]]).tolist(), channel_wise=True),
        NormalizeIntensityd(keys="image", subtrahend=stats_dict["mean"].tolist(), 
                            divisor=stats_dict["std"].tolist(), channel_wise=True),
        
        # rotation
        #RandRotated(keys="image", range_x=3.14, prob=1, 
        #           mode="bilinear", padding_mode="border", dtype=None),
        
        # random flip
        RandFlipd(keys="image", prob=0.5, spatial_axis=1),
        
        # convert to tensor and send to GPU
        EnsureTyped(keys=["image", "label"], data_type="tensor", track_meta=False, device=device),
        
        # affine
        RandAffined(keys="image", prob=1, rotate_range=3.14, shear_range=0.3, translate_range=30, 
                    scale_range=0.2),
        
        # center crop to final resultion of 168, might not fit all masks in
        CenterSpatialCropd(keys="image", roi_size=[168,168]),
        
        
        
        
        # gaussian noise
        #RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),

        # gaussian blurring
        #RandGaussianSmoothd(keys="image", prob=0.1, sigma_x=(0.5,1), sigma_y=(0.5,1), sigma_z=(0.5,1)),

        # gamma 
        #RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),
        
        # cast to final type
        type_cast,
    ])
    return transforms
    
def make_phasor_transforms_inference(device, stats_dict):
    transforms = Compose([        
        # convert to tensor and send to GPU
        EnsureTyped(keys="image", data_type="tensor", track_meta=False, device=device),
        
        # clip intensity
        #ScaleIntensityRanged(keys="image", a_min=stats_dict["min"], a_max=stats_dict["max"], 
        #                     b_min=0, b_max=1, clip=True, dtype=None),

        # normalization
        #NormalizeIntensityd(keys="image", subtrahend=(stats_dict["mean"]-stats_dict["min"])/stats_dict["range"], 
        #                    divisor=stats_dict["std"]/stats_dict["range"], channel_wise=True),
        NormalizeIntensityd(keys="image", subtrahend=stats_dict["mean"].tolist(), 
                            divisor=stats_dict["std"].tolist(), channel_wise=True),
        
        # center crop to final resultion of 168, might not fit all masks in
        CenterSpatialCropd(keys="image", roi_size=[168,168]),
        
        # cast to final type
        CastToTyped(keys="image", dtype=torch.float),
    ])
    return transforms

    
def make_phasor_dataset(all_masks, labeled_keys_idxs, device, stats_dict=None, label_float_type=False):
    data = []
    for key, idx in labeled_keys_idxs:
        mask = all_masks[key]["masks"][idx]
        if "phasor_box_no_background" not in mask:
            raise ValueError("Queried mask has no phasor_box. Use feature_extraction.add_all_phasor_boxes")
        if "class" not in mask:
            raise ValueError("Queried mask has no class. Use feature_extraction.add_class_label or visualize.label_masks_class")
        d = {"image": mask["phasor_box_no_background"], "label": mask["class"]}
        data.append(d)
        
    if stats_dict is None:
        stats_dict = get_foreground_stats(all_masks)
    transforms = make_phasor_transforms(device, stats_dict, label_float_type)
    
    dataset = CacheDataset(data, transforms, num_workers=6)
    return dataset, stats_dict

def make_phasor_dataset_inference(all_masks, keys_idxs, device, stats_dict, with_class=False):
    data = []
    for key, idx in keys_idxs:
        mask = all_masks[key]["masks"][idx]
        if "phasor_box_no_background" not in mask:
            raise ValueError("Queried mask has no phasor_box. Use feature_extraction.add_all_phasor_boxes")
        if with_class:
            if "class" not in mask:
                raise ValueError("Queried mask has no class. Use feature_extraction.add_class_label or visualize.label_masks_class")
            d = {"image": mask["phasor_box_no_background"], "label": mask["class"]}
        else:
            d = {"image": mask["phasor_box_no_background"]}
        data.append(d)

    transforms = make_phasor_transforms_inference(device, stats_dict)
    
    dataset = Dataset(data, transforms)
    return dataset