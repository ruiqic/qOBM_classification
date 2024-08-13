import os
import sys
import argparse
import torch
import numpy as np
import tempfile
import shutil
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import zoom
import cv2
from sam2.build_sam import build_sam2_video_predictor

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
                        help="Path to SAM2 checkpoint", type=str)
    parser.add_argument("-v", "--verbose", dest="verbose", 
                        help="Enable verbose mode",
                        default = False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resize', type=float, default=1,
                    help='Resize image to save memory and speed up segmentation')
    parser.add_argument('--batchsize', type=int, default=50,
                       help="The number of masks concurrently propagated")
    parser.add_argument('--modelcfg', type=str, default="sam2_hiera_l.yaml",
                    help='model_cfg of the corresponding SAM 2checkpoint')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def segment(input_dir, output_dir, checkpoint_path, verbose, resize, batch_size, model_cfg):
    if verbose:
        print("Starting segmentation")
        print("Input directory: %s"%input_dir)
        print("Output directory: %s"%output_dir)
        print("SAM2 checkpoint: %s"%checkpoint_path)
        print(f"Mask propagation batch size: {batch_size}")
    
    if not os.path.isdir(input_dir):
        raise ValueError("Provided input directory is not valid")
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        raise ValueError("Provided output directory is not valid")
        
    if resize <=0 or resize >1:
        raise ValueError("resize should be greater than 0 and not greater than 1")
    elif verbose and resize < 1:
        print(f"resizing inputs with factor {resize}")
    
    
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
        
    video_dir = tempfile.mkdtemp()
    if verbose:
        print(f"making temporary directory: {video_dir}")
        
    model_cfg = "sam2_hiera_l.yaml"
    
    file_list = list(filter(lambda x: x.endswith(".npy"), os.listdir(input_dir)))
    num_files = len(file_list)
    
    if verbose:
        print("Found %d files"%num_files)
            
    try:
        for i, file_path in enumerate(file_list):
            full_file_path = os.path.join(input_dir, file_path)
            stack = np.load(full_file_path)
            if resize < 1:
                stack = zoom(stack, (resize, resize, 1))
                area_factor = resize**2
            dim1, dim2, stack_height = stack.shape
            frames = np.zeros_like(stack).astype(np.uint16)

            if verbose:
                print("Processing %d/%d: %s with %d images"%(i+1, num_files, file_path, stack_height))

            for n in tqdm(range(stack_height), disable=not verbose, total=stack_height, desc="Preprocessing stack"):
                gray = phase2GRAY(stack[:,:,n], vmin=-0.2, vmax=0.5)
                pil_image = Image.fromarray(gray, mode='L').convert('RGB')
                pil_image.save(os.path.join(video_dir, "%05d.jpg"%n), 'JPEG')

            frame_names = [
                p for p in os.listdir(video_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            mask_generator = get_mask_generator(checkpoint_path, model_cfg=model_cfg, device=device)
            all_masks = get_all_viable_masks(mask_generator, [os.path.join(video_dir, frame_names[0])], None,
                                             min_area=3000*area_factor, max_area=15000*area_factor, 
                                             min_roundness=0.85, progress=verbose)
            del mask_generator

            predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
            inference_state = predictor.init_state(video_path=video_dir)
            masks = next(iter(all_masks.values()))["masks"]
            
            num_masks = len(masks)
            n_batches = num_masks // batch_size + 1
            if verbose:
                print(f"Propagating {n_batches} batches of masks")
                
            for batch_start_idx in range(0, num_masks, batch_size):
                predictor.reset_state(inference_state)
                if verbose:
                    print(f"Running batch {batch_start_idx//batch_size+1}/{n_batches}")
                    
                mask_batch = masks[batch_start_idx:batch_start_idx+batch_size]
                for obj_id, mask in enumerate(mask_batch):
                    _, _, _ = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id+batch_start_idx+1,
                        mask=mask["segmentation"],
                    )
                
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    for i, out_obj_id in enumerate(out_obj_ids):
                        frames[:,:,out_frame_idx][(out_mask_logits[i] > 0.0).cpu().numpy()[0]] = out_obj_id
                        
            stacked_output = frames.astype(np.uint16)
            if resize < 1:
                stacked_output = zoom(stacked_output, (1/resize, 1/resize, 1))
            del predictor

            segmentation_output_path = os.path.join(output_dir, file_path)
            np.save(segmentation_output_path, stacked_output)
    finally:
        shutil.rmtree(video_dir)
        if verbose:
            print("Cleaned up temp dir, finishing.")
            
            
if __name__ == "__main__" :
    args = parse_args()
    segment(args.input, args.output, args.checkpoint_path, args.verbose, args.resize, args.batchsize, args.modelcfg)