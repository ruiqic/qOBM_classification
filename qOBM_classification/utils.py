import random
import copy
import numpy as np
from PIL import Image
from qOBM_classification.feature_extraction import add_all_masks_boxes, add_all_phasor_boxes

def filter_extremes(phase, vmin, vmax):
    """
    Bound the phase matrix by given range.
    """
    def filt(pixel):
        if pixel < vmin:
            return vmin
        elif pixel > vmax:
            return vmax
        else:
            return pixel
    return np.vectorize(filt)(phase)

def phase2GRAY(phase, vmin=-0.2, vmax=0.5):
    """
    Converts phase to cv2 compliant grayscale matrix.
    
    phase: (x,x) numpy array
        first channel of phasor.
    vmin: float
        minimum value of bound.
    vmax: float
        maximum value of bound.
        
    Returns : (x,x) numpy array
        cv2 compliant grayscale matrix with dtype uint8
    """
    filtered = filter_extremes(phase, vmin, vmax)
    arr = (255*(filtered - np.min(filtered))/np.ptp(filtered)).astype("uint8")
    return np.ascontiguousarray(arr)


def numpy_to_png(phasor_file_path, dst):
    """
    phasor file in npy format
    """
    phasor = np.load(phasor_file_path)
    phase = phasor[:,:,0]
    
    # account for images where channel 2 is phase
    if not (-0.2<phase.mean()<0.2):
        phase = phasor[:,:,2]
        
    gray = phase2GRAY(phase, vmin=-0.2, vmax=0.5)
    img = Image.fromarray(gray , 'L')
    img.save(dst, format='PNG')

def random_sample_list(keys_idxs, sampled_num, seed=None):
    """
    randomly samples from a list
    returns sampled and leftover
    """
    random.seed(seed)
    shuffled = copy.deepcopy(keys_idxs)
    random.shuffle(shuffled)
    return shuffled[:sampled_num], shuffled[sampled_num:]

def parition_list(keys_idxs, selected_idxs):
    selected_idxs = set(selected_idxs)
    selected = []
    unselected = []
    for i in range(len(keys_idxs)):
        if i in selected_idxs:
            selected.append(keys_idxs[i])
        else:
            unselected.append(keys_idxs[i])
    return selected, unselected

def get_foreground_stats(all_masks):
    all_masks_copy = copy.deepcopy(all_masks)
    add_all_phasor_boxes(all_masks_copy, bbox_extend=0)

    fg_pixels = []
    for key in all_masks_copy:
        masks = all_masks_copy[key]["masks"]
        for mask in masks:
            phasor_box = mask["phasor_box"]
            mask_box = mask["phasor_box_no_background"].sum(axis=0).astype(bool)
            fg_pixels.append(phasor_box[:, mask_box])
    
    fg_pixels_np = np.hstack(fg_pixels)
    stats_dict = {}
    stats_dict["min"] = np.percentile(fg_pixels_np, 5, axis=1)
    stats_dict["max"] = np.percentile(fg_pixels_np, 95, axis=1)
    stats_dict["range"] = stats_dict["max"] - stats_dict["min"]
    stats_dict["mean"] = np.mean(fg_pixels_np, axis=1)
    stats_dict["std"] = np.std(fg_pixels_np, axis=1)
    
    return stats_dict

def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle 
    function from R.
    
    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.
    
    Returns
    -------
    start positions, run lengths, run values
    
    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int), 
                np.array([], dtype=int), 
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    
    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]
    
    return starts, lengths, values


def rldecode(starts, lengths, values, minlength=None):
    """
    Decode a run-length encoding of a 1D array.
    
    Parameters
    ----------
    starts, lengths, values : 1D array_like
        The run-length encoding.
    minlength : int, optional
        Minimum length of the output array.
    
    Returns
    -------
    1D array. Missing data will be filled with NaNs.
    
    """
    starts, lengths, values = map(np.asarray, (starts, lengths, values))
    # TODO: check validity of rle
    ends = starts + lengths
    n = ends[-1]
    if minlength is not None:
        n = max(minlength, n)
    x = np.full(n, np.nan)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x

def gaussian_kernel(sigma, size=5):
    """Generate a 1D Gaussian kernel."""
    kernel = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (kernel / sigma) ** 2)
    kernel /= np.sum(kernel)
    return kernel
