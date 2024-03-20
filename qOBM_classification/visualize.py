import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def mask_to_color_image(mask, color_idx, alpha=0.35):
    """
    converts boolean (H,W) mask to (H,W,4) color mask for plotting
    """
    color = plt.cm.Dark2.colors[color_idx]
    m_plot = np.tile(np.expand_dims(mask, 2), (1,1,4)).astype(float)
    for channel in range(3):
        m_plot[:,:,channel] *= color[channel]
    m_plot[:,:,3] *= 0.35
    return m_plot

def plot_img_with_masks(all_masks, key, figwidth=20):
    """
    takes the output of mask_generation.get_all_viable_masks
    plots the whole image and masks overlaid
    if 'mask' has 'class' key, it will show different class
    masks with different colors
    
    """
    image_data = all_masks[key]
    image = image_data["image"]
    masks = image_data["masks"]
    
    
    plt.figure(figsize=(figwidth,figwidth))
    plt.imshow(image)
    ax = plt.gca()
    
    masks_with_classes = {}
    for mask in masks:
        c = mask.get("class", 0)
        if c not in masks_with_classes:
            masks_with_classes[c] = []
        masks_with_classes[c].append(mask["segmentation"])
    
    for c in masks_with_classes:
        m = np.logical_or.reduce(masks_with_classes[c])
        m_plot = mask_to_color_image(m, c)
        ax.imshow(m_plot)

    plt.axis('off')
    plt.show()
    
def plot_boxes(all_masks, key, figwidth=20):
    """
    takes the output of mask_generation.get_all_viable_masks
    with boxes added through feature_extraction.add_all_masks_boxes
    plots the whole image and masks overlaid
    if 'mask' has 'class' key, it will show different class
    masks with different colors
    """
    image_data = all_masks[key]
    masks = image_data["masks"]
    num_boxes = len(masks)
    
    ncols=10
    nrows=num_boxes//ncols+1
    fig, axs = plt.subplots(figsize=(figwidth,figwidth*nrows/10), ncols=ncols, nrows=nrows)
    for i, ax in enumerate(axs.flat):
        if i == num_boxes:
            break
        mask = masks[i]
        ax.imshow(mask["image_box"])
        c = mask.get("class", 0)
        m_plot = mask_to_color_image(mask["mask_box"], c)
        ax.imshow(m_plot)
        ax.title.set_text("mask %d"%i)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
def plot_mask_box(mask, title=None):
    fig, axs = plt.subplots(figsize=(10,5), ncols=2, nrows=1)
    if title is not None:
        fig.suptitle(title)
    axs[0].imshow(mask["image_box"])
    axs[0].axis("off")
    axs[1].imshow(mask["image_box"])
    m_plot = mask_to_color_image(mask["mask_box"], 0)
    axs[1].imshow(m_plot)
    axs[1].axis("off")
    plt.show()

def label_masks_class(all_masks, keys_idxs):
    """
    takes the output of mask_generation.get_all_viable_masks
    with boxes added through feature_extraction.add_all_masks_boxes.
    call this function in Jupyter Notebook to get interactive mask labeling.
    Input integers for class label, input 'undo' do undo previous label.
    Populated the 'class' key of masks.
    
    keys_idxs : (string, int) iterable
        [(filename_key, mask_idx), ...]
    """
    num_masks = len(keys_idxs)
    for i, (key, idx) in enumerate(keys_idxs):
        mask = all_masks[key]["masks"][idx]
        title = "labeling %d/%d"%(i+1, num_masks)
        clear_output(wait=True)
        plot_mask_box(mask, title)
        #time.sleep(0.1)
        y = input("class")
        mask["class"] = int(y)
        