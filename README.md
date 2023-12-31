# qOBM Cell Classification

Cell segmentation and classification leveraging [SAM](https://github.com/facebookresearch/segment-anything) and [DINOv2](https://github.com/facebookresearch/dinov2) for qOBM T-cell images

## Installation

The code requires `python>=3.9`. Clone the repository locally, install the dependencies, and install qOBM Classification

```
git clone git@github.com:ruiqic/qOBM_classification.git
cd qOBM_classification
pip install -r requirements.txt
pip install -e .
```

## Getting Started

First, download a SAM [model checkpoint](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints). We recommend `vit_h` for the best segmentation.

The expected inputs are arrays of shape `(height, width, 3)` in `.npy` files stored in an `input_directory` like so 

```
input_directory/
├── 0220301_TCell_D5_6wellPlate_60x_45deg_32Hz_4s_dyn2_qOBM_dataDqOBM.npy
├── 0220301_TCell_D5_6wellPlate_60x_45deg_32Hz_5s_dyn1_qOBM_dataDqOBM.npy
├── 0220301_TCell_D5_6wellPlate_60x_45deg_32Hz_5s_dyn3_qOBM_dataDqOBM.npy
├── 0220301_TCell_D5_6wellPlate_60x_45deg_4Hz_5s_dyn2_qOBM_dataDqOBM.npy
├── ...
```

Generate segmnetation masks for these images from the command line to the desired `output_directory`

```
python qOBM_classification/segment.py -i input_directory -o output_directory -cp <path/to/SAM/checkpoint> -v
```

The the overhead of package loading and initialization may take a few minutes, so it is recommended to segment large batches of images at once. The segmentation outputs will have the same filenames as their corresponding inputs. The arrays within are of shape (n_objects, height, width).

