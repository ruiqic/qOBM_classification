# qOBM Cell Classification

Cell segmentation and classification leveraging [SAM 2](https://github.com/facebookresearch/segment-anything-2) and [DINOv2](https://github.com/facebookresearch/dinov2) for qOBM T-cell images

## Installation

The code requires `python>=3.10`, `torch>=2.3.1` and `torchvision>=0.18.1`. Follow [these instructions](https://pytorch.org/get-started/locally/) to install `torch` and a matching version of `torchvision`. It is recommended to install `numpy<2.0` to avoid incompatibility with many other packages.


Clone the repository locally, install the remaining dependencies, and install qOBM Classification

```console
git clone https://github.com/ruiqic/qOBM_classification.git
cd qOBM_classification
pip install -r requirements.txt
pip install -e .
```

## Getting Started

First, download a SAM 2 [model checkpoint](https://github.com/facebookresearch/segment-anything-2/tree/main?tab=readme-ov-file#download-checkpoints). We recommend `sam2_hiera_large.pt` for the best segmentation.

### Phasor Segmentation and Classification

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

```console
python qOBM_classification/segment.py -i input_directory -o output_directory -cp <path/to/SAM/checkpoint> -v
```

The the overhead of package loading and initialization may take a few minutes, so it is recommended to segment large batches of images at once. The segmentation outputs will have the same filenames as their corresponding inputs. The arrays within are of shape (n_objects, height, width).

### Phase Stack Segmentation

The expected inputs are arrays of shape `(height, width, n)`, where `n` is the number of images in the stack. Each stack can have a different `n`. Files should be stored as `.npy` in an `input_directory`, like so 

```
input_directory/
├── TCell_stack_1.npy
├── TCell_stack_2.npy
├── ...
```

Generate segmnetation masks for these files from the command line to the desired `output_directory`

```console
python qOBM_classification/segment_stack.py -i input_directory -o output_directory -cp <path/to/SAM/checkpoint> -v
```

The segmentation outputs will have the same filenames and array shapes as their corresponding inputs. The arrays have data type `uint16` with `0` representing background and positive integers representing cells, with the value being the unique ID of each tracked cell.

If running out of GPU memory, try using the `--resize` argument to reduce image size during model inference. This reduces memory consumption and speeds up the segmentation at the cost of less precise masks. The output masks will have the same size as the original input. Another option is to reduce the `--batchsize`, which limits the number of masks propagated by the model at the same time, but may require more batches.

Example command that reduces image height and width to 3/4 during inference and using batch size of 20:

```console
python qOBM_classification/segment_stack.py -i input_directory -o output_directory -cp <path/to/SAM/checkpoint> -v --resize 0.75 --batchsize 20
```
