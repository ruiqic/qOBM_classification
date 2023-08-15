# qOBM Classification

Cell segmentation and classification leveraging SAM and DINOv2 for qOBM T-cell images

## Installation

The code requires `python>=3.9`. Clone the repository locally and install with

```
git clone git@github.com:ruiqic/qOBM_classification.git
cd qOBM_classification; pip install -e .
```

## Getting Started

First download a SAM [model checkpoint](https://github.com/facebookresearch/segment-anything/tree/main#model-checkpoints). We recommend `vit_h` for the best segmentation.
