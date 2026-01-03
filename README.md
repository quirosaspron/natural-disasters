# Satellite Damage Assessment with Siamese U-Net

This project implements an automated pipeline for post-disaster damage assessment using deep learning and high-resolution satellite imagery. The architecture utilizes a Siamese U-Net with a ResNet50 backbone to perform pixel-wise segmentation across four damage severity levels.

## Overview

Rapid damage assessment is critical for humanitarian aid and disaster response. This model compares pre-disaster and post-disaster satellite images from the xBD dataset to identify structural changes and classify damage intensity.

## Model Architecture

The model is built on a Siamese U-Net framework optimized for change detection:

* **Encoder:** Dual-stream shared ResNet50 (pretrained on ImageNet) extracts high-level spatial features from both image timestamps.
* **Bottleneck:** Concatenates extracted features to create a combined representation of the scene before and after the event.
* **Decoder:** A symmetric upsampling path with skip connections that fuses low-level spatial data with high-level semantic data for precise segmentation.
* **Loss Function:** Custom Focal Loss is implemented to address the significant class imbalance between background pixels and damaged structures.

## Project Structure

```text
damage-assessment-project/
├── train/
│   ├── images/               # Pre- and post-disaster .png files
│   └── labels/               # xView2 format JSON annotations
├── best_dmg_assessment.h5    # Trained model weights
├── siam_unet.py              # Siamese U-Net architecture definition
├── damage_data_generator.py  # Keras data sequence for batch processing
├── damage_mask.py            # Utility to rasterize JSON polygons into masks
├── damage_assessment_training.py # Model training and optimization pipeline
├── damage_assessment_test.py  # Single-pair inference and validation
├── demo.py                   # Batch visualization and overlay generation
└── README.md
```


## Key Components

### Data Processing
The pipeline handles the conversion of WKT (Well-Known Text) polygons from the xView2 JSON labels into rasterized segmentation masks. The data generator ensures thread-safe, memory-efficient loading of image pairs during training.

### Damage Class Mapping
The model predicts four distinct classes of damage:

| Value | Damage Level | Description |
| :--- | :--- | :--- |
| 0 | No Damage | Undamaged structures |
| 1 | Minor Damage | Partially affected, mostly intact |
| 2 | Major Damage | Significant structural failure |
| 3 | Destroyed | Total structural collapse |

## Getting Started

1. **Data Acquisition:**
Download the dataset from xView2.org and place the files in the /train directory following the structure outlined above.

2. **Training:**
Initialize the ResNet50 encoder and begin training by running:
`python damage_assessment_training.py`

3. **Inference and Visualization:**
To run predictions on test imagery and view the damage overlays:
`python demo.py`

## Evaluation

The model's performance is measured using Mean Intersection over Union (mIoU) and pixel-wise accuracy. Focal Loss is used during training to ensure the model remains sensitive to the "Destroyed" and "Major Damage" classes, which are statistically rarer in the dataset.
