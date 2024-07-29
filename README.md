# VASIR-SEG Polyp Segmentation Using U-Net Based Models

## Project Overview
This project focuses on semantic segmentation for the Vision and Cognitive Systems course at UNIPD. We built U-Net based models to analyze their performance on the binary polyp segmentation task. Our approach involved initially testing model performance on the original dataset, followed by training on an augmented dataset, to compare performance achievements effectively. 

We used U-Net as our baseline model and enhanced each model's architecture with advanced techniques such as residual blocks, attention mechanisms, and the relatively recent UNet++ architecture. This systematic approach allowed us to assess the impact of these enhancements on model performance in polyp segmentation.


## Dataset
The Kvasir-SEG dataset used in this project contains 1,000 polyp images and their corresponding ground truth masks, specifically curated for research in medical image segmentation. Each image includes pixel-wise annotations and bounding boxes, providing both localization and segmentation information for polyps. This makes the dataset suitable for both polyp detection and segmentation tasks. 

The images in the dataset vary in resolution from 332x487 to 1920x1072 pixels. Polyps are categorized into three sizes: small, medium, and large, based on their pixel resolution. This variability in size and resolution helps in training models that are robust and generalizable.

To enhance the model performance, we utilized the Albumentations library for data augmentation. This approach helps improve the model's ability to generalize by introducing variations in the training data.

For more information and to access the dataset, please visit the [Kvasir-SEG dataset page](https://datasets.simula.no/kvasir/).


## Models
This repository contains the implementation of the following models:
- **UNET**
- **ResUNET**
- **Attention UNet**
- **UNet++**

