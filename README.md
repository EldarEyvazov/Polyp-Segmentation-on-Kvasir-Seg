# KVASIR-SEG Polyp Segmentation Using U-Net Based Models

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

## Results
We evaluated the performance of various U-Net based models on the binary polyp segmentation task. The comparative performance of these models was assessed using metrics such as Dice Coefficient, Intersection over Union (IoU), and pixel accuracy.

| Model               | Loss   | IoU    | Dice   | Accuracy |
|---------------------|--------|--------|--------|----------|
| U-NET               | 0.1767 | 0.7042 | 0.8233 | 0.9480   |
| ResUNet             | 0.3358 | 0.6536 | 0.7868 | 0.9382   |
| Attention UNet (5L) | 0.1508 | 0.7422 | 0.8492 | 0.9440   |
| UNet++ w/ DS        | 0.2465 | 0.7624 | 0.8639 | 0.9489   |

The table above summarizes the performance metrics for each model. The baseline U-Net model provided a solid benchmark with decent accuracy and segmentation quality. ResUNet showed a lower performance, indicating some challenges in capturing complex features. Attention U-Net (5L) improved the segmentation results significantly by focusing on the most relevant features within the images, resulting in a higher IoU and Dice coefficient. UNet++ with deep supervision (DS) demonstrated the best performance across all metrics, showcasing the effectiveness of advanced architecture modifications.

### Sample Results
Below are some sample results illustrating the segmentation performance of the different models. 

<p align="center">
  <img src="images/baseline_unet_results.png" alt="Baseline U-Net Results">
  <img src="images/attention_unet_results.png" alt="Attention U-Net Results">
  <img src="images/unetpp_results.png" alt="UNet++ Results">
</p>
