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

<div align="center">
  <table style="border-collapse: collapse; width: 80%;">
    <thead>
      <tr style="background-color: #d3d3d3;">
        <th style="padding: 8px; border: 1px solid #ddd;">Model</th>
        <th style="padding: 8px; border: 1px solid #ddd;">Loss</th>
        <th style="padding: 8px; border: 1px solid #ddd;">IoU</th>
        <th style="padding: 8px; border: 1px solid #ddd;">Dice</th>
        <th style="padding: 8px; border: 1px solid #ddd;">Accuracy</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">U-NET</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.1767</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.7042</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.8233</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.9480</td>
      </tr>
      <tr style="background-color: #f9f9f9;">
        <td style="padding: 8px; border: 1px solid #ddd;">ResUNet</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.3358</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.6536</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.7868</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.9382</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;">Attention UNet (5L)</td>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>0.1508</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.7422</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.8492</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.9440</td>
      </tr>
      <tr style="background-color: #f9f9f9;">
        <td style="padding: 8px; border: 1px solid #ddd;">UNet++ w/ DS</td>
        <td style="padding: 8px; border: 1px solid #ddd;">0.2465</td>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>0.7624</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>0.8639</b></td>
        <td style="padding: 8px; border: 1px solid #ddd;"><b>0.9489</b></td>
      </tr>
    </tbody>
  </table>
</div>



The table above summarizes the performance metrics for each model on the augmented dataset. The baseline U-Net model provided a solid benchmark. ResUNet showed lower performance, struggling with feature extraction. Attention U-Net (5L) significantly improved results, achieving higher IoU and Dice scores. UNet++ with deep supervision (DS) outperformed all models, demonstrating the best overall performance with a Dice coefficient of 0.8639 and the highest accuracy of 0.9489, making it our best model compared to the baseline U-Net. Our research found that the performance of every model on the augmented dataset was higher than on the original dataset.

### Sample Results
Below are some sample results illustrating the segmentation performance of the different models. 

<p align="center">
  <img src="images/baseline_unet_results.png" alt="Baseline U-Net Results">
  <img src="images/attention_unet_results.png" alt="Attention U-Net Results">
  <img src="images/unetpp_results.png" alt="UNet++ Results">
</p>
