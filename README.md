# Polyp Segmentation Using U-Net Based Models on Kvasir-SEG dataset

## Project Overview
This project focuses on semantic segmentation on Kvasir-Seg dataset. We built U-Net based models to analyze their performance on the binary polyp segmentation task. Our approach involved initially testing model performance on the original dataset, followed by training on an augmented dataset, to compare performance achievements effectively. 

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

### Visual Results

The images below illustrate the segmentation results for the original image, U-Net, and UNet++ models. The visual comparison shows how each model identifies and segments the polyps.

<p align="center">
  <img width="782" alt="image" src="https://github.com/user-attachments/assets/17b90a64-7dd5-4a14-9910-62d031390121">
</p>

The images depict the original input (left), U-Net segmentation (middle), and UNet++ segmentation (right). UNet++ shows improved segmentation quality, with more accurate and complete polyp boundaries compared to the baseline U-Net, further validating its superior performance metrics.

## References

1. Asplund, J., Kauppila, J. H., Mattsson, F., & Lagergren, J. (2018). Survival Trends in Gastric Adenocarcinoma: A Population-Based Study in Sweden. *Annals of Surgical Oncology*, 25(9), 2693-2702. doi: [10.1245/s10434-018-6627-y](https://doi.org/10.1245/s10434-018-6627-y). PMID: 29987609. PMCID: PMC6097732.

2. Lopes, G., Stern, M. C., Temin, S., Sharara, A. I., Cervantes, A., Costas-Chavarri, A., Engineer, R., Hamashima, C., Ho, G. F., Huitzil, F. D., Moghani, M. M., Nandakumar, G., Shah, M. A., Teh, C., Manjarrez, S. E. V., Verjee, A., Yantiss, R., & Correa, M. C. (2019). Early Detection for Colorectal Cancer: ASCO Resource-Stratified Guideline. *Journal of Global Oncology*, 5, 1-22. doi: [10.1200/JGO.18.00213](https://doi.org/10.1200/JGO.18.00213). PMID: 30802159. URL: [https://doi.org/10.1200/JGO.18.00213](https://doi.org/10.1200/JGO.18.00213).

3. Zhang, Z., Liu, Q., & Wang, Y. (2018). Road extraction by deep residual UNet. *IEEE Geoscience and Remote Sensing Letters*, 15(5), 749-753. doi: [10.1109/LGRS.2018.2803941](https://doi.org/10.1109/LGRS.2018.2803941).

4. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical Image Computing and Computer-Assisted Intervention* (pp. 234-241). Springer. doi: [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28).

5. Ha, D., Ali, S., Tomar, N. K., Johansen, H. D., Johansen, D., Rittscher, J., Riegler, M. A., & Halvorsen, P. (2021). Real-Time Polyp Detection, Localization and Segmentation in Colonoscopy Using Deep Learning. *IEEE Access*, 9, 40496-40510. doi: [10.1109/ACCESS.2021.3063716](https://doi.org/10.1109/ACCESS.2021.3063716). PMID: 33747684. PMCID: PMC7968127.

6. Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2019). Kvasir-SEG: A Segmented Polyp Dataset. *arXiv preprint arXiv:1911.07069*. URL: [https://arxiv.org/abs/1911.07069](https://arxiv.org/abs/1911.07069).

7. Castellino, R. A. (2005). Computer aided detection (CAD): an overview. *Cancer Imaging*, 5(1), 17-19. doi: [10.1102/1470-7330.2005.0018](https://doi.org/10.1102/1470-7330.2005.0018). PMID: 16154813. PMCID: PMC1665219.

8. Karkanis, S. A., Iakovidis, D. K., Maroulis, D. E., Karras, D. A., & Tzivras, M. (2003). Computer-aided tumor detection in endoscopic video using color wavelet features. *IEEE Transactions on Information Technology in Biomedicine*, 7(3), 141-152. doi: [10.1109/TITB.2003.813794](https://doi.org/10.1109/TITB.2003.813794). PMID: 14518727.

9. Ameling, S., Wirth, S., Paulus, D., Lacey, G., & Vilariño, F. (2009). Texture-Based Polyp Detection in Colonoscopy. *Informatik aktuell*, 346-350. doi: [10.1007/978-3-540-93860-6_70](https://doi.org/10.1007/978-3-540-93860-6_70).

10. Tajbakhsh, N., Shin, J. Y., Gurudu, S. R., Hurst, R. T., Kendall, C. B., Gotway, M. B., & Liang, J. (2016). Convolutional neural networks for medical image analysis: Full training or fine tuning?. *IEEE Transactions on Medical Imaging*, 35(5), 1299-1312. doi: [10.1109/TMI.2016.2535302](https://doi.org/10.1109/TMI.2016.2535302).

11. Shin, H. C., Roth, H. R., Gao, M., Lu, L., Xu, Z., Nogues, I., Yao, J., Mollura, D., & Summers, R. M. (2016). Deep convolutional neural networks for computer-aided detection: CNN architectures, dataset characteristics and transfer learning. *IEEE Transactions on Medical Imaging*, 35(5), 1285-1298. doi: [10.1109/TMI.2016.2528162](https://doi.org/10.1109/TMI.2016.2528162).

12. Bernal, J., et al. (2017). Comparative validation of polyp detection methods in video colonoscopy: Results from the MICCAI 2015 endoscopic vision challenge. *IEEE Transactions on Medical Imaging*, 36(6), 1231-1249. doi: [10.1109/TMI.2017.2664042](https://doi.org/10.1109/TMI.2017.2664042).

13. Ali, S., et al. (2020). An objective comparison of detection and segmentation algorithms for artefacts in clinical endoscopy. *Scientific Reports*, 10(1), 1-15. doi: [10.1038/s41598-020-65550-2](https://doi.org/10.1038/s41598-020-65550-2).

14. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *arXiv preprint arXiv:1505.04597*. URL: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597).

15. Yeung, M., Sala, E., Schönlieb, C. B., & Rundo, L. (2021). Focus U-Net: A novel dual attention-gated CNN for polyp segmentation during colonoscopy. *Computers in Biology and Medicine*, 137, 104815. doi: [10.1016/j.compbiomed.2021.104815](https://doi.org/10.1016/j.compbiomed.2021.104815).

16. Patel, K., Bur, A. M., & Wang, G. (2021). Enhanced U-Net: A Feature Enhancement Network for Polyp Segmentation. In *2021 18th Conference on Robots and Vision (CRV)* (pp. 181-188). doi: [10.1109/CRV52889.2021.00032](https://doi.org/10.1109/CRV52889.2021.00032).

17. Wu, H., Zhao, Z., & Wang, Z. (2023). META-Unet: Multi-Scale Efficient Transformer Attention Unet for Fast and High-Accuracy Polyp Segmentation. *IEEE Transactions on Automation Science and Engineering*. doi: [10.1109/TASE.2023.3292373](https://doi.org/10.1109/TASE.2023.3292373).

18. Oktay, O., Schlemper, J., Le Folgoc, L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N. Y., Kainz, B., Glocker, B., & Rueckert, D. (2018). Attention U-Net: Learning Where to Look for the Pancreas. *arXiv preprint arXiv:1804.03999*. URL: [https://arxiv.org/abs/1804.03999](https://arxiv.org/abs/1804.03999).

19. Zhou, Z., Siddiquee, M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation. *arXiv preprint arXiv:1807.10165*. URL: [https://arxiv.org/abs/1807.10165](https://arxiv.org/abs/1807.10165).

20. Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and Flexible Image Augmentations. *Information*, 11(2), 125. doi: [10.3390/info11020125](https://doi.org/10.3390/info11020125). URL: [https://www.mdpi.com/2078-2489/11/2/125](https://www.mdpi.com/2078-2489/11/2/125).


## Acknowledgments

This project was completed as part of the Vision and Cognitive Systems course at UNIPD. I would like to extend my heartfelt gratitude to my colleague [@AntoniValls](https://github.com/AntonioValls), whose collaboration and contributions were invaluable throughout this project. Together, we worked diligently to explore and enhance the performance of U-Net based models for polyp segmentation.

We are proud to share that our efforts and dedication have been recognized, as we achieved the highest grade for our project. This accomplishment reflects the depth of our research, the effectiveness of our methodologies, and the robustness of our results.


