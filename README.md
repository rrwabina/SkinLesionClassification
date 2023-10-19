# Skin Lesion Classification
A repository of Skin Lesion Classification using Deep Learning for the 2022 Qualifying Examination of PhD students in Data Science for Healthcare and Clinical Informatics

You can access the main Jupyter Notebook in the file:
```
../src/notebooks/main.ipynb
```

## Dependencies
You need to install the prerequisites.
```
pip install -r requirements.txt
```

## Introduction
Monkeypox (mpox) is an infectious disease caused by monkeypox virus. Normally, most of the mbox cases were reported from West and Central Africa where mpox is endemic. However, there have been a number of reports of monkeypox cases where most of the confirmed cases have been reported to travel from Europe and North America which were not endemic countries. This situation has never happened before where many moneybox cases have been reported concurrently in non-endemic countries. Mpox can be spread through close and direct skin-to-skin contact. Pregnant women and newborn need to be cautious as the disease can infect the fetus during pregnancy and spread to the newborn. The World Health Organization (WHO) has alerted non-endemic countries to be aware and surveillance to prevent further outbreak. In Thailand, mpox has been found in populated provinces such as Bangkok, Nonthaburi and Chon buri. Given the problem statement, early detection of mpox is crucial to prevent further infection. To alleviate the risk of contact with high risk patients and identify the patients with mpox, we can apply image processing to solve this task.

## Models Used
1. Baseline CNN (AlexNet architecture)
2. Residual Networks (ResNet50, ResNet101, ResNet152) with BottleNeck layers
3. Residual Networks (ResNet50, ResNet101, ResNet152) with BottleNeck layers and Squeeze and Excitation module

*Note: No pre-trained models were utilized in this repository. We developed the models from scratch in order to customize the model (particularly BottleNeck layers to capture fine, intricate details of lesion and Squeeze and Excitation Blocks for image recalibration).*

<center>
<img src = "/images/block_diagram.PNG" width = "1200"/>
</center>

## Baseline Results 
We applied Gradient-weighted Class Activation Mapping (GradCAM) to gain a deeper understanding of BaselineCNN's performance. Initial results shed light on the model's limitations; CHP and HEALTHY exhibit lower F1-scores, primarily because BaselineCNN struggles to capture the intricate details specific to these classes. Instead, the model focuses on non-unique aspects, resulting in misclassifications. BaselineCNN may consider these intricate details as noise and may struggle with CHP and HEALTHY class with adversarial attacks, compression articfacts, and other forms of distortion. Existing studies have concluded that while AlexNet (as BaselineCNN's foundational architecture) can detect noisem it may not be robust to certain types of noise - especially in dealing with real-world variations. In contrast, HFMD, MKP, and MSL fare well in classification, as BaselineCNN effectively identifies the pertinent features. As a result, there is a compelling need to explore and propose a more robust model for the accurate classification of these skin lesions.
<center>
<img src = "/images/baseline_result.PNG" width = "1200"/>
</center>

## Comparison to other models
<center>
<img src = "/images/chicken_mokey.PNG" width = "1200"/>
</center>

## Summary of Results
The table presents a comprehensive overview of different models employed for lesion classification, along with their key performance metrics. Two categories of models are featured: the Baseline CNN models and the ResNet models. These models vary in their architecture and include enhancements such as bottleneck layers and Squeeze-and-Excitation (SE) blocks. 

The presence of the SE block across ResNet models significantly boosts their feature recalibration, resulting in remarkably high F1-scores and accuracy, highlighting the efficacy of feature recalibration in enhancing classification performance. This recalibration improved the performance by distinguishing the small, intricate details of the lesions. Moreover, the SE blocks also serve as a color space recalibrator since the prevalence of skin-tone bias in MSLDv2.0 dataset is evident, as most of the images within classes such as Chickenpox, Cowpox, Measles, and HFMD primarily depict individuals with lighter skin tones.

<center>
<img src = "/images/summary.PNG" width = "808"/>
</center>