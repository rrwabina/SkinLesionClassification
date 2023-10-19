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

## Baseline Results 
<center>
<img src = "/images/baseline_result.PNG" width = "1200"/>
</center>

## Summary of Results
The table presents a comprehensive overview of different models employed for lesion classification, along with their key performance metrics. Two categories of models are featured: the Baseline CNN models and the ResNet models. These models vary in their architecture and include enhancements such as bottleneck layers and Squeeze-and-Excitation (SE) blocks. 

The presence of the SE block across ResNet models significantly boosts their feature recalibration, resulting in remarkably high F1-scores and accuracy, highlighting the efficacy of feature recalibration in enhancing classification performance. This recalibration improved the performance by distinguishing the small, intricate details of the lesions. Moreover, the SE blocks also serve as a color space recalibrator since the prevalence of skin-tone bias in MSLDv2.0 dataset is evident, as most of the images within classes such as Chickenpox, Cowpox, Measles, and HFMD primarily depict individuals with lighter skin tones.

<center>
<img src = "/images/summary.PNG" width = "808"/>
</center>