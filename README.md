# Blastocyst Image Classification Using Deep Learning

This repository contains the code developed for the **classification of human blastocyst images** based on their **inner cell mass (ICM) score**, using deep learning and computer vision techniques.

The work focuses on building a **robust and reproducible image classification pipeline** under real-world constraints, including limited data availability, variability in image acquisition, and strict privacy requirements.

---

## Problem Description

In assisted reproduction, the morphological assessment of blastocysts plays a critical role in embryo selection. This project addresses the automated classification of blastocyst images according to their ICM quality, aiming to support expert decision-making through AI-based image analysis.

---

## Methodology

The implemented pipeline consists of the following stages:

1. **Preprocessing**
   - Noise removal filtering to improve image quality
   - Alignment of embryo orientation to reduce variance due to tilt and acquisition angle

2. **Deep Learning Model**
   - Convolutional neural network based on **ResNet-18**
   - Supervised training using standard deep learning practices
   - Focus on generalization and robustness rather than model complexity

3. **Training & Evaluation**
   - Standard training-validation splits
   - Performance evaluation using classification metrics relevant to medical imaging tasks

---

## Implementation Details

- **Framework:** PyTorch
- **Model Architecture:** ResNet-18
- **Language:** Python
- **Data:** Private medical image dataset (not included)

Due to privacy and ethical constraints, the dataset used in this project cannot be publicly shared. The code is provided to document the methodology and enable reproducibility on similar datasets.

---

## Research Context

This research was carried out as part of the project:

**“Classification and characterization of fetal images for assisted reproduction using artificial intelligence and computer vision”**  
(Project code: **KP6-0079459**)

The project was funded under the Action *“Investment Plans of Innovation”* of the Operational Program *Central Macedonia 2014–2020*, co-funded by the **European Regional Development Fund (ERDF)** and **Greece**.

---

## Notes

- This repository focuses on **methodology and pipeline design**, not dataset distribution.
- The approach prioritizes robustness, interpretability, and practical deployment considerations.
- The code structure reflects research-oriented development under real-world constraints.

---

