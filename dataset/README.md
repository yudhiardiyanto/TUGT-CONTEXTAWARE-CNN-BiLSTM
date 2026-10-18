# Preprocessed Dataset for Multimodal TUG Assessment

## Overview

This directory contains two preprocessed datasets developed for research on automated Timed Up and Go (TUG) assessment using skeleton-based deep learning and multimodal data fusion.

The datasets support two experimental configurations of the CNN-BiLSTM framework:

1. **Skeleton + Context with Attention Fusion**
2. **Skeleton-Only**

Both datasets have undergone preprocessing and feature preparation for their respective experimental configurations. The original raw data are not included in this repository.

## Dataset Files

### 1. Multimodal Dataset — Skeleton + Context with Attention Fusion

**Filename:** `multimodal_fall_risk_dataset.csv`

[View Dataset](https://github.com/yudhiardiyanto/TUGT-CONTEXTAWARE-CNN-BiLSTM/blob/main/dataset/multimodal_fall_risk_dataset.csv)

This dataset was prepared using the **Skeleton + Context with Attention Fusion** model configuration.

The approach integrates skeleton-based movement representations with contextual information through an attention-based fusion mechanism. It is intended to support the investigation of contextual information in automated TUG assessment.

The dataset is provided in its processed form for subsequent analysis and evaluation.

### 2. Skeleton-Only Dataset

**Filename:** `skeleton_only_full_multimodal_83cols.csv`

[View Dataset](https://github.com/yudhiardiyanto/TUGT-CONTEXTAWARE-CNN-BiLSTM/blob/main/dataset/skeleton_only_full_multimodal_83cols.csv)

This dataset was prepared using the **Skeleton-Only** model configuration.

The skeleton-based approach relies on movement representations extracted from skeletal data without incorporating contextual information during the skeleton-only representation-learning stage.

The dataset is intended to support baseline evaluation and comparative analysis against the multimodal attention-fusion configuration.

Although the filename contains the term `full_multimodal_83cols`, it is associated with the Skeleton-Only experimental configuration.

## Data Processing

Both datasets represent processed research data rather than original sensor recordings or raw skeletal measurements.

The processing workflow involves preparing skeletal movement representations and organizing the resulting data for the corresponding experimental configurations.

These datasets are intended to facilitate reproducible research, comparative model evaluation, and further investigation of skeleton-based and context-aware TUG assessment methods.

## Raw Data Availability

**The original raw datasets are not publicly available in this repository.**

Researchers interested in obtaining the original raw data for academic research, methodological verification, or reproducibility studies may contact the **corresponding author**.

Raw data may be made available upon reasonable request, subject to applicable ethical considerations, participant privacy requirements, and data-sharing permissions.

## Research Use and Citation

These datasets are provided for academic and research purposes.

Researchers using these datasets are encouraged to acknowledge the original authors and cite the associated research publication when available.

## Contact

For further information regarding the datasets, preprocessing procedures, or requests for access to the original raw data, please contact the **corresponding author of the associated research study**.
