# Two-Stage Multimodal Timed Up and Go (TUG) Assessment Framework

**Repository:** [TUGT-CONTEXTAWARE-CNN-BiLSTM](https://github.com/yudhiardiyanto/TUGT-CONTEXTAWARE-CNN-BiLSTM)

## Overview

This repository accompanies research on a **two-stage multimodal framework** for automated Timed Up and Go (TUG) assessment. The framework first identifies five TUG movement phases from skeletal motion sequences using a convolutional neural network–bidirectional long short-term memory (CNN-BiLSTM) model. It then combines learned skeletal representations with environmental and individual-level measurements to classify **TUG-derived risk categories** using conventional machine-learning classifiers.

**Important:** The HIGH and LOW/NO categories in this study are derived from a **13.5-second TUG completion-time threshold**, not from prospectively observed falls. Accordingly, the reported performance is for TUG-derived categorization and does **not** establish prospective fall prediction or clinical diagnostic validity.

## Framework Architecture

```mermaid
flowchart TD
    subgraph ACQ["1. DATA ACQUISITION"]
        direction LR
        A1["A. RGB TUG Videos<br/>RGB video: 30 FPS<br/>Skeleton data: 32 joints / 33 landmarks<br/>Sit–Stand–Walk–Turn–Return"]
        A2["B. Environmental Sensor Data<br/>Temperature · Humidity · Light intensity<br/>Pressure · GPS location"]
        A3["C. Physiological, Demographic & Anthropometric Data<br/>Heart rate · SpO₂ · Body temperature<br/>Age · Sex · BMI"]
    end

    A1 --> EX["2. SKELETON EXTRACTION<br/>MediaPipe Pose: 33 landmarks<br/>(x, y, z coordinates)"]
    A2 --> CTX["3. CONTEXT DATA ACQUISITION<br/>Time-aligned environmental data<br/>(synchronized with video)"]
    A3 --> IND["Individual-level measurements"]

    subgraph PRE["4. DATA PREPROCESSING"]
        direction LR
        P1["A. Skeleton Preprocessing<br/>Joint selection and normalization<br/>Missing-value handling<br/>Temporal smoothing<br/>30-frame sequence windows (~1 s)<br/>Min–max feature scaling"]
        P2["B. Context Data Processing<br/>Missing-value handling<br/>Interpolation<br/>Feature scaling (standardization)"]
    end
    EX --> P1
    CTX --> P2

    P1 --> C["STAGE 1: CNN-BiLSTM — Three Model Configurations"]
    P2 --> C
    C --> C1["Skeleton Only"]
    C --> C2["Skeleton + Context: Late Fusion"]
    C --> C3["Skeleton + Context: Attention Fusion"]
    C1 --> D["Five TUG movement phases"]
    C2 --> D
    C3 --> D
    D --> E["64-dimensional skeletal embeddings"]

    E --> H["STAGE 2: Feature Configurations"]
    P2 --> H
    IND --> H
    H --> I["LR / SVM-RBF / RF / GB / MLP"]
    I --> J["LOSO-CV: TUG-derived HIGH vs LOW/NO"]
    J --> K["Performance evaluation"]
    I --> L["SHAP: group-level, global, and individual explanations"]
```

### Stage 1 — TUG phase classification

- Input: skeletal motion sequences from TUG recordings.
- Model: CNN-BiLSTM evaluated in three configurations: **Skeleton Only**, **Skeleton + Context (Late Fusion)**, and **Skeleton + Context (Attention Fusion)**.
- Phase labels: **Sit-to-Stand (STS), Walk Out (WO), Turn (TR), Walk In (WI), and Stand-to-Sit (STTS)**.
- Intermediate output used downstream: **64-dimensional learned skeletal embeddings**. The released code should specify which Stage 1 configuration generated these embeddings.
- Intended data split: **70% training, 20% validation, and 10% testing**, using `GroupShuffleSplit` with participant identifiers as groups, **provided this matches the released Stage 1 implementation**. Group splitting does not itself guarantee stratification by phase.

### Stage 2 — TUG-derived risk classification

Five classifiers are compared: Logistic Regression (LR), radial-basis-function Support Vector Machine (SVM-RBF), Random Forest (RF), Gradient Boosting (GB), and Multi-Layer Perceptron (MLP). Evaluation uses **Leave-One-Subject-Out Cross-Validation (LOSO-CV)** across 30 participants.

Four feature configurations are evaluated:

| Configuration | Inputs | Number of features |
|---|---|---:|
| Skeleton | Learned skeletal embeddings | 64 |
| Skeleton + Context | Skeletal embeddings + environmental context | 68 |
| Skeleton + Physiological | Skeletal embeddings + Physiological, Demographic & Anthropometric Data variables | 71 |
| All Features | Skeletal embeddings + context + Physiological, Demographic & Anthropometric Data variables | 75 |

The name **Physiological** is retained for consistency with the experiment, although this group includes physiological, anthropometric, and demographic measurements.

## Dataset

The available Stage 2 dataset is a participant-level CSV with **30 rows and 83 columns** (one row per participant). It contains 20 `HIGH` and 10 `LOW/NO` labels.

| Field(s) | Description | Stage 2 predictor? |
|---|---|---|
| `subject` | Participant identifier | No; used for grouping/evaluation |
| `duration_STS`, `duration_WO`, `duration_TR`, `duration_WI`, `duration_STTS` | TUG phase durations | **No** |
| `total_tug_time` | Overall TUG completion time; basis for the study's risk-category definition | **No** |
| `temperature`, `humidity`, `light_intensity`, `pressure` | Environmental context | Yes |
| `age`, `height_cm`, `weight_kg`, `bmi`, `body_temperature`, `mean_spo2`, `mean_heart_rate` | Physiological, Demographic & Anthropometric Data | Yes |
| `skel_emb_000` ... `skel_emb_063` | Learned skeletal embeddings | Yes |
| `fall_risk` | TUG-derived target label (`HIGH` or `LOW/NO`) | Target only |

**Leakage prevention:** Phase durations and `total_tug_time` must not be used as Stage 2 predictors because the target category is defined from TUG completion time. Any preprocessing fitted from data (such as imputation or scaling) should be fitted **within each training fold only**.

### Dataset placement

After verifying that redistribution is permitted and all identifying information has been removed, place the shareable dataset at:

```text
dataset/multimodal_fall_risk_dataset.csv
```

If the raw recordings or other supporting datasets cannot be released, describe their access restrictions and the corresponding-author request procedure in `dataset/README.md`. **Do not upload identifiable videos, personal identifiers, or unconsented health data.**

## Repository Layout (Recommended)

The structure below is a **proposed organization**, not a claim that every file is already present in this repository:

```text
TUGT-CONTEXTAWARE-CNN-BiLSTM/
├── README.md
├── requirements.txt
├── LICENSE
├── dataset/
│   ├── README.md
│   ├── data_dictionary.csv
│   └── multimodal_fall_risk_dataset.csv
├── preprocessing/
│   └── ...
├── stage1_tug_classification/
│   └── ...
├── stage2_risk_classification/
│   └── ...
├── explainability/
│   └── ...
└── results/
    ├── stage1/
    ├── stage2/
    └── shap/
```

## Software Requirements

- Python 3.10 or later (**suggested; confirm against the actual training environment**).
- Core data and machine-learning packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
- Stage 1 deep-learning framework: install the **same TensorFlow/Keras or PyTorch versions used by the released CNN-BiLSTM implementation**.
- Explainability: `shap` and its compatible dependencies.
- Optional: Jupyter Notebook / Google Colab for interactive experiments.

**Reproducibility note:** The exact Python and library versions must be taken from the original working environment. A pinned `requirements.txt` or environment lock file should be committed before claiming that the repository is fully reproducible.

## Installation

```bash
git clone https://github.com/yudhiardiyanto/TUGT-CONTEXTAWARE-CNN-BiLSTM.git
cd TUGT-CONTEXTAWARE-CNN-BiLSTM
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Once a validated dependency file has been committed:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not run the last command until `requirements.txt` exists in the repository.

## Reproducing the Experiments

### 1. Prepare the data

1. Obtain the shareable participant-level CSV and any permitted Stage 1 skeletal sequence files.
2. Verify that `subject` uniquely identifies each participant in the Stage 2 table.
3. Check column names and the target labels (`HIGH`, `LOW/NO`).
4. Exclude phase-duration and total-duration columns from Stage 2 feature matrices.

### 2. Run Stage 1

1. Preprocess skeletal sequences and construct labeled temporal windows.
2. Split participants into training, validation, and testing groups according to the **actual released implementation**; the intended target ratio is 70:20:10.
3. Train and evaluate the CNN-BiLSTM phase classifier.
4. Export 64-dimensional skeletal embeddings for downstream analysis.

**Command to add:** The Stage 1 training entry point and its arguments must be documented after the corresponding code is uploaded. Do not assume a script named `train.py` exists.

### 3. Run Stage 2

1. Build the four specified feature configurations.
2. For each LOSO fold, hold out one participant for testing and train on the other 29.
3. Fit any imputers/scalers on the training fold only.
4. Fit the five classifiers and collect out-of-fold predictions.
5. Calculate accuracy, balanced accuracy, macro-F1, ROC-AUC, and confusion matrices.

**Command to add:** Provide the exact runnable script path, command-line arguments, random seeds, and expected output files when the final Stage 2 code is committed. Existing local research scripts may contain Google Drive-specific absolute paths that must be made configurable before public use.

### 4. Generate SHAP explanations

Explain the SVM-RBF classifier with the All-Features configuration, using the appropriate fold-trained models. Produce:

- Feature-group mean absolute SHAP importance, normalized by group feature count.
- Global feature importance and SHAP summary plots.
- Individual explanations for representative true-positive, true-negative, false-positive, and false-negative predictions.

Interpret SHAP as **model-specific attribution**, not causal evidence or proof of incremental predictive value.

**Command to add:** Publish the validated SHAP script and instructions for reproducing the paper's figures.

## Main Reported Results

The following are the reported **Stage 2** results from the study, not independently reproduced by running this repository:

| Evaluation setting | Result |
|---|---:|
| Participants | 30 |
| HIGH / LOW/NO labels | 20 / 10 |
| Cross-validation | 30-fold LOSO-CV |
| Best observed classifier | SVM-RBF |
| Best observed feature configurations | Skeleton + Physiological; All Features |
| ROC-AUC | **0.640** |
| Accuracy | **0.667** |
| Balanced accuracy | **0.550** |

Adding environmental context to Skeleton + Physiological did **not** improve the observed SVM-RBF ROC-AUC. In the SHAP analysis, environmental variables had the largest **mean absolute attribution per feature** after group-size normalization; this should not be interpreted as an improvement in predictive accuracy. Given the limited sample size and modest balanced accuracy, findings are exploratory.

## Reproducibility Checklist

- [ ] Publish the final Stage 1 implementation and its input format.
- [ ] Publish the final Stage 2 evaluation script with configurable paths.
- [ ] Publish the validated SHAP analysis script.
- [ ] Upload an anonymized, redistribution-approved dataset or document access restrictions.
- [ ] Add a dataset dictionary with measurement units and missing-value conventions.
- [ ] Pin dependency versions in `requirements.txt`.
- [ ] Record random seeds, splitting logic, model hyperparameters, and environment details.
- [ ] Verify that clean-environment execution reproduces the manuscript tables and figures.
- [ ] Add an appropriate code license and clarify dataset reuse terms.

## Data Availability

The source code is intended to be made available through this repository. Supporting datasets should be released only where participant consent and applicable data-use permissions allow; otherwise, access arrangements should be specified by the corresponding author. Update this section when the public contents have been verified.

## Citation

If this repository supports a published manuscript, please cite the corresponding paper. **Add the verified paper title, authors, journal, year, and DOI after publication.**

## Disclaimer

This is a research prototype for TUG-derived assessment. It is **not** a clinically validated fall prediction or diagnostic tool and should not be used as the sole basis for clinical decisions.
