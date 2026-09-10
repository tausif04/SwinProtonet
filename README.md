# SwinProtoNet-GI-Endoscopy

## Few-Shot Gastrointestinal Endoscopy Classification with Self-Supervised Swin-Tiny

A research implementation for **few-shot gastrointestinal (GI) endoscopic image classification**
using DINO-style self-supervised pretraining, Swin-Tiny, prototype-based meta-learning,
cross-dataset validation, baseline comparison, ablation analysis, and uncertainty-aware
decision support.

---

## Overview

```text
Kvasir-Capsule
      │
      ▼
DINO-style SSL Pretraining
      │
      ▼
Swin-Tiny Encoder
      │
      ▼
SwinProtoNet Few-Shot Training
      │
      ├── In-Domain Evaluation
      ├── Cross-Dataset Validation
      ├── Ablation Study
      ├── Baseline Comparison
      └── Uncertainty-Aware Decision Support
```

The project consists of seven sequential experiment scripts.

---

## Repository Structure

```text
SwinProtoNet-GI-Endoscopy/
│
├── README.md
├── src/
│   ├── part1_ssl_pretraining.py
│   ├── part2_fewshot_training.py
│   ├── part3a_indomain_test.py
│   ├── part3b_crossdataset_validation.py
│   ├── part3c_ablation_full.py
│   ├── part3d_baseline_model_comparison.py
│   └── part3e_uncertaintyaware_decision_support.py
│
├── configs/
├── data/
├── docs/
├── requirements.txt
└── .gitignore
```

The `results/` directory is optional and does not need to be uploaded if you do not want to
share experiment outputs.

---
# Experimental Pipeline

## 1. Self-Supervised Pretraining

**`part1_ssl_pretraining.py`**

DINO-style student-teacher pretraining is used to learn a domain-specific visual
representation from Kvasir-Capsule.

Main components:

- Swin-Tiny
- `swin_tiny_patch4_window7_224`
- 224 × 224 input resolution
- 768-dimensional representation
- EMA teacher
- two augmented views
- endoscopy-specific augmentation
- validation using cosine similarity

---

## 2. Few-Shot Meta-Training

**`part2_fewshot_training.py`**

The proposed **SwinProtoNet** is trained using episodic few-shot learning on Kvasir-v2.

The pipeline includes:

- 8-way episodic learning
- prototype-based classification
- Swin-Tiny representation
- projection embedding
- temperature-scaled prototype classification
- label smoothing
- fixed train/validation/test split
- domain-specific augmentation

---

## 3. In-Domain Evaluation

**`part3a_indomain_test.py`**

Evaluates the trained model on the held-out Kvasir-v2 test set using multiple N-way and
K-shot configurations.

Metrics include:

- accuracy
- macro-F1
- balanced accuracy
- confidence intervals
- per-class precision
- recall/sensitivity
- specificity
- F1
- AUC where configured

---

## 4. Cross-Dataset Validation

**`part3b_crossdataset_validation.py`**

Tests generalization using:

- held-out Kvasir-Capsule classes
- an independent capsule-endoscopy dataset

The evaluation includes few-shot performance and comparative analysis.

---

## 5. Ablation Study

**`part3c_ablation_full.py`**

The ablation evaluates the contribution of self-supervised representation learning by comparing
different initialization/training configurations.

---

## 6. Baseline Model Comparison

**`part3d_baseline_model_comparison.py`**

The controlled comparison includes:

- MatchingNet
- RelationNet
- Baseline
- Baseline++
- MAML
- Transformer / FEAT-style adaptation
- ResNet50

The comparison uses episodic evaluation and reports accuracy, macro-F1, balanced accuracy,
confidence intervals, and seed variation.

---

## 7. Uncertainty-Aware Decision Support

**`part3e_uncertaintyaware_decision_support.py`**

The final stage combines classification, uncertainty estimation, calibration, interpretability,
and decision support.

### Uncertainty

- MC Dropout
- predictive variance
- predictive entropy
- confidence
- prediction margin

### Calibration

- Expected Calibration Error (ECE)
- Brier score
- validation-based confidence calibration

### Interpretability

- Swin-aware Grad-CAM
- prediction-focused visual explanations

### Decision Support

Predictions are assigned to three review levels according to confidence/uncertainty.

---

# Datasets

The project uses:

- **Kvasir-v2** — primary labeled few-shot classification dataset.
- **Kvasir-Capsule** — self-supervised pretraining and held-out-class evaluation.
- **SEE-AI / KYU Capsule** — external evaluation dataset where configured.
Datasets are **not included** in this repository.

---

# Installation

The experiments were developed for a CUDA-enabled environment such as Google Colab.

```bash
git clone <YOUR_REPOSITORY_URL>
cd SwinProtoNet-GI-Endoscopy
pip install -r requirements.txt
```

---

# Reproducibility

For a reproducible research run, record:

```text
Source script
Configuration
Random seed(s)
Dataset split
Checkpoint
N-way / K-shot
Number of episodes
Execution timestamp / run ID
```

Experimental results should always be generated directly by the corresponding code rather
than manually estimated or reconstructed.

---

# Research Scope

This project investigates the combination of:

```text
Self-Supervised Learning
        +
Swin Transformer
        +
Few-Shot Meta-Learning
        +
Prototype-Based Classification
        +
Cross-Dataset Validation
        +
Uncertainty Estimation
        +
Explainable AI
        +
Decision Support
```

The decision-support component is a research framework and is **not a clinically validated
diagnostic system**.

---

# Important Notes

- Raw datasets are not included.
- Model checkpoints do not need to be publicly shared.
- Detailed result CSVs and raw evaluation logs are not required in the public repository.
- The reported values in the README are summary results from the completed experiments.
- The Kvasir-Capsule held-out-class evaluation is novel-class generalization within the same
  image collection.
- Any compute-budget reduction should be documented in the relevant experiment source.
- If code or configuration changes, previously reported results should be treated as stale until
  reproduced under the updated setup.

---


### Suggested Topics

```text
few-shot-learning
meta-learning
medical-image-classification
endoscopy
gastrointestinal
deep-learning
swin-transformer
self-supervised-learning
dino
prototypical-networks
uncertainty-estimation
explainable-ai
medical-ai
computer-vision
pytorch
```

---

# Citation

```bibtex
@misc{swinprotonet_gi_endoscopy,
  title  = {SwinProtoNet-GI-Endoscopy},
  author = {Tausif},
  year   = {2026},
  note   = {Few-shot gastrointestinal endoscopy classification research project}
}
```

---

## License

Add an appropriate open-source license before making the repository public.
