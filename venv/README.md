# 🚀 MobileNet-Based CIFAR-10 Image Classification with Custom Hybrid Loss Function

[![Framework - TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![Architecture - MobileNetV1](https://img.shields.io/badge/Architecture-MobileNetV1-blue?style=flat)](https://arxiv.org/abs/1704.04861)
[![Dataset - CIFAR--10](https://img.shields.io/badge/Dataset-CIFAR--10-green?style=flat)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Course - Computational%20Science](https://img.shields.io/badge/Course-Computational%20Science-red?style=flat)](#)

## 🌟 Introduction & Project Motivation

Image classification is a fundamental pillar of modern Computer Vision and Deep Learning. While standard academic tasks often focus exclusively on training generic networks to maximize top-1 validation accuracy, this project shifts focus toward a more rigorous engineering challenge: optimizing a lightweight architecture using customized learning dynamics.

Developed as a core group project for the **Introduction to Computational Science** curriculum, this repository implements an end-to-end vision pipeline. We adapted a pre-trained **MobileNetV1** architecture via transfer learning to categorize the **CIFAR-10** dataset. To overcome standard cross-entropy limitations—such as overconfident misclassifications and training stagnation on borderline samples—we designed and evaluated a **Custom Hybrid Loss Function** that fuses **Focal Loss** with an **Entropy-based Confidence Penalty**.

### Key Learning & Engineering Objectives
* **Transfer Learning Dynamics:** Reusing spatial hierarchical features (edges, textures, shapes) pre-trained on ImageNet to learn specific patterns on small-footprint source images.
* **Loss Function Engineering:** Manipulating gradients directly through custom loss implementations to force the network to focus on high-uncertainty classes.
* **Regularization & Generalization:** Observing how penalizing low-entropy prediction distributions acts as a powerful regularizer to avoid overfitting.
* **Rigorous Evaluation:** Looking beyond standard accuracy to diagnose network failure modes using class-wise Precision, Recall, F1-Score, and Confusion Matrices.

---

## 📋 Dataset Architecture & Preprocessing
# MobileNet-CIFAR-10
Team Members
T. Harshavardhan - CS25B1042
B. Vignesh - MC25B1014
Adelly Sai Teja - CS25B1003
The **CIFAR-10** dataset contains **60,000 real-world color images** balanced evenly across 10 distinct categories:

| Property | Value |
| :--- | :--- |
| **Training Set Size** | 50,000 images (5,000 per class) |
| **Testing Set Size** | 10,000 images (1,000 per class)[cite: 2, 4] |
| **Spatial Dimensions** | $32 \times 32 \times 3$ (RGB)[cite: 2, 4] |
| **Target Classes** | Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck |

### Preprocessing Pipeline (`mobilenet_cifar10.py`)
Because the baseline images are natively small ($32 \times 32$), feeding them directly into deep networks built for higher resolutions often causes aggressive spatial degradation across early pooling layers. Our pipeline applies a strict series of transformations:
1. **Normalization:** Image pixel values are converted from integers in $[0, 255]$ to floats in $[0.0, 1.0]$ to stabilize weight initializations and gradient updates[cite: 1, 2].
2. **Label Encoding:** Category targets are mapped to categorical one-hot vectors ($Y \in \mathbb{R}^{10}$).
3. **Resolution Bilinear Upscaling:** Images are dynamically scaled using `tf.image.resize` to **$96 \times 96 \times 3$**. This satisfies the minimal dimension requirements of MobileNet while retaining the features necessary for successful transfer learning.

---

## 🏗️ Model Architecture & Transfer Learning Strategy

Instead of building a heavy Convolutional Neural Network from scratch, we exploit **MobileNetV1**. MobileNet relies heavily on **Depthwise Separable Convolutions**, which separate filtering and combination steps to dramatically drop parameter density and computational complexity while maintaining high top-k accuracy.

### Functional Topology

```text
       Input Matrix [96 × 96 × 3]
                   │
                   ▼
         MobileNetV1 Base Core
     (Pre-trained ImageNet Weights)
     [Frozen Layers: trainable=False]
                   │
                   ▼
       Global Average Pooling 2D
    (Collapses spatial 2D feature maps)
                   │
                   ▼
       Fully-Connected Dense Head
     [128 Units | ReLU Activation]
                   │
                   ▼
          Softmax Output Layer
       [10 Units | Class Probabilities]

# 🧠 The Custom Hybrid Loss Function (In-Depth)

The mathematical core of this project is the custom `HybridLoss` class.

Standard neural networks typically optimize using **Categorical Cross-Entropy Loss**, defined for a single training instance as:

$$
L_{CE} = -\sum_{c=1}^{C} y_c \log(p_c)
$$

where:

- $C$ = number of classes
- $y_c$ = ground-truth indicator for class $c$
- $p_c$ = predicted probability for class $c$

Although Cross-Entropy performs well in many classification problems, it has two important limitations:

1. It treats easy and difficult samples equally.
2. It often encourages extremely confident predictions, which can lead to overfitting and poor generalization.

To address these limitations, we designed a **Custom Hybrid Loss Function** that combines:

- **Focal Loss**
- **Entropy-Based Confidence Penalty**

The final objective function is:

$$
\mathcal{L}_{Hybrid}
=
\mathcal{L}_{Focal}
+
\lambda \cdot \mathcal{L}_{Confidence}
$$

where:

- $\lambda$ controls the strength of the confidence penalty.
- In our implementation:

$$
\lambda = 0.01
$$

---

# 1. Focal Loss Component

Focal Loss was introduced to address the problem of class imbalance and varying sample difficulty.

Instead of allowing easily classified examples to dominate the learning process, Focal Loss reduces their influence and forces the model to focus on difficult or ambiguous samples.

Let:

$$
p_t = \sum_{c=1}^{C} y_c p_c
$$

represent the predicted probability assigned to the correct class.

The Focal Loss is defined as:

$$
FL(p_t)
=
-\alpha (1-p_t)^\gamma \log(p_t)
$$

where:

- $\alpha$ = balancing coefficient
- $\gamma$ = focusing parameter
- $p_t$ = confidence assigned to the correct class

---

## Modulating Factor

The key innovation is the term:

$$
(1-p_t)^\gamma
$$

### Easy Samples

When the model predicts correctly with high confidence:

$$
p_t \rightarrow 1
$$

then:

$$
(1-p_t)^\gamma \rightarrow 0
$$

which greatly reduces the contribution of that sample to the total loss.

---

### Difficult Samples

When the model predicts incorrectly:

$$
p_t \rightarrow 0
$$

then:

$$
(1-p_t)^\gamma \rightarrow 1
$$

and the loss remains large.

Thus, difficult samples receive more attention during optimization.

---

## Hyperparameters Used

In our implementation:

```python
alpha = 1.0
gamma = 2.0
```

### Effect of γ

- γ = 0 → equivalent to Cross-Entropy Loss
- γ > 0 → progressively focuses on hard samples
- Larger γ → stronger emphasis on difficult examples

---

## Benefits of Focal Loss

- Focuses learning on difficult examples.
- Reduces dominance of easy samples.
- Improves robustness.
- Helps the model learn more discriminative features.
- Often improves classification performance on challenging datasets.

---

# 2. Confidence Penalty Component

Deep neural networks often become excessively confident in their predictions.

For example, a network may output:

```text
Class A : 99.99%
Class B : 0.01%
```

even when the prediction is uncertain.

Such overconfident outputs can:

- Increase overfitting.
- Reduce generalization ability.
- Produce poorly calibrated probabilities.

To mitigate this behavior, we introduce an entropy-based confidence penalty.

---

## Shannon Entropy

The entropy of a probability distribution is:

$$
H(p)
=
-\sum_{c=1}^{C}
p_c \log(p_c)
$$

Entropy measures uncertainty.

### High Entropy

A uniform distribution:

```text
0.1, 0.1, 0.1, ...
```

has high entropy.

### Low Entropy

A highly peaked distribution:

```text
0.99, 0.001, 0.001, ...
```

has low entropy.

---

## Confidence Penalty

We define the confidence penalty as:

$$
\mathcal{L}_{Confidence}
=
-H(p)
$$

Substituting entropy:

$$
\mathcal{L}_{Confidence}
=
\sum_{c=1}^{C}
p_c \log(p_c)
$$

---

## Why It Works

Minimizing:

$$
-H(p)
$$

encourages larger entropy values.

This prevents the network from becoming excessively certain too early during training.

As a result:

- Probability distributions become smoother.
- Decision boundaries become more stable.
- Generalization improves.
- Overfitting is reduced.

---

## Benefits of Confidence Penalty

- Discourages pathological overconfidence.
- Improves probability calibration.
- Produces smoother class distributions.
- Encourages better generalization.
- Acts as a regularization mechanism.

---

# 3. Final Hybrid Loss Function

The two components are combined:

$$
\mathcal{L}_{Hybrid}
=
\mathcal{L}_{Focal}
+
\lambda \cdot \mathcal{L}_{Confidence}
$$

Substituting both equations:

$$
\mathcal{L}_{Hybrid}
=
-\alpha (1-p_t)^\gamma \log(p_t)
+
\lambda
\sum_{c=1}^{C}
p_c \log(p_c)
$$

where:

```python
alpha = 1.0
gamma = 2.0
lambda_penalty = 0.01
```

---

# Why This Hybrid Approach Was Chosen

Each component addresses a different weakness of standard Cross-Entropy:

| Problem | Solution |
|----------|----------|
| Easy samples dominate training | Focal Loss |
| Difficult samples receive insufficient attention | Focal Loss |
| Overconfident predictions | Confidence Penalty |
| Poor probability calibration | Confidence Penalty |
| Overfitting risk | Confidence Penalty |
| Generalization limitations | Combined Hybrid Loss |

---

# Practical Impact on Our Model

The Hybrid Loss Function enabled the MobileNet classifier to:

- Focus more on difficult CIFAR-10 samples.
- Reduce excessive prediction confidence.
- Produce smoother probability distributions.
- Improve generalization on unseen data.
- Learn more balanced class boundaries.

By integrating both Focal Loss and Entropy Regularization into a single objective function, the model benefits from improved optimization behavior while maintaining strong classification performance.

# 👥 Team Contributions & Work Division

The project was developed collaboratively by four team members, with responsibilities distributed across data preparation, custom loss function design, model development, training, evaluation, and documentation.

---

## 🛠️ T Harshavardhan– Dataset Preparation & Preprocessing

### Responsibilities

- Loaded the CIFAR-10 dataset using TensorFlow/Keras utilities.
- Performed image normalization by scaling pixel values to the range **[0, 1]**.
- Converted categorical labels into one-hot encoded vectors using `to_categorical()`.
- Resized CIFAR-10 images from **32 × 32** to **96 × 96** pixels to match MobileNet input requirements.
- Verified training and testing dataset distributions.
- Prepared the complete preprocessing pipeline used throughout model training.

### Contribution Summary

This module ensured that the dataset was correctly formatted and optimized for transfer learning with MobileNet.

---

## 🧠 Adelly Sai Teja– Custom Hybrid Loss Function Design

### Responsibilities

- Designed and implemented the project's **Custom Hybrid Loss Function**.
- Developed the Focal Loss component to improve learning on difficult samples.
- Implemented the Entropy-Based Confidence Penalty to reduce model overconfidence.
- Created the custom `HybridLoss` class by extending `tf.keras.losses.Loss`.
- Added numerical stability safeguards using `tf.clip_by_value()`.
- Tuned the loss hyperparameters:

```python
alpha = 1.0
gamma = 2.0
lambda_penalty = 0.01
```

- Evaluated loss behavior during training and validation.

### Contribution Summary

This component represents the primary innovation of the project and explores how customized loss functions can influence neural network learning dynamics.

---

## 🏗️ B Vignesh– Model Architecture & Training

### Responsibilities

- Implemented the MobileNet backbone using pretrained ImageNet weights.
- Applied Transfer Learning techniques by freezing pretrained convolutional layers.
- Designed the classification head using:
    - GlobalAveragePooling2D
    - Dense layers
    - ReLU activation
    - Softmax output layer
- Compiled the model using:
    - Adam Optimizer
    - Custom Hybrid Loss Function
- Configured training parameters such as:
    - Epochs
    - Batch size
    - Validation strategy
- Executed model training and monitored performance throughout all epochs.

### Contribution Summary

This module integrated the pretrained MobileNet architecture with the custom loss function and managed the complete training workflow.

---

## 📊 Praneeth– Evaluation, Visualization & Documentation

### Responsibilities

- Evaluated model performance using multiple metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
- Generated the Classification Report using Scikit-Learn.
- Constructed and visualized the Confusion Matrix.
- Generated training and validation accuracy graphs.
- Performed prediction analysis using model inference outputs.
- Saved the final trained model in:

```text
mobilenet_cifar10_hybrid.keras
```

- Organized project outputs and visualizations.
- Prepared repository documentation and project presentation materials.

### Contribution Summary

This module focused on model evaluation, result interpretation, visualization, and project documentation to ensure the findings were clearly communicated.
