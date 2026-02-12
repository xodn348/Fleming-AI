# Expected Results

## Experiment Overview

This document outlines the expected results for our transformer pre-training experiments, based on the hypothesis that self-supervised pre-training followed by supervised fine-tuning will outperform training from scratch.

---

## Baseline Expectations

### 1. Training from Scratch (Supervised Only)

Based on existing literature for Vision Transformers trained on smaller datasets:

**CIFAR-10:**
- **ViT-Tiny/Small from scratch:** 70-75% accuracy (without extensive augmentation)
- **ViT-Tiny/Small with augmentation:** 75-82% accuracy
- **Challenge:** ViTs typically underperform CNNs when trained on small datasets due to lack of inductive bias

**CIFAR-100:**
- **ViT-Tiny/Small from scratch:** 45-55% accuracy (without extensive augmentation)
- **ViT-Tiny/Small with augmentation:** 55-65% accuracy

**Expected Issues:**
- Overfitting due to small dataset size (50k training images)
- Poor generalization without pre-training
- High variance across training runs

---

### 2. Self-Supervised Pre-training + Fine-tuning

Based on MoCo v3, DeiT, and other self-supervised ViT papers:

**Pre-training Phase (Self-Supervised on CIFAR-10 unlabeled data):**
- **Objective:** Learn visual representations without labels
- **Expected linear probing accuracy:** 60-70% on CIFAR-10
- **Training time:** 200-400 epochs for convergence

**Fine-tuning Phase (Supervised on CIFAR-10 labeled data):**

**CIFAR-10:**
- **ViT-Tiny with MoCo v3 pre-training:** 82-86% accuracy
- **ViT-Small with MoCo v3 pre-training:** 85-88% accuracy
- **Expected improvement over from-scratch:** +10-15 percentage points

**CIFAR-100:**
- **ViT-Tiny with pre-training:** 65-72% accuracy
- **ViT-Small with pre-training:** 70-76% accuracy
- **Expected improvement over from-scratch:** +12-18 percentage points

---

## Detailed Performance Expectations

### Linear Probing Results (Self-Supervised Pre-training Quality)

Linear probing (freezing backbone, training only linear classifier) indicates quality of learned representations:

| Model | CIFAR-10 Linear Probe | CIFAR-100 Linear Probe |
|-------|----------------------|------------------------|
| ViT-Tiny (no pre-train) | ~50% | ~30% |
| ViT-Tiny + MoCo v3 | **62-68%** | **45-52%** |
| ViT-Small + MoCo v3 | **65-72%** | **48-56%** |

**Note:** Linear probing results are typically 10-20% lower than full fine-tuning results.

---

### Full Fine-tuning Results

After supervised fine-tuning on labeled CIFAR-10/100:

| Model | Training Method | CIFAR-10 Accuracy | CIFAR-100 Accuracy |
|-------|----------------|-------------------|-------------------|
| **ViT-Tiny** | From Scratch | 75-80% | 55-62% |
| **ViT-Tiny** | Self-supervised + Fine-tune | **83-87%** | **68-74%** |
| **ViT-Small** | From Scratch | 78-83% | 58-66% |
| **ViT-Small** | Self-supervised + Fine-tune | **86-90%** | **72-78%** |

**Expected Gain:**
- **CIFAR-10:** +6-10 percentage points
- **CIFAR-100:** +10-14 percentage points

---

## Comparison with State-of-the-Art

### CIFAR-10 State-of-the-Art (for reference)

- **ConvNeXt:** ~97.3%
- **Swin Transformer:** ~96.5%
- **ResNet-110 with augmentation:** ~95.6%
- **DeiT-Small (pre-trained on ImageNet):** ~98.5%
- **ViT-Base (pre-trained on ImageNet-21k):** ~99.0%

**Our Expected Range:** 83-90% (realistic for small-scale self-supervised pre-training on CIFAR)

### CIFAR-100 State-of-the-Art (for reference)

- **ConvNeXt:** ~86.5%
- **Swin Transformer:** ~84.2%
- **ResNet-110 with augmentation:** ~77.8%
- **DeiT-Small (pre-trained on ImageNet):** ~90.5%
- **ViT-Base (pre-trained on ImageNet-21k):** ~92.3%

**Our Expected Range:** 68-78% (realistic for small-scale self-supervised pre-training on CIFAR)

---

## Training Efficiency Expectations

### Convergence Speed

**From Scratch (Supervised):**
- **Epochs to convergence:** 200-400 epochs
- **Training time:** ~2-4 hours (single GPU)
- **Stability:** Moderate, sensitive to hyperparameters

**Self-Supervised Pre-training + Fine-tuning:**
- **Pre-training epochs:** 200-400 epochs
- **Fine-tuning epochs:** 50-100 epochs (faster convergence)
- **Total training time:** ~3-5 hours (pre-train) + ~0.5-1 hour (fine-tune)
- **Stability:** Higher stability during fine-tuning, more robust to hyperparameters

**Expected Benefit:**
- Fine-tuning converges **2-4x faster** than training from scratch
- Better generalization with less labeled data

---

### Data Efficiency

Based on DeiT and MoCo v3 findings, we expect:

**Label Efficiency:**
- With 100% labels (50k images): +6-10% improvement
- With 10% labels (5k images): +15-20% improvement
- With 1% labels (500 images): +20-30% improvement

**Key Finding:**
- Self-supervised pre-training is most beneficial when labeled data is scarce
- The relative improvement increases as labeled data decreases

---

## Ablation Study Expectations

### Effect of Pre-training Epochs

| Pre-training Epochs | Expected Linear Probe Accuracy | Expected Fine-tuning Accuracy |
|--------------------|---------------------------------|-------------------------------|
| 0 (no pre-train) | ~50% | 75-80% |
| 100 | 55-60% | 78-82% |
| 200 | 60-65% | 82-85% |
| 400 | 62-68% | 83-87% |
| 800 | 64-70% | 84-88% |

**Expected Observation:**
- Diminishing returns after 400 epochs
- Optimal pre-training duration: 300-500 epochs

---

### Effect of Batch Size

| Batch Size | Expected Linear Probe Accuracy | Training Stability |
|-----------|--------------------------------|-------------------|
| 64 | 58-62% | Moderate |
| 128 | 60-66% | Good |
| 256 | 62-68% | **Best** |
| 512 | 60-66% | Good (requires larger model) |

**Expected Finding:**
- Batch size 256 provides best trade-off
- Larger batches require stronger augmentation

---

### Effect of Learning Rate

| Learning Rate | Expected Performance | Risk |
|--------------|---------------------|------|
| 1e-5 | Underfitting | Too conservative |
| 1e-4 | Good baseline | Safe choice |
| 5e-4 | **Best performance** | Optimal |
| 1e-3 | Training instability | Too aggressive |

**Expected Optimal:** 3e-4 to 5e-4 with cosine decay

---

## Hypothesis Validation Criteria

### Success Criteria

Our hypothesis will be considered validated if:

1. **Primary Criterion:**
   - Self-supervised + fine-tuning achieves **≥ 5% higher accuracy** than training from scratch on CIFAR-10
   - Self-supervised + fine-tuning achieves **≥ 8% higher accuracy** than training from scratch on CIFAR-100

2. **Secondary Criteria:**
   - Linear probing accuracy ≥ 60% on CIFAR-10 (indicates good pre-training)
   - Fine-tuning converges in ≤ 100 epochs (vs 200-400 for from-scratch)
   - Consistent improvement across multiple training runs (low variance)

3. **Strong Validation:**
   - CIFAR-10 accuracy ≥ 85% (ViT-Small with pre-training)
   - CIFAR-100 accuracy ≥ 70% (ViT-Small with pre-training)
   - At least 10% improvement in low-data regime (10% labels)

---

## Risk Factors and Mitigation

### Potential Issues

1. **Training Instability:**
   - **Expected:** Self-supervised ViT training can be unstable
   - **Mitigation:** Use batch normalization in projection head, random patch projection
   - **Reference:** MoCo v3 paper techniques

2. **Small Dataset Size:**
   - **Expected:** CIFAR-10 (32×32, 50k images) is small for ViT pre-training
   - **Mitigation:** Strong data augmentation, smaller patch size (4×4 or 8×8)
   - **Expected impact:** May limit absolute performance but should still show relative improvement

3. **Compute Limitations:**
   - **Expected:** Limited pre-training epochs compared to large-scale studies
   - **Mitigation:** Focus on data efficiency gains rather than absolute SOTA
   - **Expected outcome:** 83-87% instead of 90%+

---

## Statistical Expectations

### Confidence Intervals

Based on typical variance in ViT training:

**From Scratch (3 runs):**
- Mean: 77.5% ± 2.3%
- Range: 75.2% - 79.8%

**Pre-training + Fine-tuning (3 runs):**
- Mean: 85.2% ± 1.5%
- Range: 83.7% - 86.7%

**Expected Observation:**
- Pre-training reduces variance (more stable)
- Non-overlapping confidence intervals confirm statistical significance

---

## Timeline Expectations

### Experiment Execution

1. **Pre-training Phase:** 8-12 hours
2. **Linear Probing Evaluation:** 30 minutes
3. **Fine-tuning Phase:** 1-2 hours
4. **Evaluation and Analysis:** 1 hour

**Total Experiment Time:** ~12-16 hours per configuration

**Multiple Runs for Statistical Significance:** 3-5 runs × 12 hours = 1.5-3 days

---

## Summary of Key Expected Results

| Metric | Expected Value | Significance |
|--------|---------------|--------------|
| **CIFAR-10 Improvement** | +8-12% | Primary validation |
| **CIFAR-100 Improvement** | +12-16% | Primary validation |
| **Linear Probe Accuracy** | 62-68% | Pre-training quality |
| **Fine-tuning Speedup** | 2-4x faster convergence | Training efficiency |
| **Data Efficiency (10% labels)** | +15-20% improvement | Practical value |
| **Training Stability** | Lower variance | Robustness |

---

## References to Benchmark Results

- **ViT Paper:** 87.76% on ImageNet (ViT-L/16 pre-trained on ImageNet-21k)
- **MoCo v3:** 76.7% linear probe on ImageNet (ViT-B)
- **DeiT:** 83.1% on ImageNet (trained only on ImageNet, no extra data)
- **Expected CIFAR-10 transfer:** Linear scaling suggests ~80-85% with small-scale pre-training

**Adjustment for Dataset Scale:**
- ImageNet: 1.28M training images, 1000 classes
- CIFAR-10: 50k training images, 10 classes
- **Expected performance:** 15-20% lower than ImageNet-scale results due to limited pre-training data
