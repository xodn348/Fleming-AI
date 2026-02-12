# Related Work

## Vision Transformers and Self-Supervised Learning

### 1. Vision Transformer (ViT)

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** [1]
- **Authors:** Dosovitskiy et al. (Google Research, Brain Team)
- **Publication:** ICLR 2021
- **arXiv:** 2010.11929

**Key Contributions:**
- Demonstrated that pure transformer architecture can be applied directly to sequences of image patches without relying on CNNs
- Achieved state-of-the-art results on ImageNet and other benchmarks when pre-trained on large datasets
- Showed that transformers can outperform CNNs when given sufficient pre-training data (14M-300M images)

**Key Results:**
- ViT-Large pre-trained on ImageNet-21k (14M images): **87.76% top-1 accuracy** on ImageNet-1k
- ViT-Huge pre-trained on JFT-300M: **88.55% top-1 accuracy** on ImageNet-1k
- Requires less computational resources for pre-training compared to state-of-the-art CNNs
- Demonstrates excellent transfer learning performance across multiple downstream tasks

**Architecture:**
- Splits images into 16×16 patches
- Projects patches to embedding space
- Adds positional embeddings
- Processes through standard transformer encoder
- Uses [CLS] token for classification

---

### 2. MoCo v3: Self-Supervised Vision Transformers

**An Empirical Study of Training Self-Supervised Vision Transformers** [2]
- **Authors:** Xinlei Chen, Saining Xie, Kaiming He (Facebook AI Research)
- **Publication:** ICCV 2021
- **arXiv:** 2104.02057

**Key Contributions:**
- Investigated fundamental components for training self-supervised Vision Transformers
- Identified and solved training instability issues in self-supervised ViT training
- Proposed simple modifications to MoCo v2 framework for ViT architectures

**Key Results:**
- MoCo v3 with ViT-Base: **76.7% top-1 accuracy** on ImageNet-1k (linear probing)
- MoCo v3 with ViT-Large: **78.2% top-1 accuracy** on ImageNet-1k (linear probing)
- Competitive with supervised pre-training on downstream tasks
- Self-supervised ViT shows strong performance on object detection and segmentation

**Key Findings:**
- Training instability in self-supervised ViT is caused by large learning rates
- Random patch projection helps stabilize training
- Batch normalization in the projection head improves stability
- Self-supervised ViT requires different training recipes compared to CNNs

---

### 3. DeiT: Data-Efficient Image Transformers

**Training data-efficient image transformers & distillation through attention** [3]
- **Authors:** Hugo Touvron, Matthieu Cord, et al. (Facebook AI & Sorbonne University)
- **Publication:** ICML 2021
- **arXiv:** 2012.12877

**Key Contributions:**
- Introduced data-efficient training strategy for Vision Transformers
- Proposed knowledge distillation method specific to transformers
- Demonstrated competitive performance without large-scale pre-training datasets
- Trained on single computer in less than 3 days

**Key Results:**
- DeiT-Base: **83.1% top-1 accuracy** on ImageNet-1k (trained only on ImageNet-1k)
- DeiT-Base with distillation: **85.2% top-1 accuracy**
- DeiT-Small: **79.8% top-1 accuracy** with only 22M parameters
- Achieves competitive results without JFT-300M or ImageNet-21k pre-training

**Key Techniques:**
- Teacher-student distillation using attention mechanism
- Distillation token in addition to class token
- Hard-label distillation outperforms soft distillation for transformers
- Strong data augmentation and regularization

---

### 4. ConvNets vs Transformers: Recent Comparisons

**ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy** [4]
- **arXiv:** 2311.09215 (2023)

**Key Findings:**
- Performance gap between ConvNets and Transformers is smaller than commonly believed
- Properly trained ConvNets can match ViT performance on many tasks
- Architecture choice matters less than pre-training method and dataset scale
- CLIP pre-training benefits both architectures significantly

---

**Battle of the Backbones: A Large-Scale Comparison of Pretrained Models** [5]
- **arXiv:** 2310.19909 (2023)

**Key Findings:**
- Large-scale comparison across multiple computer vision tasks
- No single backbone architecture dominates all tasks
- Pre-training dataset and method have larger impact than architecture
- Transfer learning performance varies significantly across tasks

---

**ConvNets Match Vision Transformers at Scale** [6]
- **arXiv:** 2310.16764 (2023)
- **Authors:** Samuel L. Smith, Andrew Brock et al. (DeepMind)

**Key Findings:**
- ConvNets can match ViT performance when trained at web-scale
- Pre-training on JFT-4B dataset with compute budgets up to 110k TPU-v4 hours
- Challenges the belief that ViTs are fundamentally superior at large scale
- Both architectures benefit similarly from increased scale

---

### 5. Understanding Vision Transformers

**Do Vision Transformers See Like Convolutional Neural Networks?** [7]
- **arXiv:** 2108.08810 (2021)

**Key Findings:**
- ViTs develop different internal representations compared to CNNs
- ViTs have more uniform representations across layers
- Early layers in ViT already capture global information
- Self-attention enables more flexible spatial integration

---

**Three things everyone should know about Vision Transformers** [8]
- **arXiv:** 2203.09795 (2022)
- **Authors:** Hugo Touvron, Matthieu Cord et al.

**Key Findings:**
1. Instabilities during training can be reduced with appropriate normalization
2. Patch size significantly impacts performance and computational cost
3. Positional encoding choices matter for downstream tasks

---

**Better plain ViT baselines for ImageNet-1k** [9]
- **arXiv:** 2205.01580 (2022)
- **Authors:** Lucas Beyer, Xiaohua Zhai, Alexander Kolesnikov (Google)

**Key Results:**
- Improved ViT training recipe achieves **84.5% top-1 accuracy** with ViT-B/16
- Better augmentation and regularization strategies
- Demonstrates importance of training hyperparameters
- Provides reproducible baselines for research community

---

### 6. Self-Supervised Learning Objectives

**Objectives Matter: Understanding the Impact of Self-Supervised Objectives on Vision Transformer Representations** [10]
- **arXiv:** 2304.13089 (2023)

**Key Findings:**
- Joint-embedding methods (SimCLR, MoCo, DINO) vs reconstruction methods (MAE, BEiT, SimMIM)
- Different objectives lead to different learned representations
- Joint-embedding methods learn more semantic features
- Reconstruction methods excel at spatial understanding

---

**What Do Self-Supervised Vision Transformers Learn?** [11]
- **arXiv:** 2305.00729 (2023)

**Key Findings:**
- Contrastive learning (CL) captures longer-range global patterns
- Masked image modeling (MIM) focuses on local texture information
- CL trains self-attentions to capture object shapes in later layers
- Explains why CL and MIM have different downstream performance

---

### 7. Linear Probing Benchmarks

Linear probing is a standard evaluation protocol for self-supervised learning where:
1. A model is pre-trained using self-supervised objectives
2. The backbone is frozen
3. Only a linear classifier is trained on labeled data
4. Performance indicates quality of learned representations

**Typical Linear Probing Results on ImageNet-1k:**
- **Supervised ViT-B/16:** ~82% top-1 accuracy
- **MoCo v3 ViT-B:** 76.7% top-1 accuracy
- **DINO ViT-B:** 78.2% top-1 accuracy
- **MAE ViT-B:** 67.8% top-1 accuracy
- **SimCLR ResNet-50:** 69.3% top-1 accuracy

The linear probing gap between self-supervised and supervised learning has been steadily decreasing, with recent methods achieving within 5-10% of supervised performance.

---

### 8. Pre-training vs Fine-tuning Paradigms

**Key Observations from Recent Literature:**

1. **Scale Matters:**
   - Both ViTs and ConvNets benefit from larger pre-training datasets
   - JFT-300M, JFT-4B, and ImageNet-21k enable significant performance gains
   - Self-supervised pre-training can match or exceed supervised pre-training at scale

2. **Architecture vs Pre-training:**
   - Pre-training method often matters more than architecture choice
   - Properly trained ConvNets can match ViT performance
   - ViTs show advantages in transfer learning and downstream tasks

3. **Data Efficiency:**
   - DeiT demonstrates that ViTs can be trained on ImageNet-1k alone
   - Knowledge distillation significantly improves data efficiency
   - Careful regularization and augmentation are critical

4. **Self-Supervised Learning:**
   - Closing the gap with supervised learning
   - Different objectives (contrastive vs reconstruction) learn different features
   - Linear probing performance steadily improving

---

## References

[1] Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR 2021.

[2] Chen, Xie, He, "An Empirical Study of Training Self-Supervised Vision Transformers," ICCV 2021.

[3] Touvron et al., "Training data-efficient image transformers & distillation through attention," ICML 2021.

[4] "ConvNet vs Transformer, Supervised vs CLIP: Beyond ImageNet Accuracy," arXiv:2311.09215, 2023.

[5] "Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks," arXiv:2310.19909, 2023.

[6] Smith et al., "ConvNets Match Vision Transformers at Scale," arXiv:2310.16764, 2023.

[7] "Do Vision Transformers See Like Convolutional Neural Networks?," arXiv:2108.08810, 2021.

[8] Touvron et al., "Three things everyone should know about Vision Transformers," arXiv:2203.09795, 2022.

[9] Beyer, Zhai, Kolesnikov, "Better plain ViT baselines for ImageNet-1k," arXiv:2205.01580, 2022.

[10] "Objectives Matter: Understanding the Impact of Self-Supervised Objectives on Vision Transformer Representations," arXiv:2304.13089, 2023.

[11] "What Do Self-Supervised Vision Transformers Learn?," arXiv:2305.00729, 2023.
