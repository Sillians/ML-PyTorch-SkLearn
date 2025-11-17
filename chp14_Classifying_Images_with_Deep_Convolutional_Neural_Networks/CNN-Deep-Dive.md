# **Deep Convolutional Neural Networks (Deep Dive)**

### 1. Overview

A **Deep Convolutional Neural Network (CNN)** is a neural architecture designed to learn spatial hierarchies of features through convolution operations. While a standard fully connected network treats all input dimensions equally, a CNN preserves local spatial structure, making it especially effective in domains such as image classification, object detection, segmentation, video analysis, and speech processing.



### 2. Core Mathematical Foundations

#### 2.1 Convolution Operation

Given an input feature map $`X \in \mathbb{R}^{H \times W \times C_{\text{in}}}`$ and a filter/kernel $`K \in \mathbb{R}^{k \times k \times C_{\text{in}}}`$, the convolution output $`Y`$ is computed as:

$$Y(i, j) = \sum_{u=1}^{k} \sum_{v=1}^{k} \sum_{c=1}^{C_{\text{in}}} X(i+u, j+v, c)\, K(u,v,c)$$

This operation captures local spatial correlations.


#### 2.2 Stride and Padding

* **Stride $`s`$**: Step size taken when sliding the kernel.
* **Padding $`p`$**: Number of pixels added around the input.

Output spatial dimension:

$$H_{\text{out}} = \frac{H + 2p - k}{s} + 1,\quad 
W_{\text{out}} = \frac{W + 2p - k}{s} + 1$$

Proper padding enables feature maps to maintain resolution across layers.


#### 2.3 Feature Hierarchy

* Early layers capture edges, corners, and color gradients.
* Mid layers capture textures and shapes.
* Deep layers capture semantic object-level representations.



### 3. Building Blocks of CNN Architectures

| Component             | Purpose                             | Notes                             |
| --------------------- | ----------------------------------- | --------------------------------- |
| Convolution Layer     | Extract spatial-local features      | Controls learned receptive fields |
| Nonlinearity (ReLU)   | Introduces non-linearity            | $`f(x) = \max(0,x)`$              |
| Pooling (Max/Average) | Spatial downsampling                | Increases translation invariance  |
| Batch Normalization   | Stabilizes and accelerates training | Normalizes feature activations    |
| Dropout               | Regularization                      | Randomly zeroes activations       |
| Fully Connected Layer | Final reasoning/classification      | Flattens and outputs class logits |




### 4. Depth, Receptive Fields, and Inductive Biases

A key characteristic of CNNs is the notion of **receptive field**—the region of the input affecting a particular output value. Stacking convolutional layers **increases the effective receptive field** without increasing parameter count significantly, enabling CNNs to capture hierarchical structure.

CNNs embed strong inductive biases:

* **Locality**: Patterns depend on nearby pixels.
* **Translation equivariance**: A pattern shifted in the input produces a shifted activation.

This decreases the number of parameters drastically compared to MLPs.



### 5. Modern CNN Architectures (Conceptual)

| Architecture | Key Idea                                   | Benefit                       |
| ------------ | ------------------------------------------ | ----------------------------- |
| LeNet        | Small, simple CNN                          | Early digit recognition       |
| AlexNet      | Deep CNN + ReLU                            | First major ImageNet success  |
| VGG          | Repeated $`3 \times 3`$ convolutions       | Simplicity and uniform design |
| ResNet       | Residual (skip) connections                | Enables very deep networks    |
| Inception    | Multi-scale filters in parallel            | Efficient feature extraction  |
| EfficientNet | Compound scaling of depth/width/resolution | Optimal parameter usage       |



### 6. Training and Optimization Considerations

#### 6.1 Loss Function

For multi-class image classification:

$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

#### 6.2 Weight Initialization

He initialization is well-suited for ReLU:

$$W \sim \mathcal{N}\left(0, \frac{2}{\text{fan\_in}} \right)$$

#### 6.3 Regularization Strategies

| Method            | Effect                           |
| ----------------- | -------------------------------- |
| Dropout           | Reduces co-adaptation of neurons |
| Data Augmentation | Improves generalization          |
| Weight Decay      | Penalizes large weights          |
| Early Stopping    | Prevents overfitting             |



### 7. Computational Considerations

| Factor           | Impact                                                        |
| ---------------- | ------------------------------------------------------------- |
| Depth            | Enhances expressiveness, increases difficulty of optimization |
| Width            | Controls feature richness                                     |
| Kernel Size      | Controls locality of receptive fields                         |
| Memory Bandwidth | Often bottleneck during training                              |
| Accelerator Type | GPUs/MPS/TPUs greatly speed convolution                       |



### 8. Conceptual Summary

CNNs learn hierarchical representations by applying convolution filters over spatial regions with shared parameters. This yields models that are both **parameter-efficient** and **structurally aligned** with the properties of natural images. Depth enables abstraction, and architectural improvements (such as residual connections) allow effective training of very deep networks.


---

# **Vision Transformers vs Convolutional Neural Networks (Deep Dive)**



### 1. **Core Conceptual Difference**

| Aspect                   | CNNs                                        | Vision Transformers (ViTs)                             |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------ |
| **Inductive Bias**       | Strong (locality, translation equivariance) | Minimal (global context learned from data)             |
| **Feature Construction** | Hierarchical spatial filters                | Self-attention over tokenized image patches            |
| **Representation**       | Local-to-global feature buildup via depth   | Global interactions from the start                     |
| **Data Requirement**     | Efficient on small-to-medium datasets       | Typically requires large-scale datasets or pretraining |

CNNs assume that local image structure matters, embedding this assumption directly into the architecture.
ViTs do not assume locality — they allow **any patch to interact with any other patch**, learning structure from data.



### 2. **How CNNs See Images**

CNNs process images using convolution kernels that slide over local neighborhoods.

If $`X \in \mathbb{R}^{H \times W \times C}`$ is the input and $`K \in \mathbb{R}^{k \times k \times C}`$ is a filter:


$$Y(i, j) = \sum_{u=1}^{k} \sum_{v=1}^{k} \sum_{c=1}^{C} X(i+u, j+v, c)\, K(u, v, c)$$

Key properties:

* **Shared weights** → fewer parameters
* **Local receptive fields** → structure preserved
* **Stacking layers** expands effective receptive field → deeper layers capture global structure indirectly

This enforces **translation equivariance**, meaning patterns detected in one region are transferable to others.



### 3. **How Vision Transformers See Images**

ViTs treat an image as a sequence, like text in NLP.

**Step 1: Patch Tokenization**

An image $`X \in \mathbb{R}^{H \times W \times C}`$ is divided into patches of size $`P \times P`$, producing $`N = \frac{HW}{P^2}`$ patches.

Each patch is flattened:

$$x_i = \mathrm{Flatten}(X_{i}) W_e, \quad W_e \in \mathbb{R}^{(P^2 C) \times D}$$

So we obtain a sequence:

$$Z = [x_1, x_2, \dots, x_N] + E_{\text{pos}},$$

where $`E_{\text{pos}}`$ adds positional information (since transformers have no built-in notion of spatial layout).



### 4. **Self-Attention Mechanism**

For each token representation:


$$Q = Z W_Q,\quad K = Z W_K,\quad V = Z W_V$$


Attention output:


$$\mathrm{Attention}(Q, K, V) = \mathrm{Softmax}\left(\frac{QK^\top}{\sqrt{D}}\right)V$$


This yields **global receptive field at every layer**.



### 5. **Philosophical Distinction**

| Question               | CNN Answer                                                                | ViT Answer                                                       |
| ---------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| How to find structure? | Hard-coded local processing → “learn patterns from neighborhoods outward” | Let the model **learn dependencies anywhere** from the beginning |
| Architecture says:     | “Local first, global later.”                                              | “Global from the start.”                                         |

CNN = **Built-in visual prior**
ViT = **Data-driven representation learning**



### 6. **Practical Implications**

| Category                     | CNN                         | ViT                                           |
| ---------------------------- | --------------------------- | --------------------------------------------- |
| Works well on small datasets | Yes                         | Usually no (unless pretrained)                |
| Parameter efficiency         | Generally efficient         | Can be larger but optimized variants exist    |
| Long-range relationships     | Harder (must rely on depth) | Natural & direct (via attention)              |
| Interpretability             | Feature maps intuitive      | Attention maps interpretable                  |
| Compute Scaling              | Kernel operations optimized | Attention scales quadratically in token count |



### 7. **Modern Trends**

1. **Hybrid Models**
   CNN stem for local patterns + Transformer blocks for global reasoning.

2. **Conv-Based Self-Attention (e.g., ConvNeXt)**
   CNNs redesigned to behave transformer-like.

3. **Efficient Attention Architectures**
   Reduce quadratic attention complexity.

Overall trend:

* CNNs are **not obsolete**
* Transformers dominate **large-scale vision + multi-modal tasks**
* Best architectures increasingly **blend both paradigms**



### 8. **When to Use Which**

| If you have…                                | Recommended                                             |
| ------------------------------------------- | ------------------------------------------------------- |
| Limited data                                | CNN or pretrained ViT                                   |
| Large-scale datasets                        | ViT                                                     |
| Need maximum accuracy regardless of compute | ViT / Hybrid                                            |
| Real-time / edge deployment                 | CNN / Efficient ConvNet (e.g., MobileNet, EfficientNet) |



### 9. **Conceptual Summary**

* CNNs rely on **local spatial priors** and build global features gradually.
* Vision Transformers use **self-attention** to obtain **global interactions immediately**.
* ViTs generally require **large data or strong pretraining**, while CNNs work well with stronger built-in biases and fewer data.

Both remain deeply relevant; modern architectures merge their strengths.

---
