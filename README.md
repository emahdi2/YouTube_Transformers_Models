# 🚀 Deep Learning Models for Cryptocurrency Price Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)
![Manim](https://img.shields.io/badge/manim.community-blue)
---

## 📌 Overview

This repository presents a comprehensive study of machine learning and deep learning models for predicting cryptocurrency prices, focusing on **Bitcoin (BTC)** and **Ethereum (ETH)**.

It includes:

* Classical neural models
* Recurrent neural networks
* Transformer-based architectures
* A **novel hybrid Transformer–RNN approach**

---

## 🎥 Project Videos

### 🔹 Video 1: Classical & Recurrent Models

[![Watch Video](https://img.youtube.com/vi/Ixdz06hpYM8/0.jpg)](https://youtu.be/Ixdz06hpYM8)

Covers:

* Radial Basis Function Network (RBFN)
* General Regression Neural Network (GRNN)
* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU)

---

### 🔹 Video 2: Transformers & Hybrid Model

[![Watch Video](https://img.youtube.com/vi/e4coxS26gr8/0.jpg)](https://youtu.be/e4coxS26gr8)

Covers:

* Transformers and Attention Mechanisms
* Hybrid Transformer + LSTM/GRU
* Application to Bitcoin & Ethereum prediction

---

## 🧠 Models Covered

### 1. Radial Basis Function Network (RBFN)

A feedforward neural network using radial basis activation functions:

$$
y(x) = \sum_{i=1}^{N} w_i \cdot \phi(||x - c_i||)
$$

---

### 2. General Regression Neural Network (GRNN)

A probabilistic neural network for regression:

$$
\hat{y}(x) = \frac{\sum_{i=1}^{N} y_i \exp\left(-\frac{||x - x_i||^2}{2\sigma^2}\right)}{\sum_{i=1}^{N} \exp\left(-\frac{||x - x_i||^2}{2\sigma^2}\right)}
$$

---

### 3. Long Short-Term Memory (LSTM)

Captures long-term dependencies using gating mechanisms:

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$

---

### 4. Gated Recurrent Unit (GRU)

A simplified LSTM variant:

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

---

### 5. Transformers & Attention

Self-attention mechanism:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

### 6. 🔥 Novel Hybrid Model

Combines:

* Transformer (global dependencies)
* LSTM/GRU (temporal dynamics)

**Architecture Flow:**

```
Input Data → Embedding → Transformer Encoder →
Sequence Features → LSTM/GRU → Dense Layer → Output Prediction
```

---

## 📊 Application: Cryptocurrency Prediction

The models are applied to:

* Bitcoin (BTC)
* Ethereum (ETH)

### Features Used:

* Historical prices (lagged values)
* Market indicators
* Sentiment indicators (Fear & Greed Index)

---

## ⚖️ Model Comparison

| Model       | Strength          | Weakness                 |
| ----------- | ----------------- | ------------------------ |
| RBFN        | Fast training     | Limited scalability      |
| GRNN        | Strong regression | Memory intensive         |
| LSTM        | Long-term memory  | Complex training         |
| GRU         | Efficient         | Slightly less expressive |
| Transformer | Global attention  | Data-hungry              |
| Hybrid      | Best performance  | Higher complexity        |

---

## 📈 Results

* Hybrid Transformer model outperforms standalone models
* Attention improves long-range dependency learning
* Deep learning models outperform classical approaches

---

## 🛠️ Tech Stack

* Python
* TensorFlow / PyTorch
* NumPy / Pandas
* Matplotlib / Seaborn

---

## 📚 Future Work

* Multi-asset prediction
* Reinforcement learning for trading
* Real-time deployment

---

## ⭐ Support

If you find this useful, please ⭐ the repository!
