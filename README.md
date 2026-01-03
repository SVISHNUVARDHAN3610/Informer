# Stock Market Forecasting using Informer Transformer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

## 📖 Project Description

### The Challenge
Stock market data is inherently volatile and non-stationary, making accurate forecasting a significant challenge. While traditional Deep Learning models like RNNs and LSTMs excel at short-term dependencies, they often struggle with long-range dependency alignment due to vanishing gradients and serial processing limitations.

Standard Transformer models solved the long-range dependency issue but introduced a new bottleneck: **Quadratic computation complexity $O(L^2)$**, making them memory-intensive and slow for long sequences.

### The Solution: Informer
This project implements the **Informer** model, a state-of-the-art Transformer variant optimized for **Long Sequence Time-Series Forecasting (LSTF)**. By replacing the standard self-attention mechanism with **ProbSparse Attention**, the Informer reduces the time complexity and memory usage to $\mathcal{O}(L \log L)$.

This allows the model to:
1.  **Process Longer Histories:** Look further back in time to identify trends without running out of memory.
2.  **Focus on "Active" Signals:** The ProbSparse mechanism filters out "lazy" queries, focusing the model's attention only on the most significant data points.
3.  **Predict Efficiently:** The generative style decoder predicts long future sequences in a single forward step, rather than the slow step-by-step generation of traditional decoders.

## 🏗️ Technical Architecture

The repository features a custom PyTorch implementation of the complete Informer stack:

1.  **Input Embeddings:**
    * Combines **Scalar Projection** (for the price value) with **Positional Embeddings** and **Global Time Stamp Embeddings** (Hour, Day, Month) to preserve temporal context.

2.  **The Encoder (Feature Extraction):**
    * Uses **ProbSparse Self-Attention** to capture long-range dependencies efficiently.
    * Implements **Self-Attention Distilling** layers between blocks to compress the feature map dimensions, extracting the most dominant features while reducing network size.

3.  **The Decoder (Generative Forecasting):**
    * Receives a "Start Token" (a segment of the recent past) and predicts the future sequence.
    * Uses masked multi-head attention to prevent the model from seeing future values during training (look-ahead mask).

4.  **Optimization:**
    * **Loss Function:** MSE Loss (Mean Squared Error).
    * **Optimizer:** Adam optimizer with dynamic learning rate adjustment.
    * ** regularization:** Dropout layers are integrated to prevent overfitting on noisy financial data.
      
The goal is to capture long-range dependencies in financial data to forecast future price movements effectively.

## 🚀 Key Features
* **Informer Architecture:** Implements Encoder-Decoder stack with ProbSparse Self-Attention mechanism (`model.py`).
* **Modular Training:** Training logic is decoupled into a dedicated trainer class (`trainer.py`).
* **Metric Tracking:** Tracks MSE, MAE, RMSE, and R2 score during training.
* **Visualization:** Automatically generates and saves loss curves and validation plots (`logs/batch-size-32/ploting/`).
* **Data Pipeline:** Custom dataset handling for time-series sequences (`utils/dataset.py`).

## 📊 Performance Results
The model was trained for 16 epochs with a batch size of 32. Below are the metrics from the final converged state:

| Metric | Value (Validation) |
| :--- | :--- |
| **Loss** | 0.5786 |
| **MSE** | 0.9150 |
| **RMSE** | 0.9565 |
| **MAE** | 0.6249 |
| **R² Score** | 0.0432 |

### Visualizations
Training logs and plots are automatically saved to `logs/batch-size-32/ploting/`:
* `avg_loss_plot_bs_32.png`: Average loss per epoch.
* `train_valid_plot.png`: Comparison of training vs validation loss.
* `valid_plot.png`: Predictions vs Actual values on the validation set.

## 📂 Project Structure
```bash
INFORMER/
├── logs/
│   └── batch-size-32/
│       ├── model-weights/
│       │   └── model.pt              # Saved best model weights
│       └── ploting/
│           ├── avg_loss_plot_bs_32.png
│           ├── train_valid_plot.png
│           └── valid_plot.png
├── utils/
│   ├── dataset.py                    # Data loading and preprocessing
│   └── utils.py                      # Helper functions
├── final_check.py                    # Script for quick model verification
├── model.py                          # Informer architecture definition
├── train.py                          # Main entry point for training
└── trainer.py                        # Training loop implementation
```
