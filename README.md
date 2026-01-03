# Stock Market Forecasting using Informer Transformer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)

## 📌 Project Overview
This project implements the **Informer** model (a Transformer variant designed for Long Sequence Time-Series Forecasting) to predict stock market trends. The Informer architecture addresses the high memory usage and computational complexity of traditional Transformers using **ProbSparse Attention** and **Distilling Operations**.

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
