# Attention Based Spatial-Temporal Graph Convolutional Networks (ASTGCN)

## Traffic Flow Forecasting

The goal of traffic flow forecasting in this context is to predict future traffic conditions based on historical traffic data.
The model forecasts traffic flows by learning from past time slices and predicting future sequences for each node in the network.

## Model Architecture

### Spatial Module

- **Graph Convolutional Networks (GCN)**: Used to capture the spatial dependencies among nodes in the traffic network.

### Attention Mechanisms(Temporal)

- **Spatial Attention (SAtt)**: Highlights important spatial features that are crucial for predicting traffic flow.
- **Temporal Attention (TAtt)**: Identifies significant temporal patterns, helping the model to focus on critical moments in historical data.

## Framework Overview

The framework of ASTGCN consists of multiple components designed to handle different aspects of the data:

- **ST blocks**: Each spatial-temporal block contains layers for both graph convolution and attention mechanisms, to process spatial and temporal data concurrently.
- **Fusion Layer**: Integrates features processed from different time components—recent, daily, and weekly patterns—to form a comprehensive feature set for prediction.
- **Loss Function**: The model optimizes a loss function that measures the accuracy of the traffic forecasts against actual observed traffic flows.

## Temporal Segmentation Strategy

To capture various periodic dependencies in the traffic data effectively, ASTGCN segments the input data into different temporal parts:

- **Recent Segment (X^(T))**: Captures the immediate past information, crucial for short-term forecasting.
- **Daily Segment (X^(D))**: Models daily periodic patterns, useful for understanding daily traffic cycles.
- **Weekly Segment (X^(W))**: Accounts for weekly periodic trends, important for capturing behaviors like weekend vs. weekday differences.

### Performance Metrics (PeMSD4 Dataset):

- **RMSE**:
  - HA: 54.14
  - ARIMA: 68.13
  - VAR: 51.73
  - LSTM: 45.82
  - GRU: 45.11
  - STGCN: 38.29
  - GLU-STGCN: 38.41
  - GeoMAN: 37.84
  - MSTGCN (ours): 35.64
  - ASTGCN (ours): **32.82**
- **MAE**:
  - HA: 36.76
  - ARIMA: 32.11
  - VAR: 33.76
  - LSTM: 29.45
  - GRU: 28.65
  - STGCN: 25.15
  - GLU-STGCN: 27.28
  - GeoMAN: 23.64
  - MSTGCN (ours): 22.73
  - ASTGCN (ours): **21.80**

### Performance Metrics (PeMSD8 Dataset):

- **RMSE**:

  - HA: 44.03
  - ARIMA: 43.30
  - VAR: 31.21
  - LSTM: 36.96
  - GRU: 35.95
  - STGCN: 27.87
  - GLU-STGCN: 30.78
  - GeoMAN: 28.91
  - MSTGCN (ours): 26.47
  - ASTGCN (ours): **25.27**

- **MAE**:
  - HA: 29.52
  - ARIMA: 24.04
  - VAR: 21.41
  - LSTM: 23.18
  - GRU: 22.20
  - STGCN: 18.88
  - GLU-STGCN: 20.99
  - GeoMAN: 17.84
  - MSTGCN (ours): 17.47
  - ASTGCN (ours): **16.63**

## Hyperparameters

- **Chebyshev Polynomial (K)**: It was observed that as \( K \) increases, the forecasting performance slightly improves. The optimal value for \( K \) was set to 3 due to its balance between computational efficiency and performance improvement.
- **Kernel Size**: The kernel size in the temporal dimension is also set to 3 to optimize performance.
- **Convolution Kernels**: Each graph convolution layer utilizes 64 convolution kernels to process spatial features. Similarly, each temporal convolution layer also uses 64 convolution kernels.

- **Loss Function**: Mean Square Error (MSE) is used as the loss function, which measures the accuracy of predictions against the ground truth.
- **Optimization**: The loss is minimized through backpropagation during training.
- **Batch Size**: 64
- **Learning Rate**: Set at 0.0001 for optimal convergence.
