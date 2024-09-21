# Spatio-Temporal Graph Convolutional Networks (STGCN)

## Model Architecture

### Spatial Module:

- Leverages the topology of road networks through graph convolution.
- Each ST-Conv block includes **one spatial graph convolution layer**.

### Temporal Module:

- Comprises gated CNNs for temporal feature extraction.
- Each ST-Conv block includes **two temporal gated convolution layers**.

### Number of Layers/Blocks:

- The model consists of **multiple spatio-temporal convolutional blocks**.
- Exact number of blocks was not specified in the summary, but each block follows a "sandwich" structure.

### Hyperparameters:

- **Kernel Size**: 3 for both spatial and temporal convolutions.
- **Channels**: Configured as 64, 16, 64 across the layers in each ST-Conv block.

## Performance Metrics

### BJER4 Dataset:

- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **RMSE (Root Mean Squared Error)**

### PeMSD7 Dataset:

- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**
- **RMSE (Root Mean Squared Error)**

### Comparative Analysis:

- STGCN consistently outperformed traditional and other neural network-based models like Historical Average, LSVR, ARIMA, FNN, FC-LSTM, and GCGRU in terms of MAE, MAPE, and RMSE. Exact numerical values were not detailed in the summary.

## Training Method

- **Optimization**: RMSprop with an initial learning rate of \(10^{-3}\) and a decay rate of 0.7 every 5 epochs.
- **Epochs**: Trained for 50 epochs with a batch size of 50.
- **Efficiency**: Demonstrated faster training times and fewer parameters compared to conventional models.
