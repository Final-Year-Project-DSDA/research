# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting

## Datasets Used

- **METR-LA**: Traffic data from loop detectors in Los Angeles County highways, covering 4 months from March to June 2012.
- **PEMS-BAY**: Traffic data from California Transportation Agencies (CalTrans) Performance Measurement System, covering 6 months from January to May 2017.

## Performance

- DCRNN significantly outperforms various baselines in terms of:
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - Root Mean Squared Error (RMSE)
- **Performance Metrics**:
  - **15 min forecast**:
    - MAE: from 3.99 (best baseline) to 2.77
    - RMSE: from 8.45 (worst baseline) to 5.38
    - MAPE: from 9.6% (best baseline) to 7.3%
  - **30 min forecast**:
    - MAE: from 5.41 (worst baseline) to 3.15
    - RMSE: from 10.87 (worst baseline) to 6.45
    - MAPE: from 12.9% (worst baseline) to 8.8%
  - **1 hour forecast**:
    - MAE: from 6.90 (worst baseline) to 3.60
    - RMSE: from 13.76 (worst baseline) to 7.59
    - MAPE: from 17.4% (worst baseline) to 10.5%

## Compared with

1. HA (Historical Average): Uses historical averages as predictions.
2. ARIMA_Kal (ARIMA with Kalman Filter): Combines ARIMA forecasting model with Kalman filter to improve accuracy.
3. VAR (Vector Autoregression): A statistical model that captures the linear interdependencies among multiple time series.
4. SVR (Support Vector Regression): Uses support vector machines for regression tasks.
5. FNN (Feedforward Neural Network): A basic type of neural network with layers that do not form cycles.
6. FC-LSTM (Fully Connected LSTM): A recurrent neural network that uses LSTM (Long Short-Term Memory) units fully connected across time steps.

## Model Architecture

- Uses Encoder-Decoder Architecture
- **Spatial Module**: Utilizes diffusion convolution to model traffic flow as a diffusion process on a directed graph.
- **Temporal Module**: Employs an encoder-decoder architecture using Gated Recurrent Units (GRUs) with diffusion convolution.
- **Number of Layers/Blocks**: Each of the encoder and decoder comprises multiple recurrent layers (specific count not detailed).
- **Hyperparameters**:
  - Random walk steps (`K`): Typically set to 3.
  - Number of units per layer: Adjusted based on experiments, influencing model's reception field.
  - Learning rate, epochs, and batch size are tuned using validation datasets.

## Training Method

- **Scheduled Sampling**: Used to train the sequence-to-sequence model, decreasing the probability of using ground truth data over time during training.

1. Initial Training Phase: In the beginning, the model is trained in a traditional manner where the ground truth data from the previous time steps are fed as inputs to predict the future data points. This helps the model learn the correct dependencies and dynamics based on real data.

2. Gradual Transition: As training progresses, scheduled sampling begins to introduce the model's own predictions into the training process. At each step i, there is a probability ε_i of using the ground truth data and a probability 1 - ε_i of using the model’s own prediction from the previous step as the input for the next prediction.

3. Decreasing Ground Truth Exposure: The probability ε_i of using the ground truth is typically set to decrease gradually over time or over the iterations of training. This decrease might be linear or follow some other schedule, aiming to reduce to zero eventually. By the end of training, the model is primarily being fed its own predictions, mirroring the conditions it will encounter during testing or real-world deployment.

- **Optimizer**: Adam optimizer with learning rate annealing based on performance on validation data.
- **Loss Function**: Likely mean squared error or mean absolute error, focusing on minimizing forecasting error.
- **Validation Strategy**: Uses a split of data with 70% for training, 20% for validation, and 10% for testing.
