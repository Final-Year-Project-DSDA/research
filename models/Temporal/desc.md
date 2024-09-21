## Notations

### Task Notation

- **M**: Multi-step Forecasting - Predicting multiple future data points.
- **S**: Single-step Forecasting - Predicting the next single data point.
- **S (second letter)**: Short-term Forecasting - Predictions over a shorter time horizon.
- **L**: Long-term Forecasting - Predictions extending further into the future.

### Architecture Notation

- **D**: Discrete - Treating time series data as distinct, separate intervals.
- **C**: Continuous - Handling time series data as continuous flows.
- **C (Coupled)**: Coupled - Model components are directly linked or combined.
- **F**: Factorized - Model components are separated, allowing for modular design or analysis.

### Temporal Module Notation

- **T**: Time Domain - Processing directly on the time series data in the time domain.
- **F**: Frequency Domain - Transforming data to the frequency domain before processing.
- **R**: Recurrence - Utilizing recurrent neural network structures to process temporal data.
- **C**: Convolution - Employing convolutional neural networks for temporal data.
- **A**: Attention - Using attention mechanisms to focus dynamically on different parts of the data.
- **H**: Hybrid - Combining two or more methods from the above to process time series data.

### Input Graph Notation

- **R**: Required - A predefined graph structure is a necessary input for the model.
- **NR**: Not Required - The model operates without a predefined graph structure.
- **O**: Optional - The model can optionally utilize a provided graph structure.

### Learned Graph Relations

- **S**: Static - Graph structure or relations are assumed to be static over time.
- **D**: Dynamic - Graph structure or relations can change over time, and the model adapts to or learns these changes.

### Graph Heuristics

Graph heuristics refer to the principles or methods used to determine how nodes (entities) are linked within a graph structure, particularly in graph-based models such as graph neural networks (GNNs). These heuristics help define the edges between nodes based on certain criteria that describe relationships or interactions among the data elements represented by these nodes.

- **SP**: Spatial Proximity - Nodes are connected based on physical or spatial closeness.
- **PC**: Pairwise Connectivity - Nodes are linked if there is a direct interaction or link.
- **PS**: Pairwise Similarity - Nodes are connected based on the similarity of their attributes or behaviors.
- **FD**: Functional Dependency - Connections are based on functional or causal relationships between nodes.

### Handling Missing Values

- This section indicates whether the GNN methods can address missing values in input time series, crucial for robust and accurate forecasting.
