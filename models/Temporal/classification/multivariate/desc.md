# Multivariate Time Series Classification

Multivariate time series classification extends beyond the simpler univariate time series classification by introducing the complexity of inter-time series dependencies. This method is crucial for capturing intricate relationships between multiple time series data, which is often necessary in medical and brain activity monitoring:

## Examples

- **Health Monitoring**: Multivariate analysis of patient health involves integrating data from various health sensors like heart rate monitors, blood pressure sensors, and glucose monitors. This approach helps identify complex health patterns that are not visible through single data streams alone.
- **Brain Activity Analysis**: In the context of EEGs, each node's activity can be understood better in a networked context rather than in isolation. This holistic view aids in identifying distinct patterns that help differentiate between neurological conditions.

## Graph Neural Networks (GNNs)

- The complexity and interconnected nature of these time series make them suitable for analysis using spatiotemporal GNNs. These networks are suitable at capturing both inter-variable and temporal dependencies, making them ideal for such tasks.

### Raindrop Model

- The **Raindrop** architecture demonstrates the capability of spatiotemporal GNNs to handle and classify multivariate time series effectively. It is particularly noted for its ability to manage irregularly sampled data with missing values by adaptively learning a graph structure and interpolating missing observations.
