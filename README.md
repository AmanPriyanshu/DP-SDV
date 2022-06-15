# DP-SDV

## Performance:

|name              |PRIV_METRIC_NumericalLR|PRIV_METRIC_NumericalMLP|PRIV_METRIC_NumericalSVR|
|------------------|-----------------------|------------------------|------------------------|
|FAST_ML-DP        |0.083478628            |0.1780978               |0.071120968             |
|FAST_ML           |0.087402661            |0.176534839             |0.074326679             |
|Gaussian Copula-DP|0.093651126            |0.177785047             |0.189530896             |
|Gaussian Copula   |0.066141002            |0.176947755             |0.073740679             |
|CT-GAN-DP         |0.166319716            |0.178170664             |0.189561336             |
|CT-GAN            |0.072312111            |0.173496572             |0.078755983             |
|Copula-GAN-DP     |0.162198436            |0.179451562             |0.18955882              |
|Copula-GAN        |0.084989631            |0.177600009             |0.07810617              |
|TVAE-DP           |0.073633603            |0.176959062             |0.071933513             |
|TVAE              |0.053901697            |0.175550734             |0.075317994             |


## Implementations

1. Tabular Preset

So, adding noise based on the Wishart Mechanism for Differentially Private Principal Components Analysis paper's algorithm 1. Lap(0, 2d/ne), in this d is the number of columns in the covariance matrix taken from `model.get_parameters()`. Now, taking sensitivity=1. We modify the covariance matrix.

2. GaussianCopula Model

So, adding noise based on the Wishart Mechanism for Differentially Private Principal Components Analysis paper's algorithm 1. Lap(0, 2d/ne), in this d is the number of columns in the covariance matrix taken from `model.get_parameters()`. Now, taking sensitivity=1. We modify the covariance matrix.

3. CTGAN Model

Added DP-SGD

4. CopulaGAN Model

Added DP-SGD

5. TVAE Model

Added DP-SGD

