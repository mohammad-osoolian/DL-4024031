import numpy as np

def forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode):
    """
    Performs the forward propagation through a BatchNorm layer.

    Arguments:
    Z -- input, with shape (num_examples, num_features)
    gamma -- vector, BN layer parameter
    beta -- vector, BN layer parameter
    eps -- scalar, BN layer hyperparameter
    beta_avg -- scalar, beta value to use for moving averages
    mode -- boolean, indicating whether used at 'train' or 'test' time

    Returns:
    out -- output, with shape (num_examples, num_features)
    """

    if mode == 'train':
        # TODO: Mean of Z across first dimension
        mu = Z.mean(axis=0)

        # TODO: Variance of Z across first dimension
        var = Z.var(axis=0)

        # Take moving average for cache_dict['mu']
        cache_dict['mu'] = beta_avg * cache_dict['mu'] + (1-beta_avg) * mu

        # Take moving average for cache_dict['var']
        cache_dict['var'] = beta_avg * cache_dict['var'] + (1-beta_avg) * var

    elif mode == 'test':
        # TODO: Load moving average of mu
        mu = cache_dict['mu']

        # TODO: Load moving average of var
        var = cache_dict['var']

    # TODO: Apply z_norm transformation
    Z_norm = (Z - mu) / np.sqrt(var + eps)

    # TODO: Apply gamma and beta transformation to get Z tiled
    out = gamma * Z_norm + beta

    return out

Z = np.array([[1,2], [3,4]])
gamma = 1
beta = 0
beta_avg = 0
eps = 0
cache_dict = {'mu': np.array([2, 3]), 'var': np.array([2, 3])}
mode = 'train'

print("TEST normalization")
print(forward_batchnorm(Z, gamma, beta, eps, cache_dict, beta_avg, mode))
