import torch
import numpy as np

def solve_regularized_kernel_matrix_system(K_XX, K_YY, ridge_lambda):
    """
    Solves the regularized kernel system (K_XX + n * ridge_lambda * I).
    
    Args:
        K_XX (torch.Tensor): Kernel matrix of shape (n, n).
        K_YY (torch.Tensor): Kernel matrix of shape (n, n).
        ridge_lambda (float): Regularization parameter.

    Returns:
        K_XX_inv (torch.Tensor): Inverse of regularized K_XX.
        K_XX_inv_K_YY (torch.Tensor): (K_XX + n * ridge_lambda * I)^(-1) @ K_YY.
    """
    n = K_XX.shape[0]
    K_XX = add_diag(K_XX, n*ridge_lambda).double()
    K_YY = torch.cat((torch.eye(n).to(K_XX.device), K_YY), 1).double()

    W_all = torch.linalg.solve(K_XX, K_YY)

    K_XX_inv = W_all[:, :n]      # (K_XX + n*lambda*I)^(-1)
    K_XX_inv_K_YY = W_all[:, n:]  # (K_XX + n*lambda*I)^(-1) K_YY

    return K_XX_inv.float(), K_XX_inv_K_YY.float()


def compute_pdist_sq(x, y):
    """compute the squared paired distance between x and y."""
    if len(x.shape) == 1:
        return (x[:, None] - y[None, :]) ** 2

    if len(x.shape) != 2:
        raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')
    if y is not None:
        if len(y.shape) != 2:
            raise ValueError(f'x should be 1 or 2-dim, but it is {len(x.shape)}-dim')

        x_norm = torch.linalg.norm(x, dim=1, keepdim=True)
        y_norm = torch.linalg.norm(y, dim=1, keepdim=False)[None, :]

        return torch.clamp(x_norm ** 2 + y_norm ** 2 - 2.0 * x @ y.T, min=0)

    a = x.reshape(x.shape[0], -1)
    aTa = a @ a.T
    aTa_diag = torch.diag(aTa)
    aTa = torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0)

    ind = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    aTa[ind[0], ind[1]] = 0
    return aTa + aTa.transpose(0, 1)


def add_diag(x, val):
    """Add a scalar value to the diagonal of a square matrix."""
    
    if len(x.shape) != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f'x is not a square matrix: shape {x.shape}')

    idx = range(x.shape[0])
    y = x.clone()
    y[idx, idx] += val
    return y


def get_error_mean_std(data, ground_truth, pval):
    """Compute the mean and standard deviation of the error rate."""
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if ground_truth == 'H0':
        data = 1 - (data >= pval).astype(float)
    else:
        data = (data >= pval).astype(float)
    mean = data.mean()
    # if len(data) == 1:
    std = data.std()
    # else:
    #     std = std_xy(data)
    return mean, std


def get_data_median(data):
    """
    Compute the RBF kernel bandwidth using the median heuristic.
    
    Parameters:
        data (torch.Tensor): Input data of shape (n_samples, n_features).
    
    Returns:
        float: Median of pairwise squared distances.
    """
    # Compute pairwise squared distances
    pairwise_distances = compute_pdist_sq(data, data)  # Shape: (n_samples, n_samples, n_features)
    # Extract upper triangular part, excluding diagonal, and compute the median
    upper_triangular_indices = torch.triu_indices(pairwise_distances.size(0), pairwise_distances.size(1), offset=1)
    distances = pairwise_distances[upper_triangular_indices[0], upper_triangular_indices[1]]
    median_distance = torch.median(distances)
    
    # Return the bandwidth (sigma^2)
    return median_distance

def get_data_mean(data):
    """
    Compute the RBF kernel bandwidth using the median heuristic.
    
    Parameters:
        data (torch.Tensor): Input data of shape (n_samples, n_features).
    
    Returns:
        float: Median of pairwise squared distances.
    """
    # Compute pairwise squared distances
    pairwise_distances = compute_pdist_sq(data, data)  # Shape: (n_samples, n_samples, n_features)
    # Extract upper triangular part, excluding diagonal, and compute the median
    upper_triangular_indices = torch.triu_indices(pairwise_distances.size(0), pairwise_distances.size(1), offset=1)
    distances = pairwise_distances[upper_triangular_indices[0], upper_triangular_indices[1]]
    mean_distance = torch.mean(distances)

    # Return the bandwidth (sigma^2)
    return mean_distance



