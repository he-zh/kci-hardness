import numpy as np
import torch


def synthetic(n_points, ground_truth='H0', dim=1, 
             ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0, alpha=0.1, beta=1.0,
             device='cuda', seed=0, **ignored):
    """
    Generate data for the conditional independence test.
    """
    c = np.random.RandomState(seed=seed*(n_points+1)).normal(0, 1, size=(n_points, dim))
    f = np.cos
    g = np.exp

    a_m = f(c[:, ca_dim_idx:ca_dim_idx+1])
    b_m = g(c[:, cb_dim_idx:cb_dim_idx+1])

    if ground_truth == 'H1':
        r = np.sin(beta * c[:, cr_dim_idx])
        a_r = np.zeros((n_points, 1))
        b_r = np.zeros((n_points, 1))
        for i in range(n_points):
            cov_matrix = [[1, r[i]], [r[i], 1]]
            a_r[i, 0], b_r[i, 0] = np.random.RandomState(seed=seed*(n_points+1)+1+i).multivariate_normal([0, 0], cov_matrix)

    elif ground_truth == 'H0':
        a_r = np.random.RandomState(seed=seed*(n_points+1)+1).normal(0, 1, size=(n_points, 1))
        b_r = np.random.RandomState(seed=seed*(n_points+1)+2).normal(0, 1, size=(n_points, 1))
    else:
        raise NotImplementedError(f'{ground_truth} has to be H0 or H1')

    a = a_m + alpha * a_r
    b = b_m + alpha * b_r

    a = torch.tensor(a, dtype=torch.float32).to(device)
    b = torch.tensor(b, dtype=torch.float32).to(device)
    c = torch.tensor(c, dtype=torch.float32).to(device)

    return a, b, c

