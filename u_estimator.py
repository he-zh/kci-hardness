"""
Pytorch implementation of U-statistic estimator for HSIC.
Adapted from
https://github.com/romanpogodin/kernel-ci-testing/blob/main/splitkci/dependence_measures.py#L26
"""


import torch


def compute_hsic(Kx, Ky, biased):
    n = Kx.shape[0]
    Kx = Kx.double()
    Ky = Ky.double()
    if biased:
        a_vec = Kx.mean(dim=0)
        b_vec = Ky.mean(dim=0)
        # same as tr(HAHB)/m^2 for A=a_matrix, B=b_matrix, H=I - 11^T/m (centering matrix)
        mean =  (Kx * Ky).mean() - 2 * (a_vec * b_vec).mean() + a_vec.mean() * b_vec.mean()
        return mean.float()


    else:
        tilde_Kx = Kx - torch.diagflat(torch.diag(Kx))
        tilde_Ky = Ky - torch.diagflat(torch.diag(Ky))

        u = tilde_Kx * tilde_Ky
        k_row = tilde_Kx.sum(dim=1)
        l_row = tilde_Ky.sum(dim=1)
        mean_term_1 = u.sum()  # tr(KL)
        mean_term_2 = k_row.dot(l_row)  # 1^T KL 1
        mu_x = tilde_Kx.sum()
        mu_y = tilde_Ky.sum()
        mean_term_3 = mu_x * mu_y

        # Unbiased HISC.
        mean = 1 / (n * (n - 3)) * (mean_term_1 - 2. / (n - 2) * mean_term_2 + 1 / ((n - 1) * (n - 2)) * mean_term_3)
        return mean.float()

