
"""
P-value computations for kernel-based independence tests.
Adapted from https://github.com/romanpogodin/kernel-ci-testing/blob/main/splitkci/pval_computations.py
"""


import torch
from scipy.stats import gamma as gamma_distr
from scipy.stats import norm as norm_distr
from numpy.linalg import eigh
import numpy as np
from u_estimator import compute_hsic


def center_kernel_matrix(K):
    K = K - K.mean(axis=1, keepdims=True)
    return K - K.mean(axis=0, keepdims=True)

def get_uuprod(Kx, Ky, thresh):
    wx, vx = eigh(0.5 * (Kx + Kx.T))
    wy, vy = eigh(0.5 * (Ky + Ky.T))
    idx = np.argsort(-wx)
    idy = np.argsort(-wy)
    wx = wx[idx]
    vx = vx[:, idx]
    wy = wy[idy]
    vy = vy[:, idy]
    vx = vx[:, wx > np.max(wx) * thresh]
    wx = wx[wx > np.max(wx) * thresh]
    vy = vy[:, wy > np.max(wy) * thresh]
    wy = wy[wy > np.max(wy) * thresh]
    vx = vx.dot(np.diag(np.sqrt(wx)))
    vy = vy.dot(np.diag(np.sqrt(wy)))

    T = Kx.shape[0]
    num_eigx = vx.shape[1]
    num_eigy = vy.shape[1]
    size_u = num_eigx * num_eigy
    uu = np.zeros((T, size_u))
    for i in range(0, num_eigx):
        for j in range(0, num_eigy):
            uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

    if size_u > T:
        uu_prod = uu.dot(uu.T)
    else:
        uu_prod = uu.T.dot(uu)
    return uu_prod, size_u


def null_sample_spectral(uu_prod, size_u, T, thresh, null_samples):
    from numpy.linalg import eigvalsh

    eig_uu = eigvalsh(uu_prod)
    eig_uu = -np.sort(-eig_uu)
    ratio = np.max(eig_uu**2)/np.sum(eig_uu**2)
    eig_uu = eig_uu[0:np.min((T, size_u))]
    eig_uu = eig_uu[eig_uu > np.max(eig_uu) * thresh]

    f_rand = np.random.chisquare(1, (eig_uu.shape[0], null_samples))
    null_dstr = eig_uu.T.dot(f_rand)
    return null_dstr


def compute_gamma_pval_approximation(statistic_value, K, L, return_params=False):
    # From KCI: https://arxiv.org/abs/1202.3775
    K = center_kernel_matrix(K)
    L = center_kernel_matrix(L)

    KL = K * L
    mean = torch.diagonal(KL).mean()

    var = 2 * (KL ** 2).mean()

    k = mean ** 2 / var
    theta = var / mean / K.shape[0]  # scaling fix wrt the paper

    if return_params:
        return gamma_distr.sf(statistic_value.item(), a=k.item(), loc=0, scale=theta.item()), k.item(), theta.item()
    return gamma_distr.sf(statistic_value.item(), a=k.item(), loc=0, scale=theta.item())


def compute_chi_square_pval_approximation(statistic_value, K, L, null_samples, thresh=1e-5, return_params=False):

    K = center_kernel_matrix(K)
    L = center_kernel_matrix(L)
    test_stat = torch.sum(K*L).detach().cpu().numpy()
    uu_prod, size_u = get_uuprod(K.cpu().detach().numpy(), L.cpu().detach().numpy(), thresh)
    null_dstr = null_sample_spectral(uu_prod, size_u, K.shape[0], thresh=thresh, null_samples=null_samples)
    # statistic_value = statistic_value.unsqueeze(0).numpy()
    pvalue = sum(null_dstr > test_stat) / float(null_samples)
    if return_params:
        return pvalue, null_dstr

    return pvalue


def compute_wild_bootstrap_pval(statistic_value, K, L, compute_stat_func, return_params=False, n_samples=1000,
                                chunk_size=None):
    Q = torch.randn((n_samples, K.shape[0]), device='cpu')[:, :, None]
    Q[Q >= 0] = 1
    Q[Q < 0] = -1

    def compute_single_val(rademacher_vals):
        KQ = (rademacher_vals * rademacher_vals.T) * K.cpu()
        return compute_stat_func(KQ, L.cpu())

    compute_stat_vals = torch.vmap(compute_single_val, chunk_size=chunk_size)
    # todo: add an exception for OOM that suggests setting a chunk size
    stat_vals = compute_stat_vals(Q)

    pval = (stat_vals > statistic_value.cpu()).float().mean().item()

    if return_params:
        return pval, stat_vals.detach().cpu().numpy()
    return pval



def compute_pval(statistic_value, K, L, is_hsic_biased=False, pval_approx_type='wild', 
                 n_samples=1000, chunk_size=None, return_params=False):
    if not is_hsic_biased and pval_approx_type == 'gamma':
        raise NotImplementedError('P-value calculation for gamma only works for the biased statistic')

    if  pval_approx_type == 'gamma':
        return compute_gamma_pval_approximation(
            statistic_value, K=K, L=L, return_params=return_params)
    elif pval_approx_type == 'wild':
        return compute_wild_bootstrap_pval(
            statistic_value, K=K, L=L,
            return_params=return_params, n_samples=n_samples,
            compute_stat_func=lambda x, y: compute_hsic(x, y, is_hsic_biased),
            chunk_size=chunk_size)
    elif pval_approx_type == 'chi-square':
        return compute_chi_square_pval_approximation(
            statistic_value, K=K, L=L, null_samples=n_samples, thresh=1e-5,)
    else:
        raise NotImplementedError(f'{pval_approx_type} pval_approx_type is not supported.')
