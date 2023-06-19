import numpy as np
import torch
from typing import TypeVar

torch.multiprocessing.set_sharing_strategy('file_system')
Tensor = TypeVar('Tensor')
ndarray = TypeVar('ndarray')


def rel_mse_np(x_true: Tensor, x_est: Tensor) -> Tensor:
    return np.sum(np.abs(x_true - x_est) ** 2, axis=-1) / np.mean(np.sum(np.abs(x_true) ** 2, axis=-1))


def dft_matrix(n: int):
    """
    Determines the DFT matrix of size n x n for given n
    :param n: number of row/columns in matrix
    :return: DFT matrix of size n x n
    """
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(-2 * np.pi * 1j / n)
    F = np.power(omega, i * j) / np.sqrt(n)
    return torch.tensor(F, dtype=torch.complex64)


def compute_lmmse(C_h: Tensor, mu: Tensor, y: Tensor, sigma, C_y=None, A=None, device='cpu') -> Tensor:
    B, N, M = C_h.shape[0], mu.shape[1], y.shape[1]

    # preprocess real and imaginary parts of components
    h_est = torch.zeros((B, N), dtype=torch.cfloat, device=device)

    # create identity matrix for A if it is None
    if A is None:
        A = torch.eye(M, dtype=torch.cfloat).to(device)

    for i in range(B):
        # compute LMMSE estimate for given observation and delta: h = mu_h + C_h*A^H (A*C_h*A^H + C_n)^-1 (y - A*mu_h)
        rhs = y[i] - torch.matmul(A, mu[i])
        if C_y is None:
            C_n = sigma[i]**2 * torch.eye(N, device=device)
            h_est[i] = mu[i] + C_h[i] @ A.H @ torch.linalg.solve(A @ C_h[i] @ A.H + C_n, rhs)
        else:
            h_est[i] = mu[i] + C_h[i] @ A.H @ torch.linalg.solve(C_y[i], rhs)
    return h_est
