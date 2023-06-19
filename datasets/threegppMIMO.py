import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from utils import dft_matrix


class ThreeGPPMIMO(Dataset):
    def __init__(self, ant, paths, train=True, eval=True, seed=1234, snr=None):
        """
        Input:
        :param ant (str)           : antenna array (e.g. 32rx4tx for 32x4 channels)
        :param paths (str)         : number of paths to consider (e.g. 1-5 for 1 and 5 paths)
        :param train (bool)        : whether to use train or test dataset of same kind
        :param eval (bool)         : whether to use train or test dataset of same kind
        :param seed (int)          : seed for random number (for reproducibility)
        :param snr (float)         : signal-to-noise ratio in samples
        """
        self.rng = np.random.default_rng(seed)
        data_list = []
        if train:
            file_end = '-path-train.npy'
        elif eval:
            file_end = '-path-eval.npy'
        else:
            file_end = '-path-test.npy'

        # load specified numpy-files
        self.paths = paths
        prefix = "./data/3GPP_mimo/" + ant + "/"
        self.data_raw = np.load(prefix + 'scm3gpp_' + self.paths + file_end)

        # normalize data such that E[ ||h||^2 ] = N
        self.rx = self.data_raw.shape[1]
        self.data_raw *= np.sqrt(self.rx / np.mean(np.linalg.norm(self.data_raw, axis=1)**2))

        # revert vectorization
        self.data_raw = np.reshape(self.data_raw, (len(self.data_raw), int(ant[:2]), -1), order='F')

        # assign parameters
        self.snr = snr
        if isinstance(snr, int) or isinstance(snr, float):
            self.snr_ar = snr * np.ones(len(self.data_raw))
        else:
            self.snr_ar = self.rng.uniform(self.snr[0] - 9, self.snr[1] + 9, len(self.data_raw))
        self.train = train
        self.rx = self.data_raw.shape[1]
        self.tx = self.data_raw.shape[2]

        # create observation based on snr
        self.data_raw = np.reshape(self.data_raw, (-1, self.rx, self.tx), order='F')
        self.pilots = dft_matrix(self.tx).numpy().astype(complex)
        self.obs_mat = np.kron(self.pilots.T, np.eye(self.rx))
        self.Y, self.sigma = add_noise(self.data_raw, self.pilots, self.snr_ar, self.rng, get_sigmas=True)

        self.data = self.data_raw[:, np.newaxis, ...]
        self.data = np.concatenate([self.data.real, self.data.imag], axis=1)

    def __getitem__(self, index):

        Y = self.Y[index]
        H = self.data_raw[index]

        # use noisy observation as cond
        cond = np.copy(Y)

        # get noise level and label
        sigma = torch.tensor(self.sigma[index])

        # convert to tensors
        H_as_tensor = torch.tensor(H).to(torch.cfloat)
        cond_as_tensor = torch.tensor(cond).to(torch.cfloat)
        Y_as_tensor = torch.tensor(Y).to(torch.cfloat)

        return H_as_tensor, cond_as_tensor, sigma, Y_as_tensor, []

    def __len__(self):
        return len(self.data)

    def create_observations(self, snr=None):
        if snr is not None:
            self.snr_ar = snr * np.ones(len(self.data_raw))
        self.Y, self.sigma = add_noise(self.data_raw, self.pilots, self.snr_ar, self.rng, get_sigmas=True)


def add_noise(H, X, snr_dB, rng, get_sigmas=False):
    r"""
    For every MxN-dimensional channel Hi of H, scale complex standard normal noise such that we have
        SNR = 1 / σ^2
    and compute the corresponding
        x_i + standard_gauss * σ.
    """
    Y_nf = H @ X
    # SNR = E[ || h ||^2 ] / E[ || n ||^2 ] = N_tx / σ^2
    out_shape = [H.shape[0], H.shape[1], X.shape[1]]
    snr = 10 ** (snr_dB * 0.1)
    sigmas = 1 / np.sqrt(snr)
    sigmas = np.reshape(sigmas, (-1, 1, 1))
    if get_sigmas:
        return Y_nf + crandn(out_shape, rng) * sigmas, np.squeeze(sigmas)
    else:
        return Y_nf + crandn(out_shape, rng) * sigmas


def crandn(shape, rng):
    real, imag = rng.normal(0, 1/np.sqrt(2), shape), rng.normal(0, 1/np.sqrt(2), shape)
    return real + 1j * imag

