import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class ThreeGPP(Dataset):
    def __init__(self, paths, ant, train=True, eval=True, seed=1234, dft=True, snr=None):
        r"""
        Input:
        :param paths (str)         : number of paths to consider (e.g. 1-5 for 1 and 5 paths)
        :param ant (str)           : antenna array (e.g. 32rx for 32x1 channels)
        :param train (bool)        : whether to use train dataset
        :param eval (bool)         : whether to use evaluation dataset
        :param seed (int)          : seed for random number (for reproducibility)
        :param dft (bool)          : bool to do dft
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
        prefix = "./data/3GPP/" + ant + "/"
        # try:
        #     os.mkdir("./data")
        # except OSError:
        #     print('Folder exists!')
        self.data_raw = np.load(prefix + 'scm3gpp_' + self.paths + file_end)

        # normalize data such that E[ ||h||^2 ] = N
        self.rx = self.data_raw.shape[1]
        self.data_raw *= np.sqrt(self.rx / np.mean(np.linalg.norm(self.data_raw, axis=1)**2))

        # assign parameters
        self.snr = snr
        if isinstance(snr, int) or isinstance(snr, float):
            self.snr_ar = snr * np.ones(len(self.data_raw))
        else:
            self.snr_ar = self.rng.uniform(self.snr[0] - 9, self.snr[1] + 9, len(self.data_raw))
        self.dft = dft
        self.train = train

        # create observation based on snr
        self.y, self.sigma = add_noise(self.data_raw, self.snr_ar, self.rng, get_sigmas=True)

        # reshape real and imaginary part into different channels
        if self.dft:
            self.data = np.fft.fft(self.data_raw, axis=1) / np.sqrt(self.data_raw.shape[1])
            self.y = np.fft.fft(self.y, axis=-1) / np.sqrt(self.data_raw.shape[1])
        else:
            self.data = self.data_raw

        self.data = self.data[:, np.newaxis, ...]
        self.data = np.concatenate([self.data.real, self.data.imag], axis=1)

    def __getitem__(self, index):

        y = self.y[index]
        h = self.data[index]

        # use noisy observation as condition
        cond = np.squeeze(np.copy(y))[np.newaxis, ...]
        cond = np.concatenate([cond.real, cond.imag], axis=0)

        # get noise level and label
        sigma = torch.tensor(self.sigma[index])

        # convert to tensors
        h_as_tensor = torch.tensor(h).to(torch.float)
        cond_as_tensor = torch.tensor(cond).to(torch.float)
        y_as_tensor = torch.tensor(y).to(torch.cfloat)

        return h_as_tensor, cond_as_tensor, sigma, y_as_tensor, []

    def __len__(self):
        return len(self.data)

    def create_observations(self, snr=None):
        if snr is not None:
            self.snr_ar = snr * np.ones(len(self.data_raw))
        self.y, self.sigma = add_noise(self.data_raw, self.snr_ar, self.rng, get_sigmas=True)
        if self.dft:
            self.y = np.fft.fft(self.y, axis=-1) / np.sqrt(self.y.shape[-1])


def add_noise(h, snr_dB, rng, get_sigmas=False):
    r"""
    For every MxN-dimensional channel Hi of H, scale complex standard normal noise such that we have
        SNR = 1 / σ^2
    and compute the corresponding
        x_i + standard_gauss * σ.
    """
    # SNR = E[ || h ||^2 ] / E[ || n ||^2 ] = 1 / σ^2
    out_shape = h.shape
    snr = 10 ** (snr_dB * 0.1)
    sigmas = 1 / np.sqrt(snr)
    sigmas = np.reshape(sigmas, (-1, 1))
    if get_sigmas:
        return h + crandn(out_shape, rng) * sigmas, np.squeeze(sigmas)
    else:
        return h + crandn(out_shape, rng) * sigmas


def crandn(shape, rng):
    real, imag = rng.normal(0, 1/np.sqrt(2), shape), rng.normal(0, 1/np.sqrt(2), shape)
    return real + 1j * imag
