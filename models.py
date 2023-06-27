import math
import torch
from utils import dft_matrix
import numpy as np
from torch import nn
from numpy import floor
from abc import abstractmethod
from typing import List, Any, TypeVar

Tensor = TypeVar('Tensor')
ndarray = TypeVar('ndarray')


def reparameterize(mu: Tensor, log_std: Tensor, **kwargs) -> Tensor:
    """
    Sample from std. gaussian and reparameterize with found parameters.
    :param mu: (Tensor) Mean of the latent Gaussian
    :param log_std: (Tensor) Standard deviation of the latent Gaussian
    :return:
    """
    B, M = mu.shape
    try:
        eps = torch.randn((B, M)).to(kwargs['device'])
    except KeyError:
        eps = torch.randn((B, M))
    std = torch.exp(log_std)
    return eps * std + mu, eps


def kl_div_diag_gauss(mu_p: Tensor, log_std_p: Tensor, mu_q: Tensor = None, log_std_q: Tensor = None):
    """Calculates the KL divergence between the two diagonal Gaussians p and q, i.e., KL(p||q)"""
    var_p = torch.exp(2 * log_std_p)
    if mu_q is not None:
        var_q = torch.exp(2 * log_std_q)
        kl_div = 0.5 * (torch.sum(2 * (log_std_q - log_std_p) + (mu_p - mu_q) ** 2 / (var_q + 1e-8)
                                  + var_p / var_q, dim=1) - mu_p.shape[1])
    else:
        # KL divergence for q=N(0,I)
        kl_div = 0.5 * (torch.sum(-2 * log_std_p + mu_p ** 2 + var_p, dim=1) - mu_p.shape[1])
    return kl_div


def get_activation(act_str):
    return {
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU(),
        'tanh': nn.Tanh()
    }[act_str]


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, data: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, latent_code: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Tensor, **kwargs) -> Tensor:
        pass


class VAECircCov(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 input_dim: int = 1,
                 **kwargs) -> None:
        super(VAECircCov, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(kwargs['lambda_z'], device=self.device)
        self.M = torch.tensor(self.input_size, device=self.device)

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 48]
        if kernel_szs is None:
            kernel_szs = [5 for _ in range(len(hidden_dims))]

        # calculate encoder output dims
        conv_out = []
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * self.pad - kernel_szs[i]) / self.stride + 1).astype(int)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size
            conv_out.append(tmp_size)
        self.hidden_dims = hidden_dims

        self.embed_data = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=1)

        # build encoder
        in_channels = hidden_dims[0]
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels,
                              out_channels=h_dim,
                              kernel_size=kernel_szs[i],
                              stride=self.stride,
                              padding=self.pad),
                    nn.BatchNorm1d(h_dim),
                    self.act)
            )
            in_channels = h_dim
        self.fc_mu_var = nn.Linear(self.pre_latent * hidden_dims[-1], 2 * self.latent_dim)

        self.encoder = nn.Sequential(*modules)

        # build decoder
        modules = []

        hidden_dims.reverse()
        kernel_szs.reverse()

        # calculate decoder output dims
        conv_out = []
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i]
            conv_out.append(self.pre_out)

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * self.pre_latent)

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3,
                                       kernel_size=kernel_szs[i],
                                       stride=self.stride,
                                       padding=self.pad),
                    nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * self.input_size)

        self.F = dft_matrix(self.input_size).to(self.device)

        self.apply(weights_init)

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        result = z
        jacobians = 0
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = self.decoder_input(result)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z, jacobians

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        if self.cond_as_input:
            encoder_input = kwargs['cond']
        else:
            encoder_input = data
        data = data[:, 0, :] + 1j * data[:, 1, :]

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real, mu_out_imag, log_prec = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        if not kwargs['train']:
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.complex64).to(self.device)
            C = self.F.conj().T @ c_diag @ self.F
            mu = mu_out @ self.F.conj()
        else:
            C, mu = None, None

        return [mu_out, data, log_prec, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, log_prec, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        rec_loss = torch.sum(log_prec - (log_prec.exp() * ((data - mu_out).abs() ** 2)), dim=1) - self.M * torch.log(
            self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}

    def sample_estimator(self, num_samples, obs, is_cond, mu_first):
        """
        Samples from the encoder distribution and returns the corresponding likelihoods in the latent space
        and the mean and covariance matrix at the decoder.
        :param num_samples: (Int) number of samples to draw
        :param obs: (Tensor) input data (observations or channels)
        :param is_cond: (Bool) specifies if obs is y, if not it is h
        :param mu_first: (Int) use mean value as first sample in MC sampling
        :return: (Tuple)
        """
        if is_cond:
            cond = obs
        else:
            cond = None
        mu_list, C_list, loglike_list, z_list = [], [], [], []
        for i in range(num_samples):
            if mu_first and (i == 0):
                args = self.forward(obs, cond=cond, train=False)
            else:
                args = self.forward(obs, cond=cond, train=True)
            mu_enc, log_std_enc = args[3], args[4]
            mu, log_prec, z = args[0], args[2], args[-4]
            z_list.append(z)
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.complex64).to(self.device)
            C_list.append(self.F.H @ c_diag @ self.F)
            mu_list.append(mu @ self.F.conj())
            log_var = torch.exp(log_std_enc)
            loglike_list.append(torch.sum(-log_var - torch.abs(z - mu_enc) ** 2 / log_var.exp()
                                          - torch.log(self.pi), dim=1).exp())
        log_like = torch.cat(loglike_list).view(-1, len(z))
        log_like /= torch.sum(log_like, 0, keepdim=True)
        log_like = torch.ones_like(log_like, device=self.device) / log_like.shape[0]
        return mu_list, C_list, log_like


class VAECircCovReal(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size: int = 16,
                 input_dim: int = 1,
                 **kwargs) -> None:
        super(VAECircCovReal, self).__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 1
        self.stride = stride
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(kwargs['lambda_z'], device=self.device)
        self.M = torch.tensor(self.input_size, device=self.device)

        self.embed_data = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 48]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [5 for _ in range(len(hidden_dims))]

        # calculate encoder output dims
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * self.pad - kernel_szs[i]) / self.stride + 1).astype(int)
            if tmp_size < 1:
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        in_channels = hidden_dims[0]
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels,
                              out_channels=h_dim,
                              kernel_size=kernel_szs[i],
                              stride=self.stride,
                              padding=self.pad),
                    nn.BatchNorm1d(h_dim),
                    self.act)
            )
            in_channels = h_dim
        self.fc_mu_var = nn.Linear(self.pre_latent * hidden_dims[-1], 2 * self.latent_dim)

        self.encoder = nn.Sequential(*modules)

        # build decoder
        modules = []

        hidden_dims.reverse()
        kernel_szs.reverse()

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * self.pre_latent)

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3,
                                       kernel_size=kernel_szs[i],
                                       stride=self.stride,
                                       padding=self.pad),
                    nn.BatchNorm1d(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + kernel_szs[i]

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * self.input_size)

        self.F = dft_matrix(self.input_size).to(self.device)

        self.apply(weights_init)

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        result = z
        jacobians = 0
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = self.decoder_input(result)
        result = result.view(len(result), -1, self.pre_latent)
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z, jacobians

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:
        encoder_input = kwargs['cond']
        data = data[:, 0, :] + 1j * data[:, 1, :]
        cond = kwargs['cond'][:, 0, :] + 1j * kwargs['cond'][:, 1, :]

        mu_enc, log_std_enc = self.encode(encoder_input)

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real, mu_out_imag, log_var = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        if not kwargs['train']:
            sigma = kwargs['sigma'].unsqueeze(-1)
            var_h = torch.exp(log_var)
            var_y = var_h + (sigma ** 2)
            c_h_diag = torch.diag_embed(var_h).type(torch.cfloat).to(self.device)
            C_h = self.F.conj().T @ c_h_diag @ self.F
            c_y_diag = torch.diag_embed(var_y).type(torch.cfloat).to(self.device)
            C_y = self.F.conj().T @ c_y_diag @ self.F
            C = (C_h, C_y)
            mu = mu_out @ self.F.conj()
        else:
            C, mu = None, None

        return [mu_out, data, cond, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, cond, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        sigma = kwargs['sigma'].unsqueeze(-1)
        var = log_var.exp() + (sigma ** 2)

        rec_loss = torch.sum(-torch.log(var) - (((cond - mu_out).abs() ** 2) / var), dim=1) - self.M * torch.log(
            self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}

    def sample_estimator(self, num_samples, obs, sigma, is_cond, mu_first):
        """
        Samples from the encoder distribution and returns the corresponding likelihoods in the latent space
        and the mean and covariance matrix at the decoder.
        :param num_samples: (Int) number of samples to draw
        :param obs: (Tensor) input data (observations or channels)
        :param is_cond: (Bool) specifies if obs is y, if not it is h
        :param mu_first: (Int) use mean value as first sample in MC sampling
        :return: (Tuple)
        """
        if is_cond:
            cond = obs
        else:
            cond = None
        mu_list, C_list, loglike_list, z_list = [], [], [], []
        for i in range(num_samples):
            if mu_first and (i == 0):
                args = self.forward(obs, cond=cond, sigma=sigma, train=False)
            else:
                args = self.forward(obs, cond=cond, sigma=sigma, train=True)
            mu_enc, log_std_enc = args[4], args[5]
            mu, log_var, z = args[0], args[3], args[-4]
            z_list.append(z)
            c_diag = torch.diag_embed(torch.exp(log_var)).type(torch.complex64).to(self.device)
            C_list.append(self.F.H @ c_diag @ self.F)
            mu_list.append(mu @ self.F.conj())
            log_var_enc = torch.exp(log_std_enc)
            loglike_list.append(
                torch.sum(-log_var_enc - torch.abs(z - mu_enc) ** 2 / log_var_enc.exp() - torch.log(self.pi),
                          dim=1).exp())
        log_like = torch.cat(loglike_list).view(-1, len(z))
        log_like /= torch.sum(log_like, 0, keepdim=True)
        log_like = torch.ones_like(log_like, device=self.device) / log_like.shape[0]
        return mu_list, C_list, log_like


class VAECircCovMIMO(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size=None,
                 input_dim: int = 2,
                 **kwargs) -> None:
        super(VAECircCovMIMO, self).__init__()

        if input_size is None:
            input_size = [32, 4]
        self.latent_dim = latent_dim
        self.input_size = np.array(input_size)  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 2
        self.stride = stride
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.cond_as_input = kwargs['cond_as_input']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(np.prod(self.input_size), device=self.device)

        self.embed_data = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [8, 16, 32, 64]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [3 for _ in range(len(hidden_dims))]

        # calculate encoder output dims
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * self.pad - np.array(kernel_szs[i])) / self.stride + 1).astype(int)
            if (tmp_size < 1).any():
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        in_channels = hidden_dims[0]
        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=h_dim,
                              kernel_size=kernel_szs[i],
                              stride=self.stride,
                              padding=self.pad),
                    nn.BatchNorm2d(h_dim),
                    self.act)
            )
            in_channels = h_dim
        self.fc_mu_var = nn.Linear(np.prod(self.pre_latent) * hidden_dims[-1], 2 * self.latent_dim)

        self.encoder = nn.Sequential(*modules)

        # build decoder
        modules = []

        hidden_dims.reverse()
        kernel_szs.reverse()

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * np.prod(self.pre_latent))

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3,
                                       kernel_size=kernel_szs[i],
                                       stride=self.stride,
                                       padding=self.pad),
                    nn.BatchNorm2d(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + np.array(kernel_szs[i])
        self.pre_out = np.prod(self.pre_out)

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * np.prod(self.input_size))

        self.Q = torch.kron(dft_matrix(self.input_size[1]), dft_matrix(self.input_size[0])).to(self.device)

        self.X = dft_matrix(self.input_size[1]).to(self.device)
        self.X_inv = self.X.H

        self.apply(weights_init)

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N_rx x N_tx]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        jacobians = 0
        result = self.decoder_input(z)
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = result.view(len(result), -1, self.pre_latent[0], self.pre_latent[1])
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z, jacobians

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:

        if self.cond_as_input:
            enc_in_raw = (kwargs['cond'] @ self.X_inv)
        else:
            enc_in_raw = data

        # transform input with modal matrix
        data_trans = data.transpose(-1, -2).flatten(1) @ self.Q.T
        enc_in_trans = enc_in_raw.transpose(-1, -2).flatten(1) @ self.Q.T
        enc_in = enc_in_trans.view((-1, 1, data.shape[-1], data.shape[-2])).transpose(-1, -2)

        enc_in = torch.cat([enc_in.real, enc_in.imag], dim=1)

        mu_enc, log_std_enc = self.encode(enc_in)

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real, mu_out_imag, log_prec = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        if not kwargs['train']:
            c_diag = torch.diag_embed(torch.exp(-log_prec)).type(torch.cfloat).to(self.device)
            C = self.Q.H @ c_diag @ self.Q
            mu = mu_out @ self.Q.conj()
        else:
            C, mu = None, None

        return [mu_out, data, data_trans, log_prec, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, data_trans, log_prec, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        rec_loss = torch.sum(log_prec - (log_prec.exp() * ((data_trans - mu_out).abs() ** 2)), dim=1) \
                   - self.M * torch.log(self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


class VAECircCovMIMOReal(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 stride: int,
                 kernel_szs: List,
                 latent_dim: int,
                 hidden_dims: List = None,
                 input_size=None,
                 input_dim: int = 2,
                 **kwargs) -> None:
        super(VAECircCovMIMOReal, self).__init__()

        if input_size is None:
            input_size = [32, 4]
        self.latent_dim = latent_dim
        self.input_size = np.array(input_size)  # this refers to the data size x
        self.input_dim = input_dim  # this refers to the actual input dimensionality of x or cond
        self.in_channels = in_channels
        self.pad = 2
        self.stride = stride
        self.act = get_activation(kwargs['act'])
        self.device = kwargs['device']
        self.pi = torch.tensor(math.pi).to(self.device)
        self.lambda_z = torch.tensor(0.1, device=self.device)
        self.M = torch.tensor(np.prod(self.input_size), device=self.device)

        self.embed_data = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 48]
        self.hidden_dims = hidden_dims
        if kernel_szs is None:
            kernel_szs = [[5, 2] for _ in range(len(hidden_dims))]

        # calculate encoder output dims
        tmp_size = self.input_size
        for i in range(len(hidden_dims)):
            tmp_size = floor((tmp_size + 2 * self.pad - np.array(kernel_szs[i])) / self.stride + 1).astype(int)
            if (tmp_size < 1).any():
                hidden_dims = hidden_dims[:i]
                kernel_szs = kernel_szs[:i]
                break
            self.pre_latent = tmp_size

        # build encoder
        in_channels = hidden_dims[0]

        for (i, h_dim) in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=h_dim,
                              kernel_size=kernel_szs[i],
                              stride=self.stride,
                              padding=self.pad),
                    nn.BatchNorm2d(h_dim),
                    self.act)
            )
            in_channels = h_dim
        self.fc_mu_var = nn.Linear(np.prod(self.pre_latent) * hidden_dims[-1], 2 * self.latent_dim)

        self.encoder = nn.Sequential(*modules)

        # build decoder
        modules = []

        hidden_dims.reverse()
        kernel_szs.reverse()

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[0] * np.prod(self.pre_latent))

        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3,
                                       kernel_size=kernel_szs[i],
                                       stride=self.stride,
                                       padding=self.pad),
                    nn.BatchNorm2d(hidden_dims[i + 1] if i < len(hidden_dims) - 1 else 3),
                    self.act)
            )

        self.decoder = nn.Sequential(*modules)

        # calculate decoder output dims
        self.pre_out = self.pre_latent
        for i in range(len(hidden_dims)):
            self.pre_out = (self.pre_out - 1) * self.stride - 2 * self.pad + np.array(kernel_szs[i])
        self.pre_out = np.prod(self.pre_out)

        self.final_layer = nn.Linear(3 * self.pre_out, 3 * np.prod(self.input_size))

        self.Q = torch.kron(dft_matrix(self.input_size[1]), dft_matrix(self.input_size[0])).to(self.device)

        self.X = dft_matrix(self.input_size[1]).to(self.device)
        self.X_inv = self.X.H

        self.apply(weights_init)

    def encode(self, data: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param data: (Tensor) Input tensor to encoder [B x 2 x N_rx x N_tx]
        :return: (Tensor) List with mean and variance vector assuming a diagonal gaussian posterior
        """
        result = self.embed_data(data)
        result = self.encoder(result).flatten(start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        [mu, log_std] = self.fc_mu_var(result).chunk(2, dim=1)
        return [mu, log_std]

    def decode(self, z: Tensor, **kwargs) -> Tensor:
        jacobians = 0
        result = self.decoder_input(z)
        # shape result to fit dimensions after all conv layers of encoder and decode
        result = result.view(len(result), -1, self.pre_latent[0], self.pre_latent[1])
        result = self.decoder(result)
        # flatten result and put into final layer to predict [mu_real, mu_imag, c_diag]
        result = torch.flatten(result, start_dim=1)
        result = self.final_layer(result)
        return result, z, jacobians

    def forward(self, data: Tensor, **kwargs) -> List[Tensor]:

        # transform input with inverse of pilot matrix
        cond = kwargs['cond']
        cond_decor = (cond @ self.X_inv)

        # transform input with modal matrix
        cond_decor_trans = cond_decor.transpose(-1, -2).flatten(1) @ self.Q.T
        enc_in = torch.clone(cond_decor_trans)
        enc_in = enc_in.view((-1, 1, data.shape[-1], data.shape[-2])).transpose(-1, -2)

        enc_in = torch.cat([enc_in.real, enc_in.imag], dim=1)

        mu_enc, log_std_enc = self.encode(enc_in)

        if kwargs['train']:
            z_0, eps = reparameterize(mu_enc, log_std_enc, device=self.device)
        else:
            z_0, eps = mu_enc, torch.zeros_like(mu_enc).to(self.device)

        out, z, jacobians = self.decode(z_0)
        mu_out_real, mu_out_imag, log_var = out.chunk(3, dim=1)
        mu_out = mu_out_real + 1j * mu_out_imag

        if not kwargs['train']:
            var_h = torch.exp(log_var)
            c_h_diag = torch.diag_embed(var_h).type(torch.complex64).to(self.device)
            C_h = self.Q.conj().T @ c_h_diag @ self.Q
            C_y = None
            C = (C_h, C_y)
            mu = mu_out @ self.Q.conj()
        else:
            C, mu = None, None

        return [mu_out, data, cond_decor_trans, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C]

    def loss_function(self, *args, **kwargs) -> dict:
        mu_out, data, cond_decor_trans, log_var, mu_enc, log_std_enc, z_0, z, jacobians, mu, C = args

        sigma = kwargs['sigma'].unsqueeze(1)
        var = log_var.exp() + (sigma ** 2)

        rec_loss = torch.sum(-torch.log(var) - (((cond_decor_trans - mu_out).abs() ** 2) / var), dim=1) \
            - self.M * torch.log(self.pi)

        kld_loss = kl_div_diag_gauss(mu_enc, log_std_enc)

        loss = rec_loss - kwargs['alpha'] * kld_loss
        loss_back = rec_loss.mean() - kwargs['alpha'] * torch.maximum(kld_loss.mean(), self.lambda_z)

        return {'loss': loss, 'loss_back': loss_back, 'rec_loss': rec_loss, 'KLD': kld_loss}


def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) \
            or isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.normal_(0.0, 0.05)
