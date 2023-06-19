# import packages
import json
import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
from datasets.quadrigaMIMO import QuadrigaMIMO
from datasets.threegppMIMO import ThreeGPPMIMO
from utils import rel_mse_np, compute_lmmse
from models import VAECircCovMIMO as VAECircCovMIMO
from models import VAECircCovMIMOReal as VAECircCovMIMONoisy


# set simulation parameters
ant = '32rx4tx'
path = './models/'
path_t = 'C:\\Users\\MichaelBaur\\LRZ Sync+Share\\Cloud\\Promotion\\code\\vaes\\data\\3GPP_mimo\\' + ant + '\\'
data = 2  # 1=Quadriga, 2=3GPP
paths = '3'  # for 3GPP data
losmixed = 'mixed'
seed_train, seed_test = 479439743597, 2843084209824089
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
snr_lim = [-10, 30]
snr_step = 10


# load training data
snr_ar = np.arange(snr_lim[0], snr_lim[1] + 1, snr_step)
if data == 1:
    data_train = QuadrigaMIMO(ant, train=True, snr=10, seed=seed_train)
    data_test = QuadrigaMIMO(ant, train=False, eval=False, seed=seed_test, snr=0)
elif data == 2:
    data_train = ThreeGPPMIMO(ant, train=True, paths=paths, snr=10, seed=seed_train)
    data_test = ThreeGPPMIMO(ant, train=False, eval=False, paths=paths, seed=seed_test, snr=0)
else:
    raise ValueError('Choose valid data!')
rel_mse_global_cov, rel_mse_ls, rel_mse_genie, rel_mse_noisy, rel_mse_real, rel_mse_genie_cov = [], [], [], [], [], []


# define global cov estimator and LS
H_train = data_train.data_raw.reshape((len(data_train), -1), order='F')
C_global = 1 / len(H_train) * H_train.T @ H_train.conj()
A = data_train.obs_mat
A_pinv = np.linalg.solve(A.conj().T @ A, A.conj().T)  # pseudo-inverse for LS
A_tensor = torch.tensor(A, device=device).to(torch.cfloat)


# define genie-cov estimator if 3GPP data is used
if data == 2:
    t_BS = np.load(path_t + 'scm3gpp_' + paths + '-path-cov-bs-test.npy')
    t_MS = np.load(path_t + 'scm3gpp_' + paths + '-path-cov-ms-test.npy')
    cov = [np.kron(sp.linalg.toeplitz(t_MS[i]), sp.linalg.toeplitz(t_BS[i])) for i in range(len(t_BS))]
    cov = torch.tensor(np.array(cov), device=device)
    mu = torch.zeros((len(cov), cov.shape[-1]), device=device).to(torch.cfloat)


# define genie VAE
if data == 1:
    cf_name = path + 'config-vae_circ_mimo_genie-quadriga-' + losmixed + '-' + ant + '.json'
else:
    cf_name = path + 'config-vae_circ_mimo_genie-3gpp-' + ant + '-' + paths + 'p.json'
with open(cf_name, "r") as f:
    cf = json.load(f)
kernel_szs = [[cf['k'], 2] for _ in range(cf['n_hid'])]
hidden_dims = []
ch_out = cf['ch']
for i in range(cf['n_hid']):
    hidden_dims.append(int(ch_out))
    ch_out *= cf['m']
act = cf['act']
input_size = [data_train.rx, data_train.tx]
vae_genie = VAECircCovMIMO(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                           hidden_dims=hidden_dims, input_size=input_size, act=act, device=device, lambda_z=0.1,
                           cond_as_input=0, input_dim=2).eval()
if data == 1:
    model_name = path + 'best-vae_circ_mimo-quad_mimo-genie-quadriga-' + losmixed + '-' + ant + '.pt'
    vae_genie.load_state_dict(torch.load(model_name, map_location=device))
elif data == 2:
    model_name = path + 'best-vae_circ_mimo-3GPP_MIMO-genie-3gpp-' + ant + '-' + paths + 'p.pt'
    vae_genie.load_state_dict(torch.load(model_name, map_location=device))


# define VAE-noisy
hidden_dims.reverse()
vae_noisy = VAECircCovMIMO(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                           hidden_dims=hidden_dims, input_size=input_size, act=act, device=device,
                           lambda_z=0.1, cond_as_input=1, input_dim=2).eval()
if data == 1:
    model_name = path + 'best-vae_circ_mimo-quad_mimo-noisy-quadriga-' + losmixed + '-' + ant + '.pt'
    vae_noisy.load_state_dict(torch.load(model_name, map_location=device))
elif data == 2:
    model_name = path + 'best-vae_circ_mimo-3GPP_MIMO-noisy-3gpp-' + ant + '-' + paths + 'p.pt'
    vae_noisy.load_state_dict(torch.load(model_name, map_location=device))


# define VAE-real
hidden_dims.reverse()
vae_real = VAECircCovMIMONoisy(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                               hidden_dims=hidden_dims, input_size=input_size, act=act, device=device, lambda_z=0.1,
                               input_dim=2).eval()
if data == 1:
    model_name = path + 'best-vae_circ_noisy_mimo-quad_mimo-real-quadriga-' + losmixed + '-' + ant + '.pt'
    vae_real.load_state_dict(torch.load(model_name, map_location=device))
elif data == 2:
    model_name = path + 'best-vae_circ_noisy_mimo-3GPP_MIMO-real-3gpp-' + ant + '-' + paths + 'p.pt'
    vae_real.load_state_dict(torch.load(model_name, map_location=device))


# iterate over SNR and evaluate every model
h_true = data_test.data_raw.reshape((len(data_test), -1), order='F')
H_tensor = torch.tensor(data_test.data_raw, device=device).to(torch.cfloat)
with torch.no_grad():
    for snr in snr_ar:
        print('Simulating SNR of %d dB.\n' % snr)

        # create new observations for current SNR
        data_test.create_observations(snr)
        y = data_test.Y.reshape((len(data_test), -1), order='F')
        sigma = data_test.sigma

        # calculate channel estimates with global cov
        h_global_cov = np.zeros_like(h_true, dtype=complex)
        for i in range(len(h_true)):
            C_y = A @ C_global @ A.conj().T + (sigma[i] ** 2) * np.eye(A.shape[0], dtype=complex)
            h_global_cov[i] = C_global @ A.conj().T @ np.linalg.solve(C_y, y[i])
        rel_mse_global_cov.append(np.mean(rel_mse_np(h_true, h_global_cov)))

        # calculate channel estimates with LS
        h_ls = y @ A_pinv.T
        rel_mse_ls.append(np.mean(rel_mse_np(h_true, h_ls)))

        # calculate channel estimates with VAE-genie
        y_tensor = torch.tensor(y, device=device).to(torch.cfloat)
        sigma_tensor = torch.tensor(sigma, device=device).to(torch.cfloat)
        args_vae_genie = vae_genie(H_tensor, train=False)
        mu_genie, C_h_genie = args_vae_genie[-2], args_vae_genie[-1]
        h_genie = compute_lmmse(C_h_genie, mu_genie, y_tensor, sigma_tensor, None, A_tensor, device).numpy()
        rel_mse_genie.append(np.mean(rel_mse_np(h_true, h_genie)))

        # calculate channel estimates with VAE-noisy
        Y_tensor = torch.tensor(data_test.Y, device=device).to(torch.cfloat)
        args_vae_noisy = vae_noisy(H_tensor, cond=Y_tensor, train=False)
        mu_noisy, C_h_noisy = args_vae_noisy[-2], args_vae_noisy[-1]
        h_noisy = compute_lmmse(C_h_noisy, mu_noisy, y_tensor, sigma_tensor, None, A_tensor, device).numpy()
        rel_mse_noisy.append(np.mean(rel_mse_np(h_true, h_noisy)))

        # calculate channel estimates with VAE-real
        args_vae_real = vae_real(H_tensor, cond=Y_tensor, sigma=sigma_tensor, train=False)
        mu_real, C_h_real = args_vae_real[-2], args_vae_real[-1][0]
        h_real = compute_lmmse(C_h_real, mu_real, y_tensor, sigma_tensor, None, A_tensor, device).numpy()
        rel_mse_real.append(np.mean(rel_mse_np(h_true, h_real)))

        # calculate channel estimates with genie-cov
        if data == 2:
            h_genie_cov = compute_lmmse(cov, mu, y_tensor, sigma_tensor, None, A_tensor, device).numpy()
            rel_mse_genie_cov.append(np.mean(rel_mse_np(h_true, h_genie_cov)))


res_global_cov = np.array([snr_ar, rel_mse_global_cov]).T
print('global-cov:\n', res_global_cov)
plt.semilogy(snr_ar, rel_mse_global_cov, '|--c', label='global-cov')

res_ls = np.array([snr_ar, rel_mse_ls]).T
print('LS:\n', res_ls)
plt.semilogy(snr_ar, rel_mse_ls, '1--k', label='LS')

res_vae_genie = np.array([snr_ar, rel_mse_genie]).T
print('VAE-genie:\n', res_vae_genie)
plt.semilogy(snr_ar, rel_mse_genie, '2-r', label='VAE-genie')

res_vae_noisy = np.array([snr_ar, rel_mse_noisy]).T
print('VAE-noisy:\n', res_vae_noisy)
plt.semilogy(snr_ar, rel_mse_noisy, '^-m', label='VAE-noisy')

res_vae_real = np.array([snr_ar, rel_mse_real]).T
print('VAE-real:\n', res_vae_real)
plt.semilogy(snr_ar, rel_mse_real, 'p-g', label='VAE-real')

if data == 2:
    res_genie_cov = np.array([snr_ar, rel_mse_genie_cov]).T
    print('genie-cov:\n', res_genie_cov)
    plt.semilogy(snr_ar, rel_mse_genie_cov, 'x--b', label='genie-cov')

plt.legend()
plt.title('MIMO signal model with ' + ant + ' and ' + paths + ' cluster')
plt.xlabel('SNR [dB]')
plt.ylabel('normalized MSE')
plt.tight_layout()
plt.grid(True)
plt.show()
