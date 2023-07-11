# import packages
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.quadriga import Quadriga
from datasets.threegpp import ThreeGPP
from utils import rel_mse_np, compute_lmmse
from models import VAECircCov as VAECircCov
from models import VAECircCovReal as VAECircCovNoisy

# set simulation parameters
parser = argparse.ArgumentParser()
snr = 10  # SNR for the simulation, choose anything between -10 and 30 dB
samples_max = 3  # power of 2 samples to consider at max.
ant = '128rx'  # 32rx or 128rx (MIMO not included)
data = 2  # 1=Quadriga, 2=3GPP
paths = '3'  # for 3GPP data, represents number of propagation clusters
losmixed = 'mixed'  # use 'los' (LOS channels) or 'mixed' (mixed LOS/NLOS channels) if data==1 (Quadriga)
mu_first = 0 # use latent mean vector as first MC sample, if set to 1 the dotted lines from Fig.  is reproduced
path = './models/'
seed_train, seed_test = 479439743597, 2843084209824089
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# load training data
if data == 1:
    data_test = Quadriga(ant, train=False, eval=False, dft=True, seed=seed_test, snr=snr)
    f_save = './vaes/results-samples-quadriga-' + ant + '-' + str(snr) + 'dB-'
elif data == 2:
    data_test = ThreeGPP(paths, ant, train=False, eval=False, dft=True, seed=seed_test, snr=snr)
    f_save = './vaes/results-samples-3gpp-' + paths + 'p-' + ant + '-' + str(snr) + 'dB-'
else:
    raise ValueError('Choose valid data!')
rel_mse_global_cov, rel_mse_ls, rel_mse_genie, rel_mse_noisy, rel_mse_real, rel_mse_genie_cov = [], [], [], [], [], []

# reduce amount of samples if to large for RAM
r = 1000
data_test.data_raw = data_test.data_raw[:r]
data_test.data = data_test.data[:r]
data_test.y = data_test.y[:r]
data_test.sigma = data_test.sigma[:r]

# define genie VAE
if data == 1:
    cf_name = path + 'config-vae_circ_genie-quadriga-' + losmixed + '-' + ant + '.json'
else:
    cf_name = path + 'config-vae_circ_genie-3gpp-' + ant + '-' + paths + 'p.json'
with open(cf_name, "r") as f:
    cf = json.load(f)
kernel_szs = [cf['k'] for _ in range(cf['n_hid'])]
hidden_dims = []
ch_out = cf['ch']
for i in range(cf['n_hid']):
    hidden_dims.append(int(ch_out))
    ch_out *= cf['m']
input_size = data_test.rx
act = 'relu'
init = 'k_u'

vae_genie = VAECircCov(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                       hidden_dims=hidden_dims, input_size=input_size, use_iaf=0, n_blocks=0, hidden_iaf=0, act=act,
                       init=init, device=device, lambda_z=0.1, cond_as_input=0, input_dim=1).eval()
if data == 1:
    model_name = path + 'best-vae_circ-Quadriga-genie-quadriga-' + losmixed + '-' + ant + '.pt'
    vae_genie.load_state_dict(torch.load(model_name, map_location=device))
elif data == 2:
    model_name = path + 'best-vae_circ-ThreeGPP-genie-3gpp-' + ant + '-' + paths + 'p.pt'
    vae_genie.load_state_dict(torch.load(model_name, map_location=device))

# define VAE-noisy
if data == 1:
    cf_name = path + 'config-vae_circ_noisy-quadriga-' + losmixed + '-' + ant + '.json'
else:
    cf_name = path + 'config-vae_circ_noisy-3gpp-' + ant + '-' + paths + 'p.json'
with open(cf_name, "r") as f:
    cf = json.load(f)
hidden_dims.reverse()

vae_noisy = VAECircCov(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                       hidden_dims=hidden_dims, input_size=input_size, use_iaf=0, n_blocks=0, hidden_iaf=0, act=act,
                       init=init, device=device, lambda_z=0.1, cond_as_input=1, input_dim=1).eval()
if data == 1:
    model_name = path + 'best-vae_circ-Quadriga-noisy-quadriga-' + losmixed + '-' + ant + '.pt'
    vae_noisy.load_state_dict(torch.load(model_name, map_location=device))
elif data == 2:
    model_name = path + 'best-vae_circ-ThreeGPP-noisy-3gpp-' + ant + '-' + paths + 'p.pt'
    vae_noisy.load_state_dict(torch.load(model_name, map_location=device))

# define VAE-real
if data == 1:
    cf_name = path + 'config-vae_circ_noisy_real-quadriga-' + losmixed + '-' + ant + '.json'
else:
    cf_name = path + 'config-vae_circ_noisy_real-3gpp-' + ant + '-' + paths + 'p.json'
with open(cf_name, "r") as f:
    cf = json.load(f)
hidden_dims.reverse()

vae_real = VAECircCovNoisy(in_channels=2, stride=cf['st'], kernel_szs=kernel_szs, latent_dim=cf['latent_dim'],
                           hidden_dims=hidden_dims, input_size=input_size, use_iaf=0, n_blocks=0, hidden_iaf=0,
                           act=act, init=init, device=device, lambda_z=0.1, input_dim=1).eval()
if data == 1:
    model_name = path + 'best-vae_circ_noisy-Quadriga-real-quadriga-' + losmixed + '-' + ant + '.pt'
    vae_real.load_state_dict(torch.load(model_name, map_location=device))
elif data == 2:
    model_name = path + 'best-vae_circ_noisy-ThreeGPP-real-3gpp-' + ant + '-' + paths + 'p.pt'
    vae_real.load_state_dict(torch.load(model_name, map_location=device))

# transform observations back to non DFT domain
y = data_test.y.reshape((len(data_test), -1), order='F')
y = np.fft.ifft(y, axis=-1, norm='ortho')
sigma = data_test.sigma

# iterate over samples drawn from latent space
samples_ar = 2 ** np.arange(0, samples_max + 1)
h_true = data_test.data_raw.reshape((len(data_test), -1), order='F')
h_tensor = torch.tensor(data_test.data, device=device).to(torch.float).to(device)
with torch.no_grad():
    for n_samples in samples_ar:
        print('Simulating VAE estimators with %d samples.\n' % n_samples)

        # calculate channel estimates with VAE-genie
        y_tensor = torch.tensor(y, device=device).to(torch.cfloat)
        sigma_tensor = torch.tensor(sigma, device=device).to(torch.cfloat)
        mu_genie, C_h_genie = vae_genie.sample_estimator(n_samples, h_tensor, False, mu_first)
        h_genie = np.zeros_like(h_true, dtype=complex)
        for i in range(n_samples):
            h_est = compute_lmmse(C_h_genie[i], mu_genie[i], y_tensor, sigma_tensor, None, None, device)
            h_genie += (h_est / n_samples).cpu().numpy()
        rel_mse_genie.append(np.mean(rel_mse_np(h_true, h_genie)))
        del C_h_genie

        # calculate channel estimates with VAE-noisy
        y_tensor_in = torch.tensor(data_test.y, device=device).view(len(y), 1, -1).to(torch.cfloat)
        y_tensor_in = torch.cat([y_tensor_in.real, y_tensor_in.imag], dim=1).to(device)
        mu_noisy, C_h_noisy = vae_noisy.sample_estimator(n_samples, y_tensor_in, True, mu_first)
        h_noisy = np.zeros_like(h_true, dtype=complex)
        for i in range(n_samples):
            h_est = compute_lmmse(C_h_noisy[i], mu_noisy[i], y_tensor, sigma_tensor, None, None, device)
            h_noisy += (h_est / n_samples).cpu().numpy()
        rel_mse_noisy.append(np.mean(rel_mse_np(h_true, h_noisy)))
        del C_h_noisy

        # calculate channel estimates with VAE-real
        mu_real, C_h_real = vae_real.sample_estimator(n_samples, y_tensor_in, sigma_tensor, True, mu_first)
        h_real = np.zeros_like(h_true, dtype=complex)
        for i in range(n_samples):
            h_est = compute_lmmse(C_h_real[i], mu_real[i], y_tensor, sigma_tensor, None, None, device)
            h_real += (h_est / n_samples).cpu().numpy()
        rel_mse_real.append(np.mean(rel_mse_np(h_true, h_real)))
        del C_h_real

fig, ax = plt.subplots()
rel_imp_vae_genie = (rel_mse_genie[0] - rel_mse_genie) / rel_mse_genie[0]
res_vae_genie = np.array([samples_ar, rel_mse_genie, rel_imp_vae_genie]).T
print('VAE-genie:\n', res_vae_genie)
ax.plot(samples_ar, rel_mse_genie, '2-r', label='VAE-genie')

rel_imp_vae_noisy = (rel_mse_noisy[0] - rel_mse_noisy) / rel_mse_noisy[0]
res_vae_noisy = np.array([samples_ar, rel_mse_noisy, rel_imp_vae_noisy]).T
print('VAE-noisy:\n', res_vae_noisy)
ax.plot(samples_ar, rel_mse_noisy, '^-m', label='VAE-noisy')

rel_imp_vae_real = (rel_mse_real[0] - rel_mse_real) / rel_mse_real[0]
res_vae_real = np.array([samples_ar, rel_mse_real, rel_imp_vae_real]).T
print('VAE-real:\n', res_vae_real)
ax.plot(samples_ar, rel_mse_real, 'p-g', label='VAE-real')

ax.legend()
ax.set_xscale('log', base=2)
ax.set(title='sampling in latent space (SIMO signal model, ' + ant + ', ' + paths + ' cluster)',
       xlabel='samples in latent space', ylabel='normalized MSE')
plt.tight_layout()
ax.grid(True)
plt.show()
