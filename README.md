# Leveraging Variational Autoencoders for Parameterized MMSE Estimation

This is the simulation code for the article:

M. Baur, B. Fesl, and W. Utschick, "Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation," *arXiv preprint arXiv:2307.05352,* 2023.

## Abstract
In this manuscript, we propose to use a variational autoencoder-based framework for parameterizing a conditional linear minimum mean squared error estimator. The variational autoencoder models the underlying unknown data distribution as conditionally Gaussian, yielding the conditional first and second moments of the estimand, given a noisy observation. The derived estimator is shown to approximate the minimum mean squared error estimator by utilizing the variational autoencoder as a generative prior for the estimation problem. We propose three estimator variants that differ in their access to ground-truth data during the training and estimation phases. The proposed estimator variant trained solely on noisy observations is particularly noteworthy as it does not require access to ground-truth data during training or estimation. We conduct a rigorous analysis by bounding the difference between the proposed and the minimum mean squared error estimator, connecting the training objective and the resulting estimation performance. Furthermore, the resulting bound reveals that the proposed estimator entails a bias-variance tradeoff, which is well-known in the estimation literature. As an example application, we portray channel estimation, allowing for a structured covariance matrix parameterization and low-complexity implementation. Nevertheless, the proposed framework is not limited to channel estimation but can be applied to a broad class of estimation problems. Extensive numerical simulations first validate the theoretical analysis of the proposed variational autoencoder-based estimators and then demonstrate excellent estimation performance compared to related classical and machine learning-based state-of-the-art estimators.

## File Organization
Please download the data under this [link](https://syncandshare.lrz.de/getlink/fiRHpKeiMJ5hGTHPu8XuEF/data) by clicking on the _Download_ button. The password is `VAE-est-2023!`. Afterward, place the `data` folder in the same directory as the `datasets` and `models` folders.
The scripts for reproducing the paper results are `eval_baselines.py`, `eval_baselines_mimo.py`, and `eval_baselines_samples.py`. The remaining files contain auxiliary functions and classes. The folder `models` contains the pre-trained model weights with corresponding config files.

## Implementation Notes
This code is written in _Python_ version 3.8. It uses the deep learning library _PyTorch_ and the _numpy_, _scipy_, _matplotlib_, and _json_ packages. The code was tested with the versions visible in the requirements file.

Alternatively, a conda environment that meets the requirements can be created by executing the following lines:
```
conda create -n vae_est python=3.8 numpy=1.23.5 matplotlib=3.6.2 scipy=1.10.0 simplejson
conda activate vae_est  
conda install pytorch cpuonly -c pytorch
```

## Instructions
Run `eval_baselines.py` to reproduce the SIMO results from the paper. To this end, adapt the simulation parameters at the beginning of the file to select the desired scenario. Models are only available for the scenarios from the paper. Other scenario parameters will result in an error message.

Proceed analogously with `eval_baselines_mimo.py`, where the results for the MIMO signal model can be reproduced.

In `eval_baselines_samples.py`, the results for drawing different amounts of samples in the latent space can be investigated. Here, selecting any SIMO simulation scenario from the paper and an SNR value between -10 and 30 dB is possible to produce estimation results.

## License
This code is licensed under 3-clause BSD License:

>Copyright (c) 2023 M. Baur and B.Fesl.
>
>Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
>
>1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
>
>2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
>
>3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
>
>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
