# Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation

This is the simulation code for the article:

M. Baur, B. Fesl, and W. Utschick, "Leveraging Variational Autoencoders for Parameterized MMSE Channel Estimation," *arXiv preprint arXiv:2307.05352,* 2023.

## Abstract
In this manuscript, we propose to utilize the generative neural network-based variational autoencoder for channel estimation. The variational autoencoder models the underlying true but unknown channel distribution as a conditional Gaussian distribution in a novel way. The derived channel estimator exploits the internal structure of the variational autoencoder to parameterize an approximation of the mean squared error optimal estimator resulting from the conditional Gaussian channel models. We provide a rigorous analysis under which conditions a variational autoencoder-based estimator is mean squared error optimal. We then present considerations that make the variational autoencoder-based estimator practical and propose three different estimator variants that differ in their access to channel knowledge during the training and evaluation phase. In particular, the proposed estimator variant trained solely on noisy pilot observations is particularly noteworthy as it does not require access to noise-free, ground-truth channel data during training or evaluation. Extensive numerical simulations first analyze the internal behavior of the variational autoencoder-based estimators and then demonstrate excellent channel estimation performance compared to related classical and machine learning-based state-of-the-art channel estimators. 

## File Organization
Please download the data under this [link](https://syncandshare.lrz.de/getlink/fiRHpKeiMJ5hGTHPu8XuEF/data). The password is `VAE-est-2023!`. Afterward, place the `data` folder in the same directory as the `datasets` and `models` folders.
The executable files for reproducing the paper results are `eval_baselines.py`, `eval_baselines_mimo.py`, and `eval_baselines_samples.py`. The remaining files contain auxiliary functions and classes. The folder `models` contains the pre-trained model weights with corresponding config files.

## Implementation Notes
This code is written in _Python_. It uses the deep learning library _PyTorch_ and the _numpy_, _scipy_, _matplotlib_, and _json_ packages. The code was tested with the versions visible in the requirements file.

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
