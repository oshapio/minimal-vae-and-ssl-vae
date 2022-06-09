import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from matplotlib import cm
from tqdm.notebook import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class VariationalAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size=2,
        hiddens_encode_sizes=None,
        bottleneck_size=10,
        n_sample_z=3,
        output_activation=nn.Identity(),
        loss="mse",
        kl_term_weight=1.0,
    ):
        super(VariationalAutoEncoder, self).__init__()

        # hiddens_encode = [10] or hiddens_encode_sizes.clone()
        assert isinstance(
            hiddens_encode_sizes, list
        ), f"`hiddens_encode` needs to be a list but got type `{type(hiddens_encode_sizes)}`"

        encoder_layers = []
        last_size = input_size
        for layer_size in hiddens_encode_sizes:
            encoder_layers.append(nn.Linear(last_size, layer_size))
            encoder_layers.append(nn.ReLU())

            last_size = layer_size
        # Add the bottleneck
        encoder_layers.append(nn.Linear(last_size, bottleneck_size * 2))

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []

        last_size = bottleneck_size
        for layer_size in hiddens_encode_sizes[::-1]:
            decoder_layers.append(nn.Linear(last_size, layer_size))
            decoder_layers.append(nn.ReLU())

            last_size = layer_size
        decoder_layers.append(nn.Linear(last_size, input_size * 2))
        # decoder_layers.append(output_activation)

        self.output_activation = output_activation
        self.decoder = nn.Sequential(*decoder_layers)

        self.bottleneck_size = bottleneck_size
        self.input_size = input_size
        self.n_sample_z = n_sample_z
        self.kl_term_weight = kl_term_weight
        self.loss = loss

        self.bce_loss = nn.BCELoss(reduction="sum")

    def encode(self, x):
        B, _ = x.shape

        # Get the means and s.d.s for the approximate posterior network
        q_params = self.encoder(x)  # .view((B, self.bottleneck_size, 2))
        mu_z, log_sigma_z = q_params[..., : q_params.shape[-1] // 2], q_params[..., q_params.shape[-1] // 2 :]

        # Ensure non-negative variance
        sigma_z = torch.exp(log_sigma_z)

        # Sample from standard Gaussian n_sample_z times => (B, n_samples_z, bottleneck_dim)

        # Because we assume diagonal covariance, sample from B * bottleneck_size * n_sample_z univariate Gaussians

        sampled_eps_z = torch.normal(0.0, 1.0, size=(B, self.bottleneck_size))#.to(device)

        # B x n_sample_z x bottleneck_size
        sampled_z = sampled_eps_z * sigma_z + mu_z

        return mu_z, sigma_z, sampled_z

    def decode(self, x, sampled_z):
        B, _ = x.shape

        # (B * n_sample_z) x bottleneck_size
        permuted_flat_sampled_z = sampled_z.view(B, self.bottleneck_size)

        # h = F.relu(self.bottleneck_hidden(permuted_flat_sampled_z))

        p_likelihood_params = self.decoder(permuted_flat_sampled_z)  # .view(
        #     (B, self.input_size, 2)
        # )

        mu_x, log_sigma_x = (
            p_likelihood_params[..., : p_likelihood_params.shape[-1] // 2],
            p_likelihood_params[..., p_likelihood_params.shape[-1] // 2 :],
        )
        sigma_x = torch.exp(log_sigma_x)

        mu_x = self.output_activation(mu_x)
        sampled_eps_x = torch.normal(0.0, 1.0, size=(B, self.input_size))#.to(device)

        sampled_x = sampled_eps_x * sigma_x + mu_x

        return mu_x, sigma_x, sampled_x

    def forward(self, x):
        """
        x: (n_batches, datapoint_dim)

        returns reconstructed \hatx: (B, n_samples_z, n_samples_x, datapoint_dim)
        """
        B, _ = x.shape

        mu_z, sigma_z, sampled_z = self.encode(x)
        mu_x, sigma_x, sampled_x = self.decode(x, sampled_z)

        return {
            "encoded": sampled_z,
            "mu_z": mu_z,
            "sigma_z": sigma_z,
            "decoded": sampled_x,
            "mu_x": mu_x,
            "sigma_x": sigma_x,
        }

    def get_ELBO(self, input_x, forward_pass_output):
        """
        Returns the evidence lower bound E_q(z|x)[log p(x|z)] - D_KL(q||p)
        assuming Gaussians both in forward and backward models _per batch_
        """
        B, _ = input_x.shape

        negative_d_kl_term = 0.5 * torch.sum(
            1.0
            + torch.log(forward_pass_output["sigma_z"] ** 2)
            - forward_pass_output["mu_z"] ** 2
            - forward_pass_output["sigma_z"] ** 2
        )  # closed form, simple when diagonal covariance

        if self.loss == "mse":
            likelihood_under_appx_posterior = -torch.sum((input_x - forward_pass_output["mu_x"]) ** 2)

            # const_term = -self.input_size / 2 * np.log(2 * torch.pi)

            # log_variance_term = -torch.sum(torch.log(forward_pass_output["sigma_x"]))

            # quadratic_term = -torch.sum(
            #         # 0.5 *  (1.0 / (1.0 * forward_pass_output["sigma_x"] ** 2)) *
            #         0.5 * ((input_x - forward_pass_output["mu_x"]) ** 2)
            #         / (forward_pass_output["sigma_x"] ** 2)
            # )
            # likelihood_under_appx_posterior = (
            #     const_term +
            #     log_variance_term +
            #     quadratic_term
            # )  # log Gaussian is proprotional to MSE

        elif self.loss == "cont_bern":
            # likelihood_under_appx_posterior = -torch.mean(input_x* torch.log(forward_pass_output["mu_x"])  \
            # - (1 - input_x) * (1 - torch.log(forward_pass_output["mu_x"])), dim=-1)
            likelihood_under_appx_posterior = -self.bce_loss(forward_pass_output["mu_x"], input_x)

        ELBO = (
            likelihood_under_appx_posterior
            if self.kl_term_weight <= 1e-12
            else (self.kl_term_weight * negative_d_kl_term + likelihood_under_appx_posterior)
        )
        # ELBO = likelihood_under_appx_posterior
        return ELBO