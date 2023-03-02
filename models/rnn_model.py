import torch
from torch import nn, distributions
from torch import Tensor
import math
from pathlib import Path

from utils.base_model import BaseModel

"""
Based on Wang et al (2019) - Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation
and Nangi et al (2021) - Counterfactuals to Control Latent Disentangled Text Representations for Style Transfer
"""


class TVAE(BaseModel):
    def __init__(self, ntoken: int, d_model: int = 256, z_dim: int = 64, nlayers: int = 1, dropout: float = 0.1, **kwargs):
        self.ntoken = ntoken
        self.d_model = d_model
        self.z_dim = z_dim
        self.nhead_encoder = nhead_encoder
        self.nhead_decoder = nhead_decoder
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        super().__init__(**kwargs)

        self.embedder = Embedder(ntoken, d_model)

        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=nlayers,
            dropout=dropout,
            batch_first=True
        )

        # Putting variation into VAE
        self.enc_mean = nn.Linear(d_model, z_dim)
        self.enc_log_std = nn.Linear(d_model, z_dim)

        # Converting
        self.latent2hidden = nn.Linear(z_dim, d_model)

        self.decoder = nn.GRU(
            input_size=d_model,
            hidden_size=ntoken,
            num_layers=nlayers,
            dropout=dropout,
            batch_first=True
        )

        self.generator = nn.Linear(d_model, ntoken)

        # TODO rework initialization
        self.init_weights()

    def __str__(self):
        return "RNN"

    def encode(self, x: Tensor) -> Tensor:
        # [batch, sequence] -> [batch, sequence, d_model]
        x = self.embedder(x)
        hidden, _ = self.encoder(x)

        # TODO this needs to be reconsidered
        # [batch, sequence, d_model] -> [batch, d_model]
        hidden_mean = torch.mean(hidden, dim=1)
        # [batch, d_model] -> [batch, z_dim]
        z_mean = self.enc_mean(hidden_mean)
        # [batch, d_model] -> [batch, z_dim]
        z_log_std = self.enc_log_std(hidden_mean)

        # # [batch, sequence, d_model] -> [batch, sequence, z_dim]
        # z_mean = self.enc_mean(hidden)
        # # [batch, sequence, d_model] -> [batch, sequence, z_dim]
        # z_log_std = self.enc_log_std(hidden)
        z_distribution = distributions.Normal(
            loc=z_mean, scale=torch.exp(z_log_std))

        return z_distribution

    def reparametrize(self, z_dist: distributions.Distribution) -> Tensor:
        z_tilde = z_dist.rsample()

        prior_dist = distributions.Normal(loc=torch.zeros_like(
            z_dist.loc), scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()
        return z_tilde, z_prior, prior_dist

    def forward(self, src: Tensor) -> Tensor:
        z_dist = self.encode(src)

        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        memory = self.latent2hidden(z_tilde)
        
        logits = self.decoder(memory)
        logits = self.generator(logits)
        return logits, z_dist, prior_dist, z_tilde, z_prior


class Embedder(nn.Module):
    def __init__(self, ntoken, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(ntoken, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


# TODO this needs to be called on the last layers output
class Generator(nn.Module):
    def __init__(self, d_model: int, ntoken: int):
        super().__init__()
        self.proj = nn.Linear(d_model, ntoken)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.proj(x), dim=-1)
