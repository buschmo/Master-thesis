import torch
from torch import nn, distributions
from utils.base_model import BaseModel


class NaiveVAE(BaseModel):
    def __init__(self, input_size, z_dim=32, encoder_dim=128, decoder_dim=128, dropout=0.1, **kwargs):
        # set basic parameters
        self.input_size = input_size
        self.z_dim = z_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        # call super function afterwards
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.encoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.enc_mean = nn.Linear(self.encoder_dim, self.z_dim)
        self.enc_log_std = nn.Linear(self.encoder_dim, self.z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.decoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.decoder_dim, input_size),
            nn.Dropout(dropout)
        )

        self.init_weights()

    def __str__(self):
        return f"NaiveModel_{self.encoder_dim}_{self.z_dim}_{self.decoder_dim}"

    def encode(self, x):
        hidden = self.encoder(x)

        z_mean = self.enc_mean(hidden)
        z_log_std = self.enc_log_std(hidden)
        z_distribution = distributions.Normal(
            loc=z_mean, scale=torch.exp(z_log_std))

        return z_distribution

    def decode(self, z):
        hidden = self.decoder(z)
        return hidden

    def reparametrize(self, z_dist):
        z_tilde = z_dist.rsample()

        prior_dist = distributions.Normal(loc=torch.zeros_like(
            z_dist.loc), scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()
        return z_tilde, z_prior, prior_dist

    def forward(self, x):
        z_dist = self.encode(x)

        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        output = self.decode(z_tilde)

        return output, z_dist, prior_dist, z_tilde, z_prior
