import torch
from torch import nn, distributions
from code.base_model import BaseModel

class NaiveVAE(BaseModel):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        self.z_dim = 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU()
        )

        self.enc_mean = nn.Linear(256, self.z_dim)
        self.enc_log_std = nn.Linear(256, self.z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.ReLU()
        )

    def encode(self, x):
        hidden = self.encoder(x)

        z_mean = self.enc_mean(hidden)
        z_log_std = self.enc_log_std(hidden)
        z_distribution = distributions.Normal(
            loc=z_mean, scale=torch.exp(z_log_std))

        return z_distribution

    def decode(self, z):
        hidden = self.decoder(z)

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
    
    def __repr__(self):
        return self.state_dict()

    def __str__(self):
        return "NaiveModel"