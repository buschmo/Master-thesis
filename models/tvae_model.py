import torch
from torch import nn, distributions
from torch import Tensor
from pathlib import Path

from utils.base_model import BaseModel

"""
Based on Wang et al (2019) - Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation
"""


class TVAE(BaseModel):
    def __init__(self, ntoken: int, d_model: int, nhead_encoder: int, nhead_decoder: int, d_hid: int, nlayers: int, dropout: float = 0.5, use_gru=False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(ntoken, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead_encoder, dim_feed_forward=d_hid, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layers=encoder_layers, num_layers=nlayers)

        # Putting variation into VAE
        self.enc_mean = nn.Linear(self.d_model, self.z_dim)
        self.enc_log_std = nn.Linear(self.d_model, self.z_dim)

        # Converting
        self.latent2hidden = nn.Linear(self.z_dim, self.d_model)

        # TODO more variables might be needed. See EncoderLayer above
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead_decoder, dim_feed_forward=d_hid, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, nlayers)

        # TODO implement GRU from TVAE
        if use_gru:
            raise NotImplementedError

        self.generator = Generator(d_model, ntoken)

        self.init_weights()

    def __str__(self):
        return f"TVAE_{self.d_model}_{self.z_dim}"

    def encode(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        hidden = self.encoder(x, mask)

        z_mean = self.enc_mean(hidden)
        z_log_std = self.enc_log_std(hidden)
        z_distribution = distributions.Normal(
            loc=z_mean, scale=torch.exp(z_log_std))

        return z_distribution

    # TODO missing arguments
    # TODO is logit correct?
    def decode(self, z: Tensor) -> Tensor:
        hidden = self.latent2hidden(z)
        logit = self.decoder(hidden) 
        return logit

    def reparametrize(self, z_dist: distributions.Distribution) -> Tensor:
        z_tilde = z_dist.rsample()

        prior_dist = distributions.Normal(loc=torch.zeros_like(
            z_dist.loc), scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()
        return z_tilde, z_prior, prior_dist

    # TODO rename prob to something more fitting
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        z_dist = self.encode(x, mask)

        z_tilde, z_prior, prior_dist = reparametrize(z_dist)

        logit = self.decode(z_tilde)
        prob = self.generator(logit)
        return prob


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# TODO forgot the plans for this class ._.
# TODO might be unnecessary
class Embedder(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, x: Tensor) -> Tensor:
        return


# TODO this needs to be called on the last layers output
class Generator(nn.Module):
    def __init__(self, d_model: int, ntoken: int):
        super.__init__()
        self.proj = nn.Linear(d_model, ntoken)

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def create_pad_mask(matrix:Tensor, pad_token:int) -> Tensor:
    return (matrix == pad_token)