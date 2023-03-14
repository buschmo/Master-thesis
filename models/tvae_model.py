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
    def __init__(self, ntoken: int, d_model: int = 256, z_dim: int = 64, nhead_encoder: int = 8, nhead_decoder: int = 8, d_hid: int = 512, nlayers: int = 1, dropout: float = 0.1, use_gru=False, **kwargs):
        self.ntoken = ntoken
        self.d_model = d_model
        self.z_dim = z_dim
        self.nhead_encoder = nhead_encoder
        self.nhead_decoder = nhead_decoder
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.dropout = dropout
        super().__init__(**kwargs)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedder = Embedder(ntoken, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead_encoder,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=nlayers,
            # Turn of quirky optimization. leads to errors in evaluation
            enable_nested_tensor=False
        )

        # TODO two linear layers?
        # Putting variation into VAE
        self.enc_mean = nn.Linear(d_model, z_dim)
        self.enc_log_std = nn.Linear(d_model, z_dim)

        # Converting
        self.latent2hidden = nn.Linear(z_dim, d_model)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead_decoder,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=nlayers
        )

        # TODO implement GRU from TVAE
        if use_gru:
            raise NotImplementedError

        # TODO which one to use?
        # self.generator = Generator(d_model, ntoken)
        self.generator = nn.Linear(d_model, ntoken)

        # TODO rework initialization. Not necessary, as every module uses xavier/kaiming by default
        # self.init_weights()

    def __str__(self):
        return "TVAE"

    def encode(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        # [batch, sequence] -> [batch, sequence, d_model]
        x = self.embedder(x)
        x = self.pos_encoder(x)
        hidden = self.encoder(
            src=x,
            src_key_padding_mask=padding_mask
        )

        # Sentence level sampling
        # [batch, sequence, d_model] -> [batch, d_model]
        hidden_mean = torch.mean(hidden, dim=1)
        # [batch, d_model] -> [batch, z_dim]
        z_mean = self.enc_mean(hidden_mean)
        # [batch, d_model] -> [batch, z_dim]
        z_log_std = self.enc_log_std(hidden_mean)

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

    def forward(self, src: Tensor, tgt: Tensor, tgt_mask: Tensor, memory_mask: Tensor, src_key_padding_mask: Tensor, tgt_key_padding_mask: Tensor) -> Tensor:
        z_dist = self.encode(src, src_key_padding_mask)

        z_tilde, z_prior, prior_dist = self.reparametrize(z_dist)

        tgt = self.embedder(tgt)

        memory = self.latent2hidden(z_tilde)
        memory = memory.view(memory.shape[0], 1, memory.shape[1]).repeat(
            1, memory_mask.shape[1], 1)

        try:
            logits = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,  # is memory masking necessary for avg pooling?
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
        except RuntimeError as err:
            print(err)
            print(f"\tz_tilde.shape: {z_tilde.shape}")
            print(f"\tz_dist: {z_dist}")
            print(f"\tsrc={src.shape}")
            print(f"\ttgt={tgt.shape}")
            print(f"\tmemory={memory.shape}")
            print(f"\ttgt_mask={tgt_mask.shape}")
            print(f"\tmemory_mask={memory_mask.shape}")
            print(f"\ttgt_key_padding_mask={tgt_key_padding_mask.shape}")
            print(f"\tmemory_key_padding_mask={src_key_padding_mask.shape}")
            raise err
        logits = self.generator(logits)
        return logits, z_dist, prior_dist, z_tilde, z_prior


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
