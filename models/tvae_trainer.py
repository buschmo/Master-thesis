# for calculations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np

# miscellaneous
from tqdm import tqdm
from pathlib import Path
import json

# own packages
from utils.trainer import Trainer
import utils.utilitites as utl
import utils.evaluation as evl

# logging results
from torch.utils.tensorboard import SummaryWriter

# define attributes and the label dimension their represented by
ATTRIBUTE_DIMENSIONS = {
    "Simplicity": 0
}


class TVAETrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_batch_data(self, batch):
        tokens, labels = batch

        src = tokens

        # Remove the last symbol in each target sentence
        tgt = tokens[:, :-1].clone()
        tgt[tgt == self.dataset.SEP] = self.dataset.PAD
        size = tgt.shape[-1]
        # Attention mask
        tgt_mask = torch.triu(torch.ones(size, size) *
                              float('-inf'), diagonal=1)

        # right-shift for later comparison
        tgt_true = tokens[:, 1:]

        # Mask [PAD] tokens
        src_key_padding_mask = (src == self.dataset.PAD)
        tgt_key_padding_mask = (tgt == self.dataset.PAD)
        
        # TODO no memory_key_padding? no src_masking?
        return (src.to("cuda"), tgt.to("cuda"), tgt_true.to("cuda"), tgt_mask.to("cuda"), src_key_padding_mask.to("cuda"), tgt_key_padding_mask.to("cuda"), labels.to("cuda"))

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        src, tgt, tgt_true, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, labels = batch

        prob, z_dist, prior_dist, z_tilde, z_prior = self.model(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(prob, tgt_true)

        # compute KLD loss
        dist_loss = self.compute_kld_loss(
            z_dist, prior_dist, beta=self.beta, c=self.capacity
        )

        # add losses
        loss = recons_loss + dist_loss

        # compute and add regularization loss if needed
        if self.use_reg_loss:
            reg_loss = 0.0
            if type(self.reg_dim) == tuple:
                for dim in self.reg_dim:
                    reg_loss += self.compute_reg_loss(
                        z_tilde, labels[:, dim], dim, gamma=self.gamma, factor=self.delta)
            else:
                raise TypeError(
                    "Regularization dimension must be a tuple of integers")
            loss += reg_loss

        # compute accuracy
        # TODO this needs to be changed
        accuracy = self.mean_accuracy(
            weights=torch.sigmoid(prob),
            targets=tgt_true
        )

        return loss, accuracy

    @staticmethod
    def reconstruction_loss(input, target):
        input = input.permute(0,2,1)
        return F.cross_entropy(input, target)

    @staticmethod
    def mean_accuracy(weights, targets):
        weights = torch.argmax(weights, dim=-1)
        acc = torch.sum(targets==weights) / torch.numel(targets)
        return acc

    def compute_representations(self, data_loader):
        latent_codes = []
        attr_values = []
        # for sample_id, batch in tqdm(enumerate(data_loader)):
        for sample_id, batch in enumerate(data_loader):
            batch_data =self.process_batch_data(batch)
            src, tgt, tgt_true, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, labels = batch_data
            labels = labels.to("cpu")
            _, _, _, z_tilde, _ = self.model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            latent_codes.append(utl.to_numpy(z_tilde))
            attr_values.append(labels)
            if sample_id == 200:
                # TODO how about the whole dataset?
                break
        # turn lists into matrices
        latent_codes = np.concatenate(latent_codes, 0)
        attr_values = np.concatenate(attr_values, 0)
        attr_list = ATTRIBUTE_DIMENSIONS.keys()
        return latent_codes, attr_values, attr_list
