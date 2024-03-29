# for calculations
import torch
from torch.nn import functional as F
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
    "Simplicity": 0,
    "Tree_depth": 1,
    "POS": 2,
    "Length": 3,
    "TF-IDF": 4
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
        seq_length = tgt.shape[-1]
        # Attention mask
        tgt_mask = torch.triu(torch.ones(
            seq_length, seq_length), diagonal=1).bool()
        memory_mask = torch.triu(torch.ones(
            seq_length, seq_length+1), diagonal=1).bool()

        # right-shift for later comparison
        tgt_true = tokens[:, 1:]

        # Mask [PAD] tokens
        src_key_padding_mask = (src == self.dataset.PAD)
        tgt_key_padding_mask = (tgt == self.dataset.PAD)

        # TODO no memory_key_padding? no src_masking?
        return (src.to("cuda"), tgt.to("cuda"), tgt_true.to("cuda"), tgt_mask.to("cuda"), memory_mask.to("cuda"), src_key_padding_mask.to("cuda"), tgt_key_padding_mask.to("cuda"), labels.to("cuda"))

    def loss_and_acc_for_batch(self, batch, train=True):
        src, tgt, tgt_true, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, labels = batch

        try:
            prob, z_dist, prior_dist, z_tilde, z_prior = self.model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
        except RuntimeError as err:
            raise err
        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(prob, tgt_true, self.alpha)

        # compute KLD loss
        dist_loss, kld = self.compute_kld_loss(
            z_dist, prior_dist, beta=self.beta, beta_kl=self.beta_kl, c=self.capacity
        )

        # add losses
        loss = recons_loss + dist_loss

        # compute and add regularization loss if needed
        if self.use_reg_loss:
            reg_loss = 0.0
            reg_loss_unscaled = 0.0
            if type(self.reg_dim) == tuple:
                for dim in self.reg_dim:
                    reg_loss_temp, reg_loss_unscaled_temp = self.compute_reg_loss(
                        z_tilde, labels[:, dim], dim, gamma=self.gamma, factor=self.delta)
                    reg_loss += reg_loss_temp
                    reg_loss_unscaled += reg_loss_unscaled_temp
            else:
                raise TypeError(
                    "Regularization dimension must be a tuple of integers")
            loss += reg_loss
        else:
            reg_loss = torch.Tensor([0])
            reg_loss_unscaled = torch.Tensor([0])

        # compute accuracy
        accuracy = self.mean_accuracy(
            weights=torch.sigmoid(prob),
            targets=tgt_true
        )

        loss_dict = {
            "sum": loss,
            "reconstruction": recons_loss,
            "KLD": dist_loss,
            "KLD_unscaled": kld,
            "regularization": reg_loss,
            "regularization_unscaled": reg_loss_unscaled
        }

        return loss_dict, accuracy

    @staticmethod
    def reconstruction_loss(input, target, alpha):
        input = input.permute(0, 2, 1)
        return alpha * F.cross_entropy(
            input=input,
            target=target,
            ignore_index=0  # ignore the padding label
        )

    @staticmethod
    def mean_accuracy(weights, targets):
        # get predicted label
        weights = torch.argmax(weights, dim=-1)
        # remove [PAD] label (== 0) from accuracy calculation
        mask = targets.ge(0.5)
        numerator = torch.sum(targets.masked_select(
            mask) == weights.masked_select(mask))
        denominator = len(targets.masked_select(mask))
        acc = numerator / denominator
        return acc

    def compute_representations(self, data_loader):
        latent_codes = []
        attr_values = []
        # for sample_id, batch in tqdm(enumerate(data_loader)):
        for sample_id, batch in enumerate(data_loader):
            batch_data = self.process_batch_data(batch)
            src, tgt, tgt_true, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, labels = batch_data
            labels = labels.to("cpu")  # for numpy conversion later on
            _, _, _, z_tilde, _ = self.model(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            latent_codes.append(utl.to_numpy(z_tilde))
            attr_values.append(labels)
            if sample_id >= np.ceil(len(data_loader) / 4):
                # TODO how about the whole dataset?
                break
        # turn lists into matrices
        latent_codes = np.concatenate(latent_codes, 0)
        attr_values = np.concatenate(attr_values, 0)
        attr_list = ATTRIBUTE_DIMENSIONS.keys()
        return latent_codes, attr_values, attr_list
