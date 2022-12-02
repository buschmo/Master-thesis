# for calculations
import torch
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


class NaiveTrainer(Trainer):
    def __init__(self, **kwargs):
        super(**kwargs)

    def process_batch_data(self, batch):
        X, y = batch
        return (X.to("cuda"), y.to("cuda"))

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        # from image_vae_trainer
        # extract data
        inputs, labels = batch

        # perform forward pass of model
        outputs, z_dist, prior_dist, z_tilde, z_prior = self.model(inputs)

        # compute reconstruction loss
        recons_loss = self.reconstruction_loss(inputs, outputs)

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
        # TODO this might be incorrect
        accuracy = self.mean_accuracy(
            weights=torch.sigmoid(outputs),
            targets=inputs
        )

        return loss, accuracy

    # TODO staticmethod necessary? maybe move to other module?
    @staticmethod
    def reconstruction_loss(x, x_recons):
        # from image_vae_trainer
        batch_size = x.size(0)
        # x_recons = torch.sigmoid(x_recons)  # TODO sigmoid?
        recons_loss = torch.nn.functional.mse_loss(
            x_recons, x, reduction='sum'
        ).div(batch_size)
        return recons_loss

    # TODO rework to sentence usecase. Right now this is nonsense

    @staticmethod
    def mean_accuracy(weights, targets):
        # From image_vae_trainer
        predictions = torch.zeros_like(weights)
        predictions[weights >= 0.5] = 1
        binary_targets = torch.zeros_like(targets)
        binary_targets[targets >= 0.5] = 1
        correct = predictions == binary_targets
        acc = torch.sum(correct.float()) / binary_targets.view(-1).size(0)
        return acc

    def eval_model(self, data_loader, epoch_num=0):
        # From image_vae_trainer.compute_eval_metrics
        results_fp = self.model.filepath.with_stem(
            f"{self.model.filepath.stem}_{epoch_num}").with_suffix(".json")
        if results_fp.exists():
            with open(results_fp, 'r') as infile:
                self.metrics = json.load(infile)
        else:
            self.metrics = self.compute_eval_metrics(data_loader)

        if self.writer:
            self.writer.add_scalars("Disentanglement/Interpretability", {
                                    k: v[1] for k, v in self.metrics["Interpretability"].items()}, epoch_num)
            self.writer.add_scalar("Disentanglement/Mutual Information Gap",
                                   self.metrics["Mutual Information Gap"], epoch_num)
            self.writer.add_scalar("Disentanglement/Separated Attribute Predictability",
                                   self.metrics["Separated Attribute Predictability"], epoch_num)
            self.writer.add_scalar("Disentanglement/Spearman's Rank Correlation",
                                   self.metrics["Spearman's Rank Correlation"], epoch_num)
        else:
            if not results_fp.parent.exists():
                results_fp.parent.mkdir(parents=True)
            with open(results_fp, 'w') as outfile:
                json.dump(self.metrics, outfile, indent=2)
        return self.metrics

    def compute_eval_metrics(self, data_loader):
        latent_codes, attributes, attr_list = self.compute_representations(
            data_loader)
        interp_metrics = evl.compute_interpretability_metric(
            latent_codes, attributes, attr_list
        )
        metrics = {
            "Interpretability": interp_metrics
        }
        # self.metrics.update(evl.compute_modularity(latent_codes, attributes))
        metrics.update(evl.compute_mig(latent_codes, attributes))
        metrics.update(
            evl.compute_sap_score(latent_codes, attributes))
        metrics.update(
            evl.compute_correlation_score(latent_codes, attributes))
        # metrics.update(self.test_model(batch_size=batch_size))
        return metrics

    def compute_representations(self, data_loader):
        latent_codes = []
        attr_values = []
        for sample_id, (inputs, labels) in tqdm(enumerate(data_loader)):
            inputs = inputs.to("cuda")
            _, _, _, z_tilde, _ = self.model(inputs)
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

    # TODO necessary?
    def test_model(self, data_loader):
        # from image_vae_trainer
        _, _, gen_test = self.dataset.data_loaders(batch_size)
        mean_loss_test, mean_accuracy_test = self.loss_and_acc_test(gen_test)
        print('Test Epoch:')
        print(
            '\tTest Loss: ', mean_loss_test, '\n'
            '\tTest Accuracy: ', mean_accuracy_test * 100
        )
        return {
            "test_loss": mean_loss_test,
            "test_acc": mean_accuracy_test,
        }
