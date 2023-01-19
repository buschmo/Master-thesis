# for calculations
import torch
from torch.utils.data import DataLoader
import numpy as np

# miscellaneous
from tqdm import tqdm
from pathlib import Path
import json

# own packages
import utils.utilitites as utl
import utils.evaluation as evl

# logging results
from torch.utils.tensorboard import SummaryWriter

# define attributes and the label dimension their represented by
ATTRIBUTE_DIMENSIONS = {
    "Simplicity": 0
}


class Trainer():
    def __init__(self, dataset, model, checkpoint_index=0, lr=1e-4, beta=4.0, gamma=10.0, capacity=0.0, delta=1.0, use_reg_loss=True, folderpath=""):
        # from trainer
        self.writer = SummaryWriter(log_dir=Path("runs", folderpath))

        self.dataset = dataset
        self.model = model

        if torch.cuda.is_available():
            self.model.cuda()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )

        self.checkpoint_index = checkpoint_index
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.capacity = capacity
        self.use_reg_loss = use_reg_loss
        self.reg_dim = (0,)

    def train_model(self, batch_size, num_epochs):
        # from trainer
        dataset_train, dataset_val = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2]
        )
        generator_train = DataLoader(dataset_train, batch_size=batch_size)
        generator_val = DataLoader(dataset_val, batch_size=batch_size)

        for epoch_index in tqdm(range(num_epochs), desc="Epochs"):
            # Train the model
            self.model.train()
            mean_loss_dict_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True
            )

            # Evaluate the model
            self.model.eval()
            with torch.no_grad():
                mean_loss_dict_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                    data_loader=generator_val,
                    epoch_num=epoch_index,
                    train=False
                )

                self.eval_model(
                    data_loader=generator_val,
                    epoch_num=epoch_index
                )

            for k in mean_loss_dict_train:
                self.writer.add_scalar(
                    f"loss_{k}/training", mean_loss_dict_train[k], epoch_index)
                self.writer.add_scalar(
                    f"loss_{k}/validation", mean_loss_dict_val[k], epoch_index)

            self.writer.add_scalar(
                "accuracy/training", mean_accuracy_train, epoch_index)
            self.writer.add_scalar(
                "accuracy/validation", mean_accuracy_val, epoch_index)

            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_dict_train["sum"],
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_dict_val["sum"],
                'mean_accuracy_val': mean_accuracy_val
            }
            self.print_epoch_stats(**data_element)

            if self.checkpoint_index and (epoch_index % self.checkpoint_index == 0):
                self.model.save_checkpoint(epoch_index)

        self.model.save()

    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        # from trainer
        mean_loss_dict = {}
        mean_accuracy = 0
        # for batch_num, batch in tqdm(enumerate(data_loader), desc="Batch"):
        for batch_num, batch in enumerate(data_loader):
            batch_data = self.process_batch_data(batch)

            self.optimizer.zero_grad()

            loss_dict, accuracy = self.loss_and_acc_for_batch(
                batch_data, epoch_num, batch_num, train=train
            )

            loss = loss_dict["sum"]

            if train:
                loss.backward()
                self.optimizer.step()

            for k, v in loss_dict.items():
                mean_loss_dict[k] = utl.to_numpy(
                    loss_dict[k].mean()) + mean_loss_dict.setdefault(k, 0)
            if accuracy is not None:
                mean_accuracy += utl.to_numpy(accuracy)

        for k, v in mean_loss_dict.items():
            mean_loss_dict[k] = v / len(data_loader)
        mean_accuracy /= len(data_loader)
        return (mean_loss_dict, mean_accuracy)

    def loss_and_acc_for_batch(self, batch, epoch_num=None, batch_num=None, train=True):
        raise NotImplementedError

    def process_batch_data(self, batch):
        raise NotImplementedError

    def reconstruction_loss(input, target):
        raise NotImplementedError

    @staticmethod
    def compute_kld_loss(z_dist, prior_dist, beta, c=0.0):
        # from trainer
        kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
        kld_loss = kld.sum(1).mean()
        kld_loss = beta * (kld_loss - c).abs()
        return kld_loss, kld

    @staticmethod
    def compute_reg_loss(z, labels, reg_dim, gamma, factor=1.0):
        # from trainer
        x = z[:, reg_dim]
        reg_loss = Trainer.reg_loss_sign(x, labels, factor=factor)
        return gamma * reg_loss

    @staticmethod
    def reg_loss_sign(latent_code, attribute, factor=1.0):
        # compute latent distance matrix
        latent_code = latent_code.view(-1, 1).repeat(1, latent_code.shape[0])
        lc_dist_mat = (latent_code - latent_code.transpose(1, 0)).view(-1, 1)

        # compute attribute distance matrix
        attribute = attribute.view(-1, 1).repeat(1, attribute.shape[0])
        attribute_dist_mat = (
            attribute - attribute.transpose(1, 0)).view(-1, 1)

        # compute regularization loss
        loss_fn = torch.nn.L1Loss()
        lc_tanh = torch.tanh(lc_dist_mat * factor)
        attribute_sign = torch.sign(attribute_dist_mat)
        sign_loss = loss_fn(lc_tanh, attribute_sign.float())

        return sign_loss

    @staticmethod
    def mean_accuracy(weights, targets):
        raise NotImplementedError

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
        raise NotImplementedError

    # TODO necessary?
    def test_model(self, data_loader):
        raise NotImplementedError

    # TODO implement
    def loss_and_acc_test():
        # from image_vae_trainer
        pass

    @staticmethod
    def print_epoch_stats(epoch_index, num_epochs, mean_loss_train, mean_accuracy_train, mean_loss_val, mean_accuracy_val):
        # from trainer
        print(
            f'Train Epoch: {epoch_index + 1}/{num_epochs}')
        print(f'\tTrain Loss: {mean_loss_train}'
              f'\tTrain Accuracy: {mean_accuracy_train * 100} %'
              )
        print(
            f'\tValid Loss: {mean_loss_val}'
            f'\tValid Accuracy: {mean_accuracy_val* 100} %'
        )
