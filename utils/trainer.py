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
    def __init__(self, dataset, model, checkpoint_index=0, lr=1e-4, beta=4.0, gamma=10.0, capacity=0.0, delta=1.0, timestamp=""):
        # from trainer
        self.writer = SummaryWriter(
            log_dir=Path("runs", dataset.__str__(),  "_".join(
                [model.__str__(), timestamp]))
        )

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
        self.use_reg_loss = True
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
            mean_loss_train, mean_accuracy_train = self.loss_and_acc_on_epoch(
                data_loader=generator_train,
                epoch_num=epoch_index,
                train=True
            )

            # Evaluate the model
            self.model.eval()
            mean_loss_val, mean_accuracy_val = self.loss_and_acc_on_epoch(
                data_loader=generator_val,
                epoch_num=epoch_index,
                train=False
            )

            self.eval_model(
                data_loader=generator_val,
                epoch_num=epoch_index
            )

            self.writer.add_scalar(
                "loss/training", mean_loss_train, epoch_index)
            self.writer.add_scalar(
                "loss/validation", mean_loss_val, epoch_index)

            data_element = {
                'epoch_index': epoch_index,
                'num_epochs': num_epochs,
                'mean_loss_train': mean_loss_train,
                'mean_accuracy_train': mean_accuracy_train,
                'mean_loss_val': mean_loss_val,
                'mean_accuracy_val': mean_accuracy_val
            }
            self.print_epoch_stats(**data_element)

            if self.checkpoint_index and (epoch_index % self.checkpoint_index == 0):
                self.model.save_checkpoint(epoch_index)

        self.model.save()

    def loss_and_acc_on_epoch(self, data_loader, epoch_num=None, train=True):
        # from trainer
        mean_loss = 0
        mean_accuracy = 0
        for batch_num, (X, y) in tqdm(enumerate(data_loader), desc="Batch"):
            batch_data = (X.to("cuda"), y.to("cuda"))

            self.optimizer.zero_grad()

            loss, accuracy = self.loss_and_acc_for_batch(
                batch_data, epoch_num, batch_num, train=train
            )

            if train:
                loss.backward()
                self.optimizer.step()

            mean_loss += utl.to_numpy(loss.mean())
            if accuracy is not None:
                mean_accuracy += utl.to_numpy(accuracy)

        mean_loss /= len(data_loader)
        mean_accuracy /= len(data_loader)
        return (mean_loss, mean_accuracy)

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

    @staticmethod
    def compute_kld_loss(z_dist, prior_dist, beta, c=0.0):
        # from trainer
        kld = torch.distributions.kl.kl_divergence(z_dist, prior_dist)
        kld = kld.sum(1).mean()
        kld = beta * (kld - c).abs()
        return kld

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

    # TODO rework to sentence usecase
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
