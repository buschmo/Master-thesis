import torch
import datetime
import time
import click

from models.naive_model import NaiveVAE
from models.naive_trainer import NaiveTrainer
from models.tvae_model import TVAE
from models.tvae_trainer import TVAETrainer
from utils.datasets import SimpleGermanDatasetBERT, SimpleWikipediaDatasetBERT, SimpleGermanDatasetWordPiece, SimpleWikipediaDatasetWordPiece

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("-M", "--model", "model", type=click.Choice(["TVAE", "Naive"], default="TVAE", case_sensitive=False), show_default=True, help="The model to be used.")
@click.option("-D", "--dataset", "dataset", type=click.Choice(["German", "Wikipedia", "All"], default="All", case_sensitive=False), show_default=True, help="Determine the dataset(s) to be used.")
@click.option("-N", "--num-epochs",  type=int, default=50, show_default=True, help="Number of epochs to be trained.")
@click.option("--reg/--no-reg", "use_reg_loss", default=True,  show_default=True, help="Use regualarization as defined by ")
@click.option("-C", "--checkpoint-index", "checkpoint_index", type=int, default=0, show_default=True, help="Frequency of checkpoint creation. 0 disables checkpoints.")
@click.option("-d", "--dim", "d_model", type=int, default=256, show_default=True, help="Dimension size of the TVAE model")
@click.option("-z", "--z-dim", "z_dim", type=int, default=64, show_default=True, help="Size of the latent dimension.")
@click.option("-ne", "--nhead-encoder", "nhead_encoder", type=int, default=8, show_default=True, help="Number of attention heads in transformer encoder (Also used as linear dim of encoder in NaiveModel)")
@click.option("-nd", "--nhead-decoder", "nhead_decoder", type=int, default=8, show_default=True, help="Number of attention heads in transformer decoder (Also used as linear dim of decoder in NaiveModel)")
@click.option("-dh", "--d-hid", "d_hid", type=int, default=512, show_default=True, help="Dimension of transformer's linear layer")
@click.option("-nl", "--nlayers", "nlayers", type=int, default=1, show_default=True, help="Number of transformer blocks")
@click.option("-do", "--dropout", "dropout", type=float, default=0.1, show_default=True, help="Dropout value")
def main(model: str, dataset: str, num_epochs: int, use_reg_loss: bool, checkpoint_index: int, d_model: int, z_dim: int, nhead_encoder: int, nhead_decoder: int, d_hid: int, nlayers: int, dropout: float):
    if device == "cpu":
        print("Cuda was not found.")
        return

    if dataset == "German":
        Datasets = [SimpleGermanDatasetWordPiece] if model == "TVAE" else [
            SimpleGermanDatasetWordPiece]
    elif dataset == "Wikipedia":
        Datasets = [SimpleWikipediaDatasetWordPiece] if model == "TVAE" else [
            SimpleGermanDatasetBERT]
    elif dataset == "All":
        Datasets = [SimpleGermanDatasetWordPiece, SimpleWikipediaDatasetWordPiece] if model == "TVAE" else [
            SimpleGermanDatasetBERT, SimpleWikipediaDatasetBERT]

    if model == "TVAE":
        for Dataset in Datasets:
            dataset = Dataset()
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            model = TVAE(
                ntoken=dataset.vocab_size,
                d_model=256,
                z_dim=64,
                nhead_encoder=8,
                nhead_decoder=8,
                d_hid=512,
                nlayers=1,
                dropout=0.1,
                use_gru=False,
                foldername=dataset.__str__(),
                timestamp=timestamp)
            trainer = TVAETrainer(
                dataset=dataset,
                model=model,
                checkpoint_index=1,
                use_reg_loss=use_reg_loss,
                timestamp=timestamp
            )

            trainer.train_model(
                batch_size=8,
                num_epochs=num_epochs
            )

    else:
        for Dataset in Datasets:
            dataset = Dataset()
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            model = NaiveVAE(
                input_size=dataset.getInputSize(),
                z_dim=z_dim,
                encoder_dim=nhead_encoder,
                decoder_dim=nhead_decoder,
                foldername=dataset.__str__(),
                timestamp=timestamp
            )
            trainer = NaiveTrainer(
                dataset=dataset,
                model=model,
                checkpoint_index=0,
                use_reg_loss=use_reg_loss,
                timestamp=timestamp
            )

            trainer.train_model(
                batch_size=64,
                num_epochs=num_epochs
            )


if __name__ == "__main__":
    main()
