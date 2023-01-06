import torch
import datetime
import time
import click
from pathlib import Path

from models.naive_model import NaiveVAE
from models.naive_trainer import NaiveTrainer
from models.tvae_model import TVAE
from models.tvae_trainer import TVAETrainer
from utils.datasets import SimpleGermanDatasetBERT, SimpleWikipediaDatasetBERT, DatasetWordPiece

device = "cuda" if torch.cuda.is_available() else "cpu"


@click.command()
@click.option("-t", "--train", "train", type=bool, default=False, show_default=True, help="")
@click.option("-e", "--evaluate", "evaluate", type=click.Path(exists=True, path_type=Path), help="Evaluate a specific model.")
@click.option("-M", "--model", "model", type=click.Choice(["TVAE", "Naive"], case_sensitive=False), default="TVAE", show_default=True, help="The model to be used.")
@click.option("-D", "--dataset", "dataset", type=click.Choice(["German", "Wikipedia", "All"], case_sensitive=False), default="All", show_default=True, help="Determine the dataset(s) to be used.")
@click.option("-L", "--emb-length", "emb_length", type=int, default=512, show_default=True, help="Sets the length of the WordPiece embedding.")
@click.option("-N", "--num-epochs", type=int, default=50, show_default=True, help="Number of epochs to be trained.")
@click.option("-B", "--batch-size", "batch_size", type=int, default=32, show_default=True, help="Size of the batches to be trained.")
@click.option("--reg/--no-reg", "use_reg_loss", default=True,  show_default=True, help="Use regualarization as defined by ")
@click.option("-C", "--checkpoint-index", "checkpoint_index", type=int, default=0, show_default=True, help="Frequency of checkpoint creation. 0 disables checkpoints.")
@click.option("-d", "--dim", "d_model", type=int, default=256, show_default=True, help="Dimension size of the TVAE model")
@click.option("-z", "--z-dim", "z_dim", type=int, default=64, show_default=True, help="Size of the latent dimension.")
@click.option("-ne", "--nhead-encoder", "nhead_encoder", type=int, default=8, show_default=True, help="Number of attention heads in transformer encoder (Also used as linear dim of encoder in NaiveModel)")
@click.option("-nd", "--nhead-decoder", "nhead_decoder", type=int, default=8, show_default=True, help="Number of attention heads in transformer decoder (Also used as linear dim of decoder in NaiveModel)")
@click.option("-dh", "--d-hid", "d_hid", type=int, default=512, show_default=True, help="Dimension of transformer's linear layer")
@click.option("-nl", "--nlayers", "nlayers", type=int, default=1, show_default=True, help="Number of transformer blocks")
@click.option("-do", "--dropout", "dropout", type=float, default=0.1, show_default=True, help="Dropout value")
def main(train: bool, evaluate: Path, model: str, dataset: str, emb_length: int, num_epochs: int, batch_size: int, use_reg_loss: bool, checkpoint_index: int, d_model: int, z_dim: int, nhead_encoder: int, nhead_decoder: int, d_hid: int, nlayers: int, dropout: float):
    if device == "cpu":
        print("Cuda was not found.")
        return

    if dataset == "German":
        datasets = [DatasetWordPiece(large=False, max_length=emb_length)] if model == "TVAE" else [
            SimpleGermanDatasetBERT()]
    elif dataset == "Wikipedia":
        datasets = [DatasetWordPiece(large=True, max_length=emb_length)] if model == "TVAE" else [
            SimpleGermanDatasetBERT()]
    elif dataset == "All":
        datasets = [DatasetWordPiece(large=False, max_length=emb_length), DatasetWordPiece(large=True, max_length=emb_length)] if model == "TVAE" else [
            SimpleGermanDatasetBERT(), SimpleWikipediaDatasetBERT()]

    if evaluate:
        eval(evaluate)

    if train:
        if model == "TVAE":
            for dataset in datasets:
                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                    '%Y-%m-%d_%H:%M:%S'
                )
                model = TVAE(
                    ntoken=dataset.vocab_size,
                    d_model=d_model,
                    z_dim=z_dim,
                    nhead_encoder=nhead_encoder,
                    nhead_decoder=nhead_decoder,
                    d_hid=d_hid,
                    nlayers=nlayers,
                    dropout=dropout,
                    use_gru=False,
                    foldername=dataset.__str__(),
                    timestamp=timestamp)
                trainer = TVAETrainer(
                    dataset=dataset,
                    model=model,
                    checkpoint_index=checkpoint_index,
                    use_reg_loss=use_reg_loss,
                    timestamp=timestamp
                )

                trainer.train_model(
                    batch_size=batch_size,
                    num_epochs=num_epochs
                )

        else:
            for dataset in datasets:
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
                    checkpoint_index=checkpoint_index,
                    use_reg_loss=use_reg_loss,
                    timestamp=timestamp
                )

                trainer.train_model(
                    batch_size=batch_size,
                    num_epochs=num_epochs
                )


def eval(path):
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    if path.is_dir():
        runs = set()
        l = [i for i in path.iterdir()]
        d = set()
        for i in l:
            d.add(i.name[:19])
        d = {k: [] for k in d}
        for i in l:
            if len(i.stem) != 19:
                d[i.stem[:19]].append(i)
        dataset = DatasetWordPiece(large=True)
        data_loader = DataLoader(dataset, batch_size=32)
        for k, v in tqdm(d.items(), desc="Model Iteration"):
            # Writer anlegen
            # Datensätze laden
            if k == "2022-12-27_10:09:49":
                reg = True
            else:
                reg = False
            old_epoch = -1
            for f in tqdm(v, desc="Epochs Iteration"):
                epoch_num = int(f.stem.split("_")[-1])
                if epoch_num == old_epoch+1:
                    old_epoch = epoch_num
                else:
                    print("Error!")
                    break
                model = TVAE(
                    ntoken=dataset.vocab_size,
                    d_model=256,
                    z_dim=64,
                    nhead_encoder=8,
                    nhead_decoder=8,
                    d_hid=512,
                    nlayers=1,
                    dropout=0.1,
                    use_gru=False
                )
                model.load_state_dict(torch.load(f))

                trainer = TVAETrainer(
                    dataset=dataset,
                    model=model,
                    checkpoint_index=0,
                    use_reg_loss=reg,
                    timestamp="FullEvaluation"
                )

                model.eval()
                trainer.eval_model(data_loader, epoch_num)


if __name__ == "__main__":
    main()
