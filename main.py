import torch
import datetime
import time
import click
import pprint
import json
from pathlib import Path
from itertools import product
from tqdm import tqdm

from models.naive_model import NaiveVAE
from models.naive_trainer import NaiveTrainer
from models.tvae_model import TVAE
from models.tvae_trainer import TVAETrainer
from utils.datasets import DatasetBERT, DatasetWordPiece


@click.command()
@click.option("--dry-run", "dry_run", is_flag=True, type=bool, default=False, show_default=True, help="Do not train/evaluate any model.")
@click.option("--train", "train", is_flag=True, type=bool, default=False, show_default=True, help="Flag, if a model is to be trained.")
@click.option("--evaluate", "evaluate", type=click.Path(exists=True, path_type=Path), help="Evaluate a specific model.")
@click.option("--no-log", "no_log", is_flag=True, type=bool, default=False, show_default=True, help="Toggle logging.")
@click.option("--save-model", "save_model", is_flag=True, type=bool, default=False, show_default=True, help="Toggle model saving.")
@click.option("--iteration", "iteration", type=int, default=1, show_default=True, help="How many times the program is repeated.")
@click.option("-M", "--model", "model_selection", type=click.Choice(["TVAE", "Naive"], case_sensitive=False), default="TVAE", show_default=True, help="The model to be used.")
@click.option("-D", "--dataset", "dataset", type=click.Choice(["German", "Wikipedia", "All"], case_sensitive=False), default="German", show_default=True, help="Determine the dataset(s) to be used.")
@click.option("-N", "--name", "name", type=str, default="", show_default=True, help="Alternative name of the output folders.")
@click.option("-E", "--emb-length", "emb_length", type=int, default=128, show_default=True, help="Sets the length of the WordPiece embedding.")
@click.option("--num-epochs", "--epochs", "num_epochs", type=int, default=25, show_default=True, help="Number of epochs to be trained.")
@click.option("-B", "--batch-size", "batch_size", type=int, default=[64], multiple=True, show_default=True, help="Size of the batches to be trained.")
@click.option("-klM", "--kl-cycles", "kl_Ms", type=int, default=[4], multiple=True, show_default=True, help="for kl cyclical annealing; multiple values possible.")
@click.option("-klR", "--kl-proportion", "kl_Rs", type=float, default=[0.5], multiple=True, show_default=True, help="for kl cyclical annealing; multiple values possible.")
@click.option("-L", "--learning-rate", "learning_rate", type=float, default=[1e-4], multiple=True, show_default=True, help="Learning rate(s); multiple values possible.")
@click.option("-Ca", "--capacity", "capacity", type=float, default=[0.0], multiple=True, show_default=True, help="Capacity, capacity of kld's bottleneck channel; multiple values possible.")
@click.option("-Al", "--alpha", "alpha", type=float, default=[1.0], multiple=True, show_default=True, help="Alpha, impact of reconstruction error on loss; multiple values possible.")
@click.option("-Be", "--beta", "--beta-scale", "beta", type=float, default=[1.0], multiple=True, show_default=True, help="Scaling value for beta, scales beta given by kl annealing; impact of kld-regularization on loss; multiple values possible.")
@click.option("-Ga", "--gamma", "gamma", type=float, default=[10.0], multiple=True, show_default=True, help="Gamma, impact of ar-regularization on loss; multiple values possible.")
@click.option("-De", "--delta", "delta", type=float, default=[1.0], multiple=True, show_default=True, help="Delta, tuning the spread of the posterior distribution in ar-regularization (scales z distances); multiple values possible.")
@click.option("--no-reg", "use_reg_loss", is_flag=True, type=bool, default=True, show_default=True, help="Use regularization as defined by Pati et al (2020) - 'Attribute-based Regularization of Latent Spaces for Variational Auto-Encoders'.")
@click.option("-C", "--checkpoint-index", "checkpoint_index", type=int, default=0, show_default=True, help="Frequency of checkpoint creation. 0 disables checkpoints.")
@click.option("-d", "--d-model", "d_model", type=int, default=256, show_default=True, help="Internal dimension size of the TVAE model. Each sublayer produces this output size.")
@click.option("-z", "--z-dim", "z_dim", type=int, default=128, show_default=True, help="Size of the latent dimension.")
@click.option("-ne", "--nhead-encoder", "nhead_encoder", type=int, default=8, show_default=True, help="Number of attention heads in transformer encoder (Also used as linear dim of encoder in NaiveModel).")
@click.option("-nd", "--nhead-decoder", "nhead_decoder", type=int, default=8, show_default=True, help="Number of attention heads in transformer decoder (Also used as linear dim of decoder in NaiveModel).")
@click.option("-dh", "--d-hid", "d_hid", type=int, default=1024, show_default=True, help="Dimension of transformer's linear layer.")
@click.option("-nl", "--layers", "--nlayers", "nlayers", type=int, default=[1], multiple=True, show_default=True, help="Number of transformer blocks; multiple values possible.")
@click.option("-do", "--dropout", "dropout", type=float, default=0.1, show_default=True, help="Dropout value for the model.")
def main(dry_run: bool, train: bool, evaluate: Path, no_log: bool, save_model: bool, iteration: int, model_selection: str, dataset: str, name: str, emb_length: int, num_epochs: int, batch_size: int, kl_Ms: int, kl_Rs: float, learning_rate: float, capacity: float, alpha: float, beta: float, gamma: float, delta: float, use_reg_loss: bool, checkpoint_index: int, d_model: int, z_dim: int, nhead_encoder: int, nhead_decoder: int, d_hid: int, nlayers: int, dropout: float):
    # TODO assert value must adhere to specific ranges
    # e.g. 0 < lr < 10 for example

    args = locals()

    print("Parameters:")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(args)

    if not torch.cuda.is_available():
        print("Cuda was not found.")
        return

    if dataset == "German":
        if model_selection == "TVAE":
            datasets = [DatasetWordPiece(large=False, max_length=emb_length)]
        else:
            datasets = [DatasetBERT(large=False)]
    elif dataset == "Wikipedia":
        if model_selection == "TVAE":
            datasets = [DatasetWordPiece(large=True, max_length=emb_length)]
        else:
            datasets = [DatasetBERT(large=True)]
    elif dataset == "All":
        if model_selection == "TVAE":
            datasets = [DatasetWordPiece(large=False, max_length=emb_length), DatasetWordPiece(
                large=True, max_length=emb_length)]
        else:
            datasets = [DatasetBERT(large=False), DatasetBERT(large=True)]

    if dry_run or not (train or evaluate):
        return

    if evaluate:
        eval(evaluate)

    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime(
        '%Y-%m-%d_%H:%M:%S'
    )

    if not name:
        folder_path = Path(f"{timestamp}_{model_selection}_{str(dataset)}")
    else:
        folder_path = Path(f"{timestamp}_{name}")
    p = Path(
        "logs", f"{folder_path.stem}_summary.json")
    folder_log = Path("logs", folder_path)
    if not no_log:
        if not folder_log.exists():
            folder_log.mkdir(parents=True)
        with open(p, "w") as fp:
            json.dump(args, fp, indent=4, sort_keys=True)

    parameters = [i for i in product(
        batch_size, nlayers, kl_Ms, kl_Rs, learning_rate, alpha, beta, gamma, delta, capacity, datasets)]
    if train:
        for batch_size, nlayer, kl_M, kl_R, lr, al, be, ga, de, ca, dataset in tqdm(parameters, desc="Models"):
            args["dataset"] = str(dataset)
            args["batch_size"] = batch_size
            args["nlayers"] = nlayer
            args["kl_Ms"] = kl_M
            args["kl_Rs"] = kl_R
            args["learning_rate"] = lr
            args["alpha"] = al
            args["beta"] = be
            args["gamma"] = ga
            args["delta"] = de
            args["capacity"] = ca
            for _ in tqdm(range(iteration), desc="Repetitions"):
                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                    '%Y-%m-%d_%H:%M:%S'
                )

                if model_selection == "TVAE":
                    model = TVAE(
                        ntoken=dataset.vocab_size,
                        d_model=d_model,
                        z_dim=z_dim,
                        nhead_encoder=nhead_encoder,
                        nhead_decoder=nhead_decoder,
                        d_hid=d_hid,
                        nlayers=nlayer,
                        dropout=dropout,
                        use_gru=False,
                        foldername=dataset.__str__(),
                        timestamp=timestamp
                    )
                    Trainer = TVAETrainer
                else:
                    model = NaiveVAE(
                        input_size=dataset.getInputSize(),
                        z_dim=z_dim,
                        encoder_dim=nhead_encoder,
                        decoder_dim=nhead_decoder,
                        foldername=dataset.__str__(),
                        timestamp=timestamp
                    )
                    Trainer = NaiveTrainer

                if not no_log:
                    path_log = Path(
                        folder_log, f"{timestamp}_{str(model)}_{str(dataset)}.json")
                    with open(path_log, "w") as fp:
                        json.dump(args, fp, indent=4, sort_keys=True)

                if not no_log:
                    path = Path(folder_path, "_".join(
                        [timestamp, str(model), "Reg"+str(use_reg_loss)]))
                else:
                    path = ""

                trainer = Trainer(
                    dataset=dataset,
                    model=model,
                    checkpoint_index=checkpoint_index,
                    M=kl_M,
                    R=kl_R,
                    lr=lr,
                    alpha=al,
                    beta=be,
                    gamma=ga,
                    delta=de,
                    capacity=ca,
                    use_reg_loss=use_reg_loss,
                    save_model=save_model,
                    folderpath=path
                )

                model.update_filepath(folderpath=path)
                # while True:
                try:
                    trainer.train_model(
                        batch_size=batch_size,
                        num_epochs=num_epochs
                    )
                    # continue
                except ValueError as err:
                    with open("log.txt", "a") as fp:
                        fp.write(str(err))
                        fp.write(f"\n\nSee{str(path_log)}\n\n\n\n")
                    # break
                # except torch.cuda.OutOfMemoryError as err:
                #     continue


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
