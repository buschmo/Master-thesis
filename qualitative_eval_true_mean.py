from models.tvae_model import TVAE
from models.tvae_trainer import TVAETrainer
from utils.datasets import DatasetWordPiece
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging
import click
import json
import pandas as pd
import numpy as np
logging.set_verbosity_error()

print(torch.__version__)

models = [
    (Path("ModelGerman.pt"),
        3,
        DatasetWordPiece(large=False, max_length=128),
        14779805221749554585),
    (Path("ModelGermanNoReg.pt"),
        3,
        DatasetWordPiece(large=False, max_length=128),
        9911442391652574031),
    (Path("ModelWiki.pt"),
        3,
        DatasetWordPiece(large=True, max_length=128),
        6003809420069737480),
    (Path("ModelWikiNoReg.pt"),
        3,
        DatasetWordPiece(large=True, max_length=128),
        6003809420069737480)
]

ATTRIBUTE_DIMENSIONS = {
    "Simplicity": 0,
    "Tree_depth": 1,
    "POS": 2,
    "Length": 3,
    "TF-IDF": 4
}


def compute_latent_interpolations(z, mean=0, std=2, dim=0, num_points=9):
    x = torch.linspace(mean-2*std, mean+2*std, num_points)
    z = z.repeat(num_points, 1)
    z[:, dim] = x.contiguous()
    return z


def sentence_accuracy(weights, targets):
    # get predicted label
    weights = torch.argmax(weights, dim=-1)
    # remove [PAD] label (== 0) from accuracy calculation
    x = weights.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    masked = np.ma.masked_array(x == y, y < 0.5)
    numerator = np.sum(np.ma.masked_array(x == y, y < 0.5), axis=-1)
    denominator = np.sum(y > 0.5, axis=-1)
    acc = numerator/denominator
    return acc


def to_numpy(t):
    return t.detach().cpu().numpy()


def to_float(t):
    return float(t.detach().cpu())


def decode(batch, trainer, model):
    src, tgt, tgt_true, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, labels = trainer.process_batch_data(
        batch)

    z_dist = model.encode(src, src_key_padding_mask)
    z_tilde, z_prior, prior_dist = model.reparametrize(z_dist)

    logits = model.decode(
        z_tilde=z_tilde,
        tgt=tgt,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask
    )
    out_tokens = torch.argmax(logits, dim=-1)

    return out_tokens, tgt_true


@click.command()
@click.option("-s","--sample", "flag_sample", is_flag=True, type=bool, default=False, show_default=True, help="Flag, if sampling should be done.")
@click.option("-m","--mean", "flag_mean", is_flag=True, type=bool, default=False, show_default=True, help="Flag, if true mean and std should be used for lantent sampling.")
def main(flag_sample: bool, flag_mean:bool):
    if not Path("results").exists():
        Path("results").mkdir()

    batch_size = 64

    interpolations = {}
    examples = {}
    for path_model, nlayers, dataset, seed in models:
        model = TVAE(ntoken=dataset.vocab_size, nlayers=nlayers)
        model.load_state_dict(torch.load(str(path_model)))
        model.cuda()
        model.eval()
        trainer = TVAETrainer(dataset=dataset, model=model)

        # prepare dataset
        generator = torch.Generator().manual_seed(seed)
        _, dataset_val = torch.utils.data.random_split(
            dataset, [0.8, 0.2], generator=generator
        )
        data_loader = DataLoader(dataset_val, batch_size=batch_size)

        accuracies = []

        # for saving a good and a bad example
        sent_best = None
        sent_true_best = None
        acc_best = 0
        sent_worst = None
        sent_true_worst = None
        acc_worst = 1

        # for saving 3 good sentences for interpolation
        sents_interpolation = []
        interpolations[path_model.stem] = {}

        # for saving sampling
        path_sampling = Path("results/"+path_model.stem+"Sampling.dat")
        sampling_z = []
        sampling_attr = []
        sampling_out_tokens = []
        sampling_tgt_true = []


        # for num, batch in enumerate(data_loader):
        for num, batch in tqdm(enumerate(data_loader), leave=True, total=len(data_loader), desc=f"Sentence batch"):
            src, tgt, tgt_true, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, labels = trainer.process_batch_data(
                batch)

            z_dist = model.encode(src, src_key_padding_mask)
            z_tilde, z_prior, prior_dist = model.reparametrize(z_dist)

            logits = model.decode(
                z_tilde=z_tilde,
                tgt=tgt,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            out_tokens = torch.argmax(logits, dim=-1)
            accuracy = sentence_accuracy(logits, tgt_true)

            accuracies.append(accuracy)
            # save sampling
            sampling_z.append(to_numpy(z_tilde))
            sampling_attr.append(to_numpy(labels))
            sampling_out_tokens.append(to_numpy(out_tokens))
            sampling_tgt_true.append(to_numpy(tgt_true))

        accuracies = np.concatenate(accuracies)
        sampling_z = np.vstack(sampling_z)
        sampling_attr = np.vstack(sampling_attr)
        sampling_out_tokens = np.vstack(sampling_out_tokens)
        sampling_tgt_true = np.vstack(sampling_tgt_true)
        sampling_out_tokens = np.vstack(sampling_out_tokens)
        sampling_tgt_true = np.vstack(sampling_tgt_true)

        # get mean and std
        sampling_mean = np.mean(sampling_z, axis=0)
        sampling_std = np.std(sampling_z, axis=0)

        if flag_sample:
            sampling_z = pd.DataFrame(sampling_z)
            sampling_attr = pd.DataFrame(
                sampling_attr, columns=ATTRIBUTE_DIMENSIONS.keys())

            sampling = pd.concat([sampling_z, sampling_attr], axis=1)
            sampling.to_csv(path_sampling, sep="\t", index=False)

        # save accuracy>90% sentence
        for dim in np.flatnonzero(accuracies > .9):
            if (sampling_attr[dim][3] > 5 and sampling_attr[dim][3] < 15):
                # save sentences
                sent_best = [int(i)for i in sampling_out_tokens[dim]]
                sent_true_best = [int(i)for i in sampling_tgt_true[dim]]
                acc_best = accuracies[dim]
                break
        # save 60%>accuracy>50% sentence
        for dim in np.where(((accuracies > .5) & (accuracies < .6)))[0]:
            if (sampling_attr[dim][3] > 5 and sampling_attr[dim][3] < 15):
                # save sentences
                sent_worst = [int(i) for i in sampling_out_tokens[dim].tolist()]
                sent_true_worst = [int(i) for i in sampling_tgt_true[dim].tolist()]
                acc_worst = accuracies[dim]
                break

        examples[path_model.stem] = {
            "sent_best": dataset.tokenizer.decode(sent_best),
            "sent_true_best": dataset.tokenizer.decode(sent_true_best),
            "acc_best": acc_best,
            "sent_worst": dataset.tokenizer.decode(sent_worst),
            "sent_true_worst": dataset.tokenizer.decode(sent_true_worst),
            "acc_worst": acc_worst,
            # "sampling_mean": sampling_mean,
            # "sampling_std": sampling_std
        }

        for dim in np.where((accuracies > .9))[0]:
            if (sampling_attr[dim][3] > 5 and sampling_attr[dim][3] < 15):
                sent_z = torch.Tensor(sampling_z[dim]).view(1,-1).to("cuda").float()
                sent_true = dataset.tokenizer.decode(
                    [int(i) for i in sampling_tgt_true[dim]])

                batch = dataset[dim]
                _, tgt, _, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, _ = trainer.process_batch_data(
                    (batch[0].view(1, -1).repeat(9,1), batch[1].view(1, -1).repeat(9,1)))

                interpolations[path_model.stem][sent_true] = {}
                interpolations[path_model.stem][sent_true]["Accuracy"] = accuracies[dim]
                # interpolate for every attribute
                for attr, attr_dim in ATTRIBUTE_DIMENSIONS.items():
                    if flag_mean:
                        mean = float(sampling_mean[attr_dim])
                        std = float(sampling_std[attr_dim])
                    else:
                        mean= float(0)
                        std = float(2)
                    # [z_dim] -> ["batch", z_dim]
                    interp = compute_latent_interpolations(
                        sent_z, mean, std, dim=attr_dim)

                    logits = model.decode(
                        z_tilde=interp,
                        tgt=tgt,
                        tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask
                    )
                    out_tokens = torch.argmax(logits, dim=-1)

                    interpolations[path_model.stem][sent_true][attr] = {}
                    interpolations[path_model.stem][sent_true][attr]["mean"] = mean
                    interpolations[path_model.stem][sent_true][attr]["std"] = std
                    interpolations[path_model.stem][sent_true][attr]["label"] = sampling_attr[dim][attr_dim]
                    for i, out_token in enumerate(out_tokens):
                        out_token = [int(i) for i in out_token.tolist()]
                        interp_sent = dataset.tokenizer.decode(out_token)
                        interpolations[path_model.stem][sent_true][attr][mean-(2*std) + (i*std/2)] = interp_sent

                # Break if enough were found already
                if len(interpolations[path_model.stem]) >= 3:
                    break

    # save two example sentences
    with open("results/Examples.json", "w") as fp:
        json.dump(examples, fp, indent=4)

    with open("results/Interpolation.json", "w") as fp:
        json.dump(interpolations, fp, indent=4)


if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("No cuda available")
