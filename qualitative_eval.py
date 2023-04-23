from models.tvae_model import TVAE
from models.tvae_trainer import TVAETrainer
from utils.datasets import DatasetWordPiece
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging
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


def compute_latent_interpolations(z, mean, std, dim=0, num_points=9):
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


def main():
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
        sampling_z = []
        sampling_attr = []
        # sampling_mean = 0
        # sampling_std = 0

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

            # save sampling
            sampling_z.append(to_numpy(z_tilde))
            sampling_attr.append(to_numpy(labels))

            # save mean and std
            # sampling_mean += torch.sum(z_dist.loc)
            # sampling_std += torch.sum(z_dist.scale)

            # save best sentence
            if accuracy.max() > acc_best:
                acc_best = float(accuracy.max())
                sent_best = [int(i)
                             for i in out_tokens[accuracy.argmax()].tolist()]
                sent_true_best = [int(i)
                                  for i in tgt_true[accuracy.argmax()].tolist()]
            # save worst sentence
            if accuracy.min() < acc_worst:
                acc_worst = float(accuracy.min())
                sent_worst = [int(i)
                              for i in out_tokens[accuracy.argmin()].tolist()]
                sent_true_worst = [
                    int(i) for i in tgt_true[accuracy.argmin()].tolist()]

            # Save 3 sentences with accuracy over 90%
            if any(accuracy > .9) and len(interpolations[path_model.stem]) < 3:
                # interpolate every sentence
                for dim in np.flatnonzero(accuracy > .9):
                    # get sentence
                    sent_z = z_tilde[dim]
                    sent_true = dataset.tokenizer.decode(
                        [int(i) for i in tgt_true[dim].tolist()])

                    # _, tgt, _, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, _ = trainer.process_batch_data(
                    #     (batch[0][dim].view(1, -1).repeat(9,1), batch[1][dim].view(1, -1).repeat(9,1)))
                    interp_tgt = tgt[dim].view(1, -1).repeat(9, 1)
                    interp_src_key_padding_mask = src_key_padding_mask[dim].view(
                        1, -1).repeat(9, 1)
                    interp_tgt_key_padding_mask = tgt_key_padding_mask[dim].view(
                        1, -1).repeat(9, 1)

                    interpolations[path_model.stem][sent_true] = {}
                    # interpolate for every attribute
                    for attr, attr_dim in ATTRIBUTE_DIMENSIONS.items():
                        mean = to_float(z_dist.loc[dim][attr_dim])
                        std = to_float(z_dist.scale[dim][attr_dim])

                        # [z_dim] -> ["batch", z_dim]
                        interp = compute_latent_interpolations(
                            sent_z, mean, std, dim=attr_dim)

                        logits = model.decode(
                            z_tilde=interp,
                            tgt=interp_tgt,
                            tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            src_key_padding_mask=interp_src_key_padding_mask,
                            tgt_key_padding_mask=interp_tgt_key_padding_mask
                        )
                        out_tokens = torch.argmax(logits, dim=-1)

                        interpolations[path_model.stem][sent_true][attr] = {}
                        interpolations[path_model.stem][sent_true][attr]["mean"] = mean
                        interpolations[path_model.stem][sent_true][attr]["std"] = std
                        for i, out_token in enumerate(out_tokens):
                            out_token = [int(i) for i in out_token.tolist()]
                            interp_sent = dataset.tokenizer.decode(out_token)
                            interpolations[path_model.stem][sent_true][attr][i] = interp_sent

                    # Break if enough were found already
                    if len(interpolations[path_model.stem]) >= 3:
                        break

        # sampling_mean = torch.div(sampling_mean, len(data_loader))
        # sampling_std = torch.div(sampling_std, len(data_loader))

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

        # with open(path_model.with_name(stem+"Sampling.dat"), "w") as fp:
        sampling_z = np.vstack(sampling_z)
        sampling_attr = np.vstack(sampling_attr)

        sampling_z = pd.DataFrame(sampling_z)
        sampling_attr = pd.DataFrame(
            sampling_attr, columns=ATTRIBUTE_DIMENSIONS.keys())

        sampling = pd.concat([sampling_z, sampling_attr], axis=1)

        sampling.to_csv("results/"+path_model.stem+"Sampling.dat", sep="\t", index=False)

        # out_tokens = torch.argmax(logits, dim=-1)
        # out_tokens = [int(i) for i in list(out_tokens.data.to("cpu")[0])]
        # print("1", dataset.tokenizer.decode(out_tokens))
        # true_tokens = list(d["tgt_true"][0].cpu().numpy())
        # print("2", dataset.tokenizer.decode(true_tokens))
        # if num > 5:
        #     break

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
