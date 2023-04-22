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
logging.set_verbosity_error()

print(torch.__version__)

models = [
    (Path("ModelGerman.pt"),
        3,
        DatasetWordPiece(large=False, max_length=128),
        "14779805221749554585"),
    (Path("ModelGermanNoReg.pt"),
        3,
        DatasetWordPiece(large=False, max_length=128),
        "9911442391652574031"),
    (Path("ModelWiki.pt"),
        3,
        DatasetWordPiece(large=True, max_length=128),
        "6003809420069737480"),
    (Path("ModelWikiNoReg.pt"),
        3,
        DatasetWordPiece(large=True, max_length=128),
        "6003809420069737480")
]

ATTRIBUTE_DIMENSIONS = {
    "Simplicity": 0,
    "Tree_depth": 1,
    "POS": 2,
    "Length": 3,
    "TF-IDF": 4
}


def compute_latent_interpolations(z, mean, std, dim=0, num_points=7):
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


def main():
    # needed for processing batch data
    fake_label = torch.IntTensor([[1]])
    batch_size = 64

    for path_model, nlayers, dataset, seed in models:
        model = TVAE(ntoken=dataset.vocab_size, nlayers=nlayers)
        model.load_state_dict(torch.load(str(path_model)))
        # model.cuda()
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
        interpolations = {}

        # for saving sampling
        sampling_z = []
        sampling_attr = []

        for num, batch in tqdm(enumerate(data_loader), leave=False, total=len(data_loader), desc=f"Sentences"):
            # for num, batch in enumerate(data_loader):
            # t = torch.Tensor(dataset.encode())
            # t = t.view(1, -1).long()
            # batch = trainer.process_batch_data((t, fake_label))
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
            sampling_z.append(z_tilde)
            sampling_attr.append(labels)

            # save best sentence
            if accuracy.max() > acc_best:
                acc_best = accuracy.max()
                sent_best = out_tokens[accuracy.argmax()]
                sent_true_best = tgt_true[accuracy.argmax()]
            # save worst sentence
            if accuracy.min() < acc_worst:
                acc_worst = accuracy.min()
                sent_worst = out_tokens[accuracy.argmin()]
                sent_true_worst = tgt_true[accuracy.argmin()]

            # Save 3 sentences with accuracy over 90%
            if any(accuracy > .9) and len(interpolations) < 3:
                # interpolate every sentence
                for dim in np.flatnonzero(accuracy > .9):
                    # get sentence
                    sent_z = z_tilde[dim]
                    sent_true = tgt_true[dim]

                    interpolations[sent_true] = {}
                    # interpolate for every attribute
                    for attr, attr_dim in ATTRIBUTE_DIMENSIONS.items():
                        mean, std = z_dist.loc, z_dist.scale
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

                        interpolations[sent_true][attr] = {}
                        for i, out_token in enumerate(out_tokens):
                            out_token = [int(i) for i in out_token.tolist()]
                            interp_sent = dataset.tokenizer.decode(out_token)

                            interpolations[sent_true][attr][i] = interp_sent

                    # Break if enough were found already
                    if len(interpolations) >= 3:
                        break

        stem = path_model.stem
        # save two example sentences
        with open(path_model.with_name(stem+"Example.json"), "w") as fp:
            json.dump({
                "sent_best": dataset.tokenizer.decode(sent_best),
                "sent_true_best": dataset.tokenizer.decode(sent_true_best),
                "acc_best": acc_best,
                "sent_worst": dataset.tokenizer.decode(sent_worst),
                "sent_true_worst": dataset.tokenizer.decode(sent_true_worst),
                "acc_worst": acc_worst
            }, fp, indent=4)

        with open(path_model.with_name(stem+"Interpolation.json"), "w") as fp:
            json.dump(interpolations, fp, indent=4)

        # with open(path_model.with_name(stem+"Sampling.dat"), "w") as fp:
        sampling_z = torch.concat(sampling_z)
        sampling_attr = torch.concat(sampling_attr)

        sampling_z = pd.DataFrame(sampling_z)
        sampling_attr = pd.DataFrame(
            sampling_attr, columms=ATTRIBUTE_DIMENSIONS.keys())

        sampling = pd.concat([sampling_z, sampling_attr], axis=1)

        sampling.to_csv(path_model.with_name(
            stem+"Sampling.dat"), sep="\t", index=False)

        # out_tokens = torch.argmax(logits, dim=-1)
        # out_tokens = [int(i) for i in list(out_tokens.data.to("cpu")[0])]
        # print("1", dataset.tokenizer.decode(out_tokens))
        # true_tokens = list(d["tgt_true"][0].cpu().numpy())
        # print("2", dataset.tokenizer.decode(true_tokens))
        # if num > 5:
        #     break


if __name__ == "__main__":
    if torch.cuda.is_available():
        main()
    else:
        print("No cuda available")
