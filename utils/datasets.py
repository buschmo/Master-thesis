import click
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import pipeline, BertTokenizer
from transformers import logging
logging.set_verbosity_error()


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

    def __str__(self):
        raise NotImplementedError

    def getInputSize(self):
        return self.embeddings.shape[1]


class DatasetBERT(BaseDataset):
    def __init__(self, large: bool = False):
        if large:
            self.path_easy = Path("data/SimpleWikipedia/BERTeasy.pt")
            self.path_normal = Path("data/SimpleWikipedia/BERTnormal.pt")
            self.path_easy_input = Path(
                "data/SimpleWikipedia/sentence-aligned.v2/simple.aligned")
            self.path_normal_input = Path(
                "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned")
            self.path_easy_attribute = Path(
                "data/SimpleWikipedia/simple_attribute.aligned.pt")
            self.path_normal_attribute = Path(
                "data/SimpleWikipedia/normal_attribute.aligned.pt")
            self.str = "SimpleWikipediaCorpus"
        else:
            self.path_easy = Path("data/SimpleGerman/BERTeasy.pt")
            self.path_normal = Path("data/SimpleGerman/BERTnormal.pt")
            self.path_easy_input = Path("data/SimpleGerman/fixed_easy.txt")
            self.path_normal_input = Path("data/SimpleGerman/fixed_normal.txt")
            self.path_easy_attribute = Path(
                "data/SimpleGerman/fixed_easy_attribute.pt")
            self.path_normal_attribute = Path(
                "data/SimpleGerman/fixed_normal_attribute.pt")
            self.str = "SimpleGermanCorpus"
        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        t_easy = torch.load(self.path_easy)
        t_normal = torch.load(self.path_normal)

        t_easy_attr = torch.load(self.path_easy_attribute)
        t_normal_attr = torch.load(self.path_normal_attribute)

        self.embeddings = torch.cat(
            [t_easy, t_normal])
        self.labels = torch.cat([t_easy_attr, t_normal_attr])

    def __str__(self):
        return self.str

    def createDataset(self):
        def generator(file, model_name):
            tokenizer = BertTokenizer.from_pretrained(model_name)
            with open(file) as fp:
                if self.large:
                    lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
                else:
                    lines = [i.strip() for i in fp.readlines()]
            for line in lines:
                if len(tokenizer.encode(line)) > 512:
                    continue
                yield line

        if large:
            # English BERT from https://huggingface.co/bert-base-uncased
            model_name = "bert-base-uncased"
        else:
            # German BERT from https://huggingface.co/deepset/gbert-base
            model_name = "deepset/gbert-base"
        pipe = pipeline(model=model_name, tokenizer=model_name,
                        task="feature-extraction")

        if not self.path_easy.exists():
            g1 = generator(self.path_easy_input, model_name)
            # use CLS embedding as sentence embedding
            easy_embeddings = [i[0][0] for i in tqdm(
                pipe(g1), desc=f"{self.str} Easy embeddings")]
            t = torch.tensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator(self.path_normal_input, model_name)
            # use CLS embedding as sentence embedding
            normal_embeddings = [i[0][0] for i in tqdm(
                pipe(g2), desc=f"{self.str} Normal embeddings")]
            t = torch.tensor(normal_embeddings)
            torch.save(t, self.path_normal)


class DatasetWordPiece(BaseDataset):
    def __init__(self, large: bool = False, max_length: int = 512):
        self.large = large
        if large:
            self.path_easy_input = Path(
                "data/SimpleWikipedia/sentence-aligned.v2/simple.aligned")
            self.path_normal_input = Path(
                "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned")
            self.path_easy = Path(
                f"data/SimpleWikipedia/WordPieceEasy{max_length}.pt")
            self.path_normal = Path(
                f"data/SimpleWikipedia/WordPieceNormal{max_length}.pt")
            self.path_easy_attribute = Path(
                "data/SimpleWikipedia/simple_attribute.aligned.pt")
            self.path_normal_attribute = Path(
                "data/SimpleWikipedia/normal_attribute.aligned.pt")
            self.str = "SimpleWikipediaCorpus"
            model_name = "bert-base-uncased"

        else:
            self.path_easy_input = Path("data/SimpleGerman/fixed_easy.txt")
            self.path_normal_input = Path("data/SimpleGerman/fixed_normal.txt")
            self.path_easy = Path(
                f"data/SimpleGerman/WordPieceEasy{max_length}.pt")
            self.path_normal = Path(
                f"data/SimpleGerman/WordPieceNormal{max_length}.pt")
            self.path_easy_attribute = Path(
                "data/SimpleGerman/fixed_easy_attribute.pt")
            self.path_normal_attribute = Path(
                "data/SimpleGerman/fixed_normal_attribute.pt")
            self.str = "SimpleGermanCorpus"
            model_name = "deepset/gbert-base"

        self.str += str(max_length)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer.get_vocab())
        # Save symbols for start, pad and end. Needed for trainer
        self.CLS, self.PAD, self.SEP = self.tokenizer.encode("[PAD]")

        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        t_easy = torch.load(self.path_easy).long()
        t_normal = torch.load(self.path_normal).long()

        t_easy_attr = torch.load(self.path_easy_attribute)
        t_normal_attr = torch.load(self.path_normal_attribute)

        self.embeddings = torch.cat(
            [t_easy, t_normal])
        self.labels = torch.cat([t_easy_attr, t_normal_attr])

    def __str__(self):
        return self.str

    def createDataset(self):
        self.over_model_length = 0
        self.over_max_length = 0

        def generator(file):
            with open(file) as fp:
                if self.large:
                    lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
                else:
                    lines = [i.strip() for i in fp.readlines()]
            for line in lines:
                line_tokens = self.encode(line)
                if not line_tokens:
                    continue
                yield line_tokens

        if not self.path_easy.exists():
            g1 = generator(self.path_easy_input)
            easy_embeddings = [i for i in tqdm(g1, desc=f"Easy {self}")]
            t = torch.IntTensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator(self.path_normal_input)
            normal_embeddings = [i for i in tqdm(g2, desc=f"Normal {self}")]
            t = torch.IntTensor(normal_embeddings)
            torch.save(t, self.path_normal)
        print(
            f"Lines longer than model length: {self.over_model_length}\nLines longer than max length: {self.over_max_length}")

    def encode(self, line):
        line_tokens = self.tokenizer.encode(line, padding="max_length")
        if len(line_tokens) > 512:
            self.over_model_length += 1
            return None
        if line_tokens[self.max_length-1] not in [self.PAD, self.SEP]:
            # content is over max_length
            self.over_max_length += 1
            return None
        return line_tokens[:self.max_length]


@click.command()
@click.option("-w", "--wordpiece", "wordpiece", type=bool, is_flag=True, show_default=True, help="Create WordPiece datasets.")
@click.option("-l", "--emb-length", "emb_length", type=int, default=512, show_default=True, help="Set the maximum length of WordPiece embedding")
@click.option("-b", "--bert", "bert", type=bool, is_flag=True, show_default=True, help="Create BERT embedding datasets.")
def main(wordpiece: bool, emb_length: int, bert: bool):
    if wordpiece:
        DatasetWordPiece(large=False, max_length=emb_length)
        DatasetWordPiece(large=True, max_length=emb_length)
    if bert:
        DatasetBERT(large=False)
        DatasetBERT(large=True)


if __name__ == "__main__":
    main()
