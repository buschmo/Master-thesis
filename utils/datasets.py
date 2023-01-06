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
        return self.embeddings[index], self.labels[index].view(1,)

    def __str__(self):
        raise NotImplementedError

    def getInputSize(self):
        return self.embeddings.shape[1]


class SimpleGermanDatasetBERT(BaseDataset):
    def __init__(self):
        self.path_easy = Path("data/SimpleGerman/BERTeasy.pt")
        self.path_normal = Path("data/SimpleGerman/BERTnormal.pt")
        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        t_easy = torch.load(self.path_easy)
        t_normal = torch.load(self.path_normal)

        self.embeddings = torch.cat(
            [t_easy, t_normal])
        self.labels = torch.cat(
            [torch.zeros(t_easy.shape[0]), torch.ones(t_normal.shape[0])])

    def __str__(self):
        return "SimpleGermanCorpus"

    def createDataset(self):
        def generator(file):
            with open(file) as fp:
                lines = [i.strip() for i in fp.readlines()]
            for line in lines:
                yield line

        # German BERT from https://huggingface.co/deepset/gbert-base
        model_name = "deepset/gbert-base"
        pipe = pipeline(model=model_name, tokenizer=model_name,
                        task="feature-extraction")

        if not self.path_easy.exists():
            g1 = generator("data/SimpleGerman/fixed_easy.txt")
            # use CLS embedding as sentence embedding
            easy_embeddings = [i[0][0] for i in tqdm(
                pipe(g1), desc="German Easy embeddings")]
            t = torch.tensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator("data/SimpleGerman/fixed_normal.txt")
            # use CLS embedding as sentence embedding
            normal_embeddings = [i[0][0] for i in tqdm(
                pipe(g2), desc="German Normal embeddings")]
            t = torch.tensor(normal_embeddings)
            torch.save(t, self.path_normal)


class SimpleWikipediaDatasetBERT(BaseDataset):
    def __init__(self):
        self.path_easy = Path("data/SimpleWikipedia/BERTeasy.pt")
        self.path_normal = Path("data/SimpleWikipedia/BERTnormal.pt")
        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        t_easy = torch.load(self.path_easy)
        t_normal = torch.load(self.path_normal)

        self.embeddings = torch.cat(
            [t_easy, t_normal])
        self.labels = torch.cat(
            [torch.zeros(t_easy.shape[0]), torch.ones(t_normal.shape[0])])

    def __str__(self):
        return "SimpleWikipediaCorpus"

    def createDataset(self):
        def generator(file, tokenizer_name):
            tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
            with open(file) as fp:
                lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
            for line in lines:
                if len(tokenizer.encode(line)) > 512:
                    continue
                yield line

        # English BERT from https://huggingface.co/bert-base-uncased
        model_name = "bert-base-uncased"
        pipe = pipeline(model=model_name, tokenizer=model_name,
                        task="feature-extraction")

        if not self.path_easy.exists():
            g1 = generator(
                "data/SimpleWikipedia/sentence-aligned.v2/simple.aligned", model_name)
            # use CLS embedding as sentence embedding
            easy_embeddings = [i[0][0] for i in tqdm(
                pipe(g1), desc="Wikipedia Easy embeddings")]
            t = torch.tensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator(
                "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned", model_name)
            # use CLS embedding as sentence embedding
            normal_embeddings = [i[0][0] for i in tqdm(
                pipe(g2), desc="Wikipedia Normal embeddings")]
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
            self.path_easy = Path(f"data/SimpleWikipedia/WordPieceEasy{max_length}.pt")
            self.path_normal = Path(f"data/SimpleWikipedia/WordPieceNormal{max_length}.pt")
            self.str = "SimpleWikipediaCorpus"
            model_name = "bert-base-uncased"

        else:
            self.path_easy_input = Path("data/SimpleGerman/fixed_easy.txt")
            self.path_normal_input = Path("data/SimpleGerman/fixed_normal.txt")
            self.path_easy = Path(f"data/SimpleGerman/WordPieceEasy{max_length}.pt")
            self.path_normal = Path(f"data/SimpleGerman/WordPieceNormal{max_length}.pt")
            self.str = "SimpleGermanCorpus"
            model_name = "deepset/gbert-base"

        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer.get_vocab())
        # Save symbols for start, pad and end. Needed for trainer
        self.CLS, self.PAD, self.SEP = self.tokenizer.encode("[PAD]")

        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        t_easy = torch.load(self.path_easy).long()
        t_normal = torch.load(self.path_normal).long()

        self.embeddings = torch.cat(
            [t_easy, t_normal])
        self.labels = torch.cat(
            [torch.zeros(t_easy.shape[0]), torch.ones(t_normal.shape[0])]).view((-1, 1))

    def __str__(self):
        return self.str

    def createDataset(self):
        def generator(file):
            with open(file) as fp:
                if self.large:
                    lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
                else:
                    lines = [i.strip() for i in fp.readlines()]
            for line in lines:
                line_token = self.tokenizer.encode(line, padding="max_length")
                if len(line_token) > self.max_length:
                    continue
                tokens = line_token[:self.max_length-1]
                tokens.append(line_token[-1])
                yield tokens

        if not self.path_easy.exists():
            g1 = generator(self.path_easy_input)
            easy_embeddings = [i for i in tqdm(g1, desc=f"Easy {self}")]
            t = torch.LongTensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator(self.path_normal_input)
            normal_embeddings = [i for i in tqdm(g2, desc=f"Normal {self}")]
            t = torch.LongTensor(normal_embeddings)
            torch.save(t, self.path_normal)
