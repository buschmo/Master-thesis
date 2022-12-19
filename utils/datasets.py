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
            easy_embeddings = [i[0][0] for i in tqdm(pipe(g1), desc="German Easy embeddings")]
            t = torch.tensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator("data/SimpleGerman/fixed_normal.txt")
            # use CLS embedding as sentence embedding
            normal_embeddings = [i[0][0] for i in tqdm(pipe(g2), desc="German Normal embeddings")]
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
            easy_embeddings = [i[0][0] for i in tqdm(pipe(g1), desc="Wikipedia Easy embeddings")]
            t = torch.tensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator(
                "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned", model_name)
            # use CLS embedding as sentence embedding
            normal_embeddings = [i[0][0] for i in tqdm(pipe(g2), desc="Wikipedia Normal embeddings")]
            t=torch.tensor(normal_embeddings)
            torch.save(t, self.path_normal)


class SimpleGermanDatasetWordPiece(BaseDataset):
    def __init__(self):
        self.path_easy = Path("data/SimpleGerman/WordPieceEasy.pt")
        self.path_normal = Path("data/SimpleGerman/WordPieceNormal.pt")

        model_name = "deepset/gbert-base"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.CLS, self.PAD, self.SEP = self.tokenizer.encode("[PAD]") # start pad end symbols

        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        t_easy = torch.load(self.path_easy).long()
        t_normal = torch.load(self.path_normal).long()

        self.embeddings = torch.cat(
            [t_easy, t_normal])
        self.labels = torch.cat(
            [torch.zeros(t_easy.shape[0]), torch.ones(t_normal.shape[0])]).view((-1,1))

    def __str__(self):
        return "SimpleGermanCorpus"

    def createDataset(self):
        def generator(file):
            with open(file) as fp:
                lines = [i.strip() for i in fp.readlines()]
            for line in lines:
                line_token = self.tokenizer.encode(line, padding="max_length")
                if len(line_token) > 512:
                    continue
                yield line_token

        if not self.path_easy.exists():
            easy_embeddings = [i for i in tqdm(
                generator("data/SimpleGerman/fixed_easy.txt"), desc="Easy German")]
            t = torch.LongTensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            normal_embeddings = [i for i in tqdm(generator(
                "data/SimpleGerman/fixed_normal.txt"), desc="Normal German")]
            t = torch.LongTensor(normal_embeddings)
            torch.save(t, self.path_normal)


class SimpleWikipediaDatasetWordPiece(BaseDataset):
    def __init__(self):
        self.path_easy = Path("data/SimpleWikipedia/WordPieceEasy.pt")
        self.path_normal = Path("data/SimpleWikipedia/WordPieceNormal.pt")

        # English BERT from https://huggingface.co/bert-base-uncased
        model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.CLS, self.PAD, self.SEP = self.tokenizer.encode("[PAD]") # start pad end symbols

        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        t_easy = torch.load(self.path_easy).long()
        t_normal = torch.load(self.path_normal).long()

        self.embeddings = torch.cat(
            [t_easy, t_normal])
        self.labels = torch.cat(
            [torch.zeros(t_easy.shape[0]), torch.ones(t_normal.shape[0])]).view((-1,1))

    def __str__(self):
        return "SimpleWikipediaCorpus"

    def createDataset(self):
        def generator(file):
            with open(file) as fp:
                lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
            for line in lines:
                line_token = self.tokenizer.encode(line, padding="max_length")
                if len(line_token) > 512:
                    continue
                yield line_token

        if not self.path_easy.exists():
            g1 = generator(
                "data/SimpleWikipedia/sentence-aligned.v2/simple.aligned")
            # use CLS embedding as sentence embedding
            easy_embeddings = [i for i in tqdm(g1, desc="Wikipedia Easy")]
            t = torch.LongTensor(easy_embeddings)
            torch.save(t, self.path_easy)
        if not self.path_normal.exists():
            g2 = generator(
                "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned")
            # use CLS embedding as sentence embedding
            normal_embeddings = [i for i in tqdm(g2, desc="Wikipedia Normal")]
            t = torch.LongTensor(normal_embeddings)
            torch.save(t, self.path_normal)
