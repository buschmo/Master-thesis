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
        super.__init__()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

    def __str__(self):
        raise NotImplementedError

    def getInputSize(self):
        return self.embeddings.shape[1]


class SimpleGermanDatasetBERT(BaseDataset):
    def __init__(self):
        self.path_easy = Path("data/SimpleGerman/easy.csv")
        self.path_normal = Path("data/SimpleGerman/normal.csv")
        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        df_easy = pd.read_csv(self.path_easy, index_col=0)
        df_normal = pd.read_csv(self.path_normal, index_col=0)
        labels = np.concatenate(
            [np.zeros(df_easy.shape[0]), np.ones(df_normal.shape[0])])

        self.embeddings = pd.concat(
            [df_easy, df_normal]).to_numpy(dtype="float32")
        self.labels = pd.DataFrame(labels).to_numpy(dtype="float32")

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
            easy = [i[0][0] for i in tqdm(pipe(g1), desc="Easy")]
            df = pd.DataFrame(easy)
            df.to_csv(self.path_easy)
        if not self.path_normal.exists():
            g2 = generator("data/SimpleGerman/fixed_normal.txt")
            # use CLS embedding as sentence embedding
            normal = [i[0][0] for i in tqdm(pipe(g2), desc="Normal")]
            df = pd.DataFrame(normal)
            df.to_csv(self.path_normal)


class SimpleWikipediaDatasetBERT(BaseDataset):
    def __init__(self):
        self.path_easy = Path("data/SimpleWikipedia/easy.csv")
        self.path_normal = Path("data/SimpleWikipedia/normal.csv")
        if not self.path_easy.exists() or not self.path_normal.exists():
            self.createDataset()
        df_easy = pd.read_csv(self.path_easy, index_col=0)
        df_normal = pd.read_csv(self.path_normal, index_col=0)
        labels = np.concatenate(
            [np.zeros(df_easy.shape[0]), np.ones(df_normal.shape[0])])

        self.embeddings = pd.concat(
            [df_easy, df_normal]).to_numpy(dtype="float32")
        self.labels = pd.DataFrame(labels).to_numpy(dtype="float32")

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
            easy = [i[0][0] for i in tqdm(pipe(g1), desc="Easy")]
            df = pd.DataFrame(easy)
            df.to_csv(self.path_easy)
        if not self.path_normal.exists():
            g2 = generator(
                "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned", model_name)
            # use CLS embedding as sentence embedding
            normal = [i[0][0] for i in tqdm(pipe(g2), desc="Normal")]
            df = pd.DataFrame(normal)
            df.to_csv(self.path_normal)
