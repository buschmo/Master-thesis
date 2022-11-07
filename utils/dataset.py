import numpy as np
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from transformers import logging
logging.set_verbosity_error()

# os.chdir(Path(os.environ["ALTHOME"])/"Master-thesis")

path_easy = Path("data/df_easy.csv")
path_normal = Path("data/df_normal.csv")

def generator(file):
    with open(file) as fp:
        lines = [i.strip() for i in fp.readlines()]
    for line in lines:
        yield line


if not path_easy.exists() or not path_normal.exists():
    # model_name = "dbmdz/bert-base-german-cased"
    model_name = "deepset/gbert-base"
    pipe = pipeline(model=model_name, tokenizer=model_name,
                    task="feature-extraction")

    if not path_easy.exists():
        g1 = generator("data/fixed_easy.txt")
        # use CLS embedding as sentence embedding
        easy = [i[0][0] for i in tqdm(pipe(g1), desc="Easy")]
        df = pd.DataFrame(easy)
        df.to_csv("data/df_easy.csv")
    if not path_normal.exists():
        g2 = generator("data/fixed_normal.txt")
        # use CLS embedding as sentence embedding
        normal = [i[0][0] for i in tqdm(pipe(g2), desc="Normal")]
        df = pd.DataFrame(normal)
        df.to_csv("data/df_normal.csv")


class TextDataset(Dataset):
    def __init__(self):
        df_easy = pd.read_csv(path_easy, index_col=0)
        df_normal = pd.read_csv(path_normal, index_col=0)
        labels = np.concatenate([np.zeros(df_easy.shape[0]), np.ones(df_normal.shape[0])])

        self.embeddings = pd.concat([df_easy,df_normal]).to_numpy(dtype="float32")
        self.labels = pd.DataFrame(labels).to_numpy(dtype="float32")
        
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]

    def getInputSize(self):
        return self.embeddings.shape[1]