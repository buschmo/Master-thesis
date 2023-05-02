import os
from pathlib import Path
# Switch to correct folder
if not "__path__" in locals():
    os.chdir("..")
    __path__ = Path().absolute()

from tqdm import tqdm
import torch
from transformers import pipeline, BertTokenizer
    

def main():
    for large in tqdm([False,True], desc="Models"):
        if large:
            path_easy = Path(
                "data/SimpleWikipedia/sentence-aligned.v2/simple.aligned")
            path_normal = Path(
                "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned")
            model_name = "bert-base-uncased"
        else:
            path_easy = Path("data/SimpleGerman/fixed_easy.txt")
            path_normal = Path("data/SimpleGerman/fixed_normal.txt")
            model_name = "deepset/gbert-base"
            
        tokenizer = BertTokenizer.from_pretrained(model_name)
        CLS, PAD, SEP = tokenizer.encode("[PAD]")
        over_max_length = 0
        for file in [path_easy, path_normal]:
            with open(file) as fp:
                if large:
                    lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
                else:
                    lines = [i.strip() for i in fp.readlines()]
                for line in tqdm(lines, leave=False, desc="Lines"):
                    line_tokens = tokenizer.encode(line, padding="max_length")
                    if line_tokens[128-1] not in [PAD, SEP]:
                        # content is over max_length
                        over_max_length += 1
        name = "Wikipedia" if large else "German"
        print(f"{name}: {over_max_length}")
    
if __name__ == "__main__":
    main()