import click
import spacy
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BertTokenizer


def text_replace(s, replacements):
    for k, v in replacements.items():
        s = s.replace(v, k)
    return s


def convert_umlaut():
    """
        Some sentences in the Simple German dataset have encoding error.
        German "Umlaute" are not parsed correctly. This method fixes that.
    """
    l = ["Ä", "ä", "Ö", "ö", "Ü", "ü", "ß"]
    replacements = {
        key: str(bytes(key, encoding="utf-8"), encoding="latin-1") for key in l}
    with open("../data/SimpleGerman/spacy_easy.txt") as fp:
        text = fp.read()
    text = text_replace(text, replacements)
    with open("../data/SimpleGerman/fixed_easy.txt", "w") as fp:
        fp.write(text)
    with open("../data/SimpleGerman/spacy_normal.txt") as fp:
        text = fp.read()
    text = text_replace(text, replacements)
    with open("../data/SimpleGerman/fixed_normal.txt", "w") as fp:
        fp.write(text)


def get_lines(file, wiki=False):
    with open(file) as fp:
        if wiki:
            lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
        else:
            lines = [i.strip() for i in fp.readlines()]
    lines = list(map(lambda line: re.sub(
        r"(?<=[a-zA-ZÄäÖöÜü])-(?=[a-zA-ZÄäÖöÜü])", r"", line), lines))
    return lines


def filter_lines(line, tokenizer):
    _, PAD, SEP = tokenizer.encode("[PAD]")
    line_tokens = tokenizer.encode(line, padding="max_length")
    if line_tokens[127] not in [PAD, SEP]:
        # The sequence is longer than 128 tokens
        return False
    return True


def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def create_attribute_file(paths, path_output, nlp, wiki=False):
    if wiki:
        model_name = "bert-base-uncased"
    else:
        model_name = "deepset/gbert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    lines = []
    l_simplicity = []
    for i, path in enumerate(paths):
        new_lines = get_lines(path, wiki=wiki)
        new_lines = [
            line for line in tqdm(new_lines, desc="Filter list") if filter_lines(line, tokenizer)]
        lines.extend(new_lines)
        l_simplicity += [i] * len(new_lines)
    # remove hyphens
    docs = nlp.pipe(lines)

    l_depth = []
    l_pos = []
    l_len = []
    for doc in tqdm(docs, total=len(lines), desc="Docs"):
        depths = map(lambda x: walk_tree(x.root, 0), doc.sents)
        depth = max(depths)
        l_depth.append(depth)
        l_pos.append(len(set(map(lambda token: token.pos_, doc))))
        l_len.append(len(doc))

    vectorizer = TfidfVectorizer(norm=None)
    X = vectorizer.fit_transform(lines)
    l_tfidf = []
    for x in tqdm(X, total=len(lines), desc="TF-IDF"):
        x = x.A
        x[x == 0] = np.NaN
        if np.isnan(x).all():
            l_tfidf.append(0)
        else:
            l_tfidf.append(np.nanquantile(x, .75))

    # add simplicity attribute
    lists = [l_simplicity, l_depth, l_pos, l_len, l_tfidf]
    # save as torch tensors
    tensors = [torch.tensor(l) for l in lists]
    tensor = torch.stack(tensors, dim=1)

    if not path_output.parent.exists():
        path_output.parent.mkdir(parents=True)
    torch.save(tensor, path_output)


def create_attribute_files():
    nlp_wiki = spacy.load("en_core_web_lg")
    nlp_ger = spacy.load("de_core_news_lg")

    paras = [
        [
            [
                Path("data/SimpleWikipedia/sentence-aligned.v2/simple.aligned"),
                Path("data/SimpleWikipedia/sentence-aligned.v2/normal.aligned")
            ],
            Path("data/SimpleWikipedia/attributes.pt"),
            nlp_wiki,
            True
        ],
        [
            [
                Path("data/SimpleGerman/fixed_easy.txt"),
                Path("data/SimpleGerman/fixed_normal.txt")
            ],
            Path("data/SimpleGerman/attributes.pt"),
            nlp_ger,
            False
        ]
    ]
    for para in tqdm(paras, desc="Sets"):
        create_attribute_file(*para)


@ click.command()
@ click.option("-u", "umlaut", is_flag=True, default=False, help="Convert German 'Umlaute' from latin-1 to UTF-8")
@ click.option("-a", "attributes", is_flag=True, default=False, help="Create attributes files")
def main(umlaut, attributes):
    if umlaut:
        convert_umlaut()
    if attributes:
        create_attribute_files()


if __name__ == "__main__":
    main()
