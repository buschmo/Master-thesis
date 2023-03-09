import click
import spacy
import torch
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import re
from itertools import chain


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
    return lines


def walk_tree(node, depth):
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    else:
        return depth


def create_attribute_file(path, nlp, wiki=False):
    if output.exists():
        return
    lines = get_lines(path, wiki=wiki)
    # remove hyphens
    lines = map(lambda line: re.sub(r"(\w)-(\w)", r"\1\2", line), lines)
    docs = nlp.pipe(lines)

    l_depth = []
    l_pos = []
    for doc in tqdm(docs, desc="Docs"):
        depths = map(lambda x: walk_tree(x.root, 0), doc.sents)
        depth = max(depths)
        l_depth.append(depth)
        l_pos.append(len(set(map(lambda token: token.pos_, chain(doc.sents)))))

    # save as torch tensors
    saving = [
        [Path(path.parents[-2], path.stem + "_depth.pt"), l_depth],
        [Path(path.parents[-2], path.stem + "_pos.pt"), l_pos]
    ]
    for output, content in saving:
        if not output.parent.exists():
            output.parent.mkdir(parents=True)
        tensor = torch.tensor(content)
        torch.save(tensor, output)


def create_attribute_files():
    nlp_wiki = spacy.load("en_core_web_lg")
    nlp_ger = spacy.load("de_core_news_lg")

    paras = [
        [Path(
            "data/SimpleWikipedia/sentence-aligned.v2/simple.aligned"), nlp_wiki, True],
        [Path(
            "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned"), nlp_wiki, True],
        [Path("data/SimpleGerman/fixed_easy.txt"), nlp_ger, False],
        [Path("data/SimpleGerman/fixed_normal.txt"), nlp_ger, False]
    ]
    for para in tqdm(paras, desc="Sets"):
        create_attribute_file(*para)


@click.command()
@click.option("-u", "umlaut", is_flag=True, default=False)
@click.option("-s", "spacy", is_flag=True, default=False)
def main(umlaut, spacy):
    if umlaut:
        convert_umlaut()
    if spacy:
        create_attribute_files()


if __name__ == "__main__":
    main()
