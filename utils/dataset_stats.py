from transformers import BertTokenizer
import json
from pathlib import Path
from tqdm import tqdm
import spacy
import click

p_stats = Path("results/dataset_stats.json")
p_cul = Path("results/dataset_stats_cul.json")
p_tree = Path("results/datasets_parsetree_depth.json")

path_german_easy = Path("data/SimpleGerman/fixed_easy.txt")
path_german_normal = Path("data/SimpleGerman/fixed_normal.txt")

path_wiki_easy = Path(
    "data/SimpleWikipedia/sentence-aligned.v2/simple.aligned")
path_wiki_normal = Path(
    "data/SimpleWikipedia/sentence-aligned.v2/normal.aligned")


if not p_stats.parent.exists():
    p_stats.parent.mkdir()


def get_lines(file, wiki=False):
    with open(file) as fp:
        if wiki:
            lines = [i.split("\t")[-1].strip() for i in fp.readlines()]
        else:
            lines = [i.strip() for i in fp.readlines()]
    return lines


def tokenize(lines, tokenizer):
    result = {}
    for line in tqdm(lines, desc="Iteration"):
        encoding = tokenizer.encode(line)
        l = len(encoding)
        i = result.get(l, 0)
        result[l] = i+1

    result = {int(k): result[k] for k in sorted(result.keys())}
    return result


def calc_stats():
    results = {}

    model_name = "deepset/gbert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    lines = get_lines(path_german_easy)
    results["German Easy"] = tokenize(lines, tokenizer)
    lines = get_lines(path_german_normal)
    results["German Normal"] = tokenize(lines, tokenizer)
    results["German"] = {k: results["German Easy"].get(k, 0) + results["German Normal"].get(
        k, 0) for k in sorted(set(results["German Easy"]) | set(results["German Normal"]))}

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    lines = get_lines(path_wiki_easy, True)
    results["Wikipedia Easy"] = tokenize(lines, tokenizer)
    lines = get_lines(path_wiki_normal, True)
    results["Wikipedia Normal"] = tokenize(lines, tokenizer)
    results["Wikipedia"] = {k: results["Wikipedia Easy"].get(k, 0) + results["Wikipedia Normal"].get(
        k, 0) for k in sorted(set(results["Wikipedia Easy"]) | set(results["Wikipedia Normal"]))}

    with open(p_stats, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)


def load():
    if not p_stats:
        calc_stats()
    with open(p_stats) as fp:
        data = json.load(fp)
    result = {}
    for key, dic in data.items():
        result[key] = {}
        for k, v in dic.items():
            result[key][int(k)] = v
    return result


def culm_stats():
    data = load()
    results = {"German": {}, "Wikipedia": {}}
    i = 0
    for k in sorted(data["German"].keys(), reverse=True):
        i += data["German"][k]
        results["German"][k] = i
    i = 0
    for k in sorted(data["Wikipedia"].keys(), reverse=True):
        i += data["Wikipedia"][k]
        results["Wikipedia"][k] = i

    with open(p_cul, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)


def calc_tree_depth():
    def walk_tree(node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max(walk_tree(child, depth + 1) for child in node.children)
        else:
            return depth

    results = {"German": {}, "Wikipedia": {}}
    nlp = spacy.load("de_core_news_lg")
    lines = get_lines(path_german_easy) + get_lines(path_german_normal)
    n_sents = 0
    for line in tqdm(lines, desc="German Tree Depth"):
        doc = nlp(line)
        sents = [sent for sent in doc.sents]
        # if len(sents) > 1:
        #     print(doc)
        #     print(sents)
        #     return
        n_sents += len(sents)
        for sent in sents:
            node = sent.root
            depth = walk_tree(node, 0)
            results["German"][depth] = results["German"].get(depth,0) + 1
    print(f"There are {len(n_sents-lines)} more sents than lines")

    with open(p_tree, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

@click.command()
@click.option("-c", "--calc", is_flag=True, help="Calculate the stats")
@click.option("-cu", "--culm", is_flag=True, help="Show the cumulative stats")
@click.option("-t", "--tree-depth", "tree_depth", is_flag=True, help="Depth of the parse tree")
def main(calc, culm, tree_depth):
    if calc:
        calc_stats()
    if culm:
        culm_stats()
    if tree_depth:
        calc_tree_depth()


if __name__ == "__main__":
    main()
