from transformers import BertTokenizer
import json
from pathlib import Path
from tqdm import tqdm
import click
p_stats = Path("dataset_stats.json")
p_cul = Path("dataset_stats_cul.json")
def f(file, tokenizer):
    result = {}
    with open(file) as fp:
        lines = [i.strip() for i in fp.readlines()]
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
    path_easy = Path("data/SimpleGerman/fixed_easy.txt")
    path_normal = Path("data/SimpleGerman/fixed_normal.txt")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    results["German Easy"] = f(path_easy, tokenizer)
    results["German Normal"] = f(path_normal, tokenizer)
    results["German"] = {k: results["German Easy"].get(k,0) + results["German Normal"].get(k,0) for k in sorted(set(results["German Easy"]) | set(results["German Normal"]))}


    model_name = "bert-base-uncased"
    path_easy = Path("data/SimpleWikipedia/sentence-aligned.v2/simple.aligned")
    path_normal = Path("data/SimpleWikipedia/sentence-aligned.v2/normal.aligned")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    results["Wikipedia Easy"] = f(path_easy, tokenizer)
    results["Wikipedia Normal"] = f(path_normal, tokenizer)
    results["Wikipedia"] = {k: results["Wikipedia Easy"].get(k,0) + results["Wikipedia Normal"].get(k,0) for k in sorted(set(results["Wikipedia Easy"]) | set(results["Wikipedia Normal"]))}

    with open(p_stats, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)

def load():
    if not p_stats:
        calc_stats()
    with open(p_stats) as fp:
        data = json.load(fp)
    result= {}
    for key,dic in data.items():
        result[key] = {}
        for k,v in dic.items():
            result[key][int(k)] = v
    return result

def culm_stats():
    data = load()
    results = {"German": {}, "Wikipedia": {}}    
    i = 0
    for k in sorted(data["German"].keys(), reverse=True):
        i+=data["German"][k]
        results["German"][k] = i
    i = 0
    for k in sorted(data["Wikipedia"].keys(), reverse=True):
        i+=data["Wikipedia"][k]
        results["Wikipedia"][k] = i
        
    with open(p_cul, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4)
    

@click.command()
@click.option("-c", "--calc", is_flag=True, help="Calculate the stats")
@click.option("-cu", "--culm", is_flag=True, help="Show the cumulative stats")
def main(calc, culm):
    if calc:
        calc_stats()
    if culm:
        culm_stats()
        
if __name__ == "__main__":
    main()