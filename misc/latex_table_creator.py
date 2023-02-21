from pathlib import Path
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pprint
import json
from shutil import copyfile

p = pprint.PrettyPrinter(indent=4)
os.chdir(Path(os.environ["MASTER"]))


all_keys = [
    'Disentanglement/Interpretability',
    'Disentanglement/Mutual Information Gap',
    'Disentanglement/Separated Attribute Predictability',
    "Disentanglement/Spearman's Rank Correlation",
    'accuracy/training',
    'accuracy/validation',
    'loss_KLD/training',
    'loss_KLD/validation',
    'loss_KLD_batchwise/training',
    'loss_KLD_batchwise/validation',
    'loss_KLD_unscaled/training',
    'loss_KLD_unscaled/validation',
    'loss_KLD_unscaled_batchwise/training',
    'loss_KLD_unscaled_batchwise/validation',
    'loss_reconstruction/training',
    'loss_reconstruction/validation',
    'loss_regularization/training',
    'loss_regularization/validation',
    'loss_sum/training',
    'loss_sum/validation'
]

keys = [
    "Disentanglement_Interpretability Mean",
    "Disentanglement_Interpretability Simplicity",
    "Disentanglement_Mutual Information Gap",
    "Disentanglement_Separated Attribute Predictability",
    "Disentanglement_Spearman's Rank Correlation",
    "accuracy_training",
    "accuracy_validation",
    "loss_KLD_training",
    "loss_KLD_validation",
    "loss_KLD_unscaled_training",
    "loss_KLD_unscaled_validation",
    "loss_reconstruction_training",
    "loss_reconstruction_validation",
    "loss_regularization_training",
    "loss_regularization_validation",
    "loss_sum_training",
    "loss_sum_validation"
]

# LaTeX table


def add_plot(path, colour="blue", key="accuracy_training", dots=False):
    if dots:
        mark = "dots"
    else:
        mark = "line"
    plot = f"    \\addplot[\n        color={colour},\n        mark={mark},\n        ]\n        table[x=step, y={key}] {{{str(path)}}};\n"
    return plot


def add_figures(path_list, colour="blue", key="accuracy_training"):
    axis = ""
    for i, path in enumerate(path_list):
        if i < 1:
            axis += add_plot(path, colour, key)
        else:
            axis += add_plot(path, colour, key, dots=True)
    return axis


def get_figure_path(title, key):
    title = title.replace(" ", "_")
    path_figure = Path(f"figures/{title}/{key}.tex")
    path_csv = Path(f"figures/data/{title}.csv")
    if not path_figure.parent.exists():
        path_figure.parent.mkdir(parents=True)
    if not path_csv.parent.exists():
        path_csv.parent.mkdir(parents=True)
    return path_figure, path_csv


def merge(paths, path_csv):
    paths = [pd.read_csv(path, sep="\t", index_col="step") for path in paths]
    mean = pd.concat(paths).groupby(level=0).mean()
    min = pd.concat(paths).groupby(level=0).min()
    max = pd.concat(paths).groupby(level=0).max()

    path_mean = path_csv.with_stem(path_csv.stem+"_mean")
    path_min = path_csv.with_stem(path_csv.stem+"_min")
    path_max = path_csv.with_stem(path_csv.stem+"_max")
    mean.to_csv(path_mean, sep="\t")
    min.to_csv(path_min, sep="\t")
    max.to_csv(path_max, sep="\t")
    return [path_mean, path_min, path_max]


def make_picture(tables):
    for title, figure in tables.items():
        for key in keys:
            path_figure, path_csv = get_figure_path(title, key)
            description = figure["Description"]

            xlabel = figure.get("xlabel", "epochs")
            ylabel = figure.get("ylabel", "value")
            xmax = figure.get("xmax", "100")
            ymax = figure.get("ymax", "1")
            xtick = figure.get("xtick", "{0,20,40,60,80,100}")
            ytick = figure.get("ytick", "{0,0.2,0.4,0.6,0.8,1}")
            legend_pos = figure.get("legend pos", "north west")

            figure_str = f"\\begin{{tikzpicture}}\n    \\begin{{axis}}[\n        title={title},\n        xlabel={xlabel},\n        ylabel={ylabel},\n        xmin=0, xmax={xmax},\n        ymin=0, ymax=1,\n        xtick={xtick},\n        %ytick={ytick},\n        legend pos={legend_pos},\n        ymajorgrids=true,\n        grid style=dashed,\n    ]\n\n"
            for axis in figure["Axis"].values():
                colour = axis.get("Axis", "blue")
                path_df = axis["Path"]
                if isinstance(path_df, list):
                    path_df = merge(path_df, path_csv)
                else:
                    copyfile(path_df, path_csv)
                    path_df = [path_csv]

                figure_str += add_figures(path_df, colour, key)

            legend = ",".join(figure["Axis"].keys())
            figure_str += f"    \\legend{{{legend}}}\n    \\end{{axis}}\n\\end{{tikzpicture}}"
            with open(path_figure, "w") as fp:
                fp.write(figure_str)


def main():
    # load data frame, calculate boxplot / mean, output it into (single) named thingy
    with open("tables.json") as fp:
        tables = json.load(fp)
    make_picture(tables)


if __name__ == "__main__":
    main()
