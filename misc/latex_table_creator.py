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

try:
    PATH_FIGURES = Path(os.environ["PATH_FIGURES"])
except KeyError:
    PATH_FIGURES = Path()

try:
    PATH_TABLES = Path(os.environ["PATH_TABLES"])
except KeyError:
    PATH_TABLES = Path("tables.json")

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

# keys = [
#     "Disentanglement_Interpretability Mean",
#     "Disentanglement_Interpretability Simplicity",
#     "Disentanglement_Mutual Information Gap",
#     "Disentanglement_Separated Attribute Predictability",
#     "Disentanglement_Spearman's Rank Correlation",
#     "accuracy_training",
#     "accuracy_validation",
#     "loss_KLD_training",
#     "loss_KLD_validation",
#     "loss_KLD_unscaled_training",
#     "loss_KLD_unscaled_validation",
#     "loss_reconstruction_training",
#     "loss_reconstruction_validation",
#     "loss_regularization_training",
#     "loss_regularization_validation",
#     "loss_sum_training",
#     "loss_sum_validation"
# ]

keys = {
    "Disentanglement_Interpretability_Mean": "Interpretability Mean",
    "Disentanglement_Interpretability_Simplicity": "Interpretability Simplicity",
    "Disentanglement_Mutual_Information_Gap": "Mutual Information Gap",
    "Disentanglement_Separated_Attribute_Predictability": "Separated Attribute Predictability",
    "Disentanglement_Spearman's_Rank_Correlation": "Spearman's Rank Correlation",
    "accuracy_training": "Accuracy on training set",
    "accuracy_validation": "Accuracy on validation set",
    "loss_KLD_training": "KLD loss on training set",
    "loss_KLD_validation": "KLD loss on validation set",
    "loss_KLD_unscaled_training": "KLD loss (unscaled) on training set",
    "loss_KLD_unscaled_validation": "KLD loss (unscaled) on validation set",
    "loss_reconstruction_training": "Reconstruction loss on training set",
    "loss_reconstruction_validation": "Reconstruction loss on validation set",
    "loss_regularization_training": "Regularization loss on training set",
    "loss_regularization_validation": "Regularization loss on validation set",
    "loss_sum_training": "Loss on training set",
    "loss_sum_validation": "Loss on validation set"
}

ylabels = {
    "Disentanglement_Interpretability_Mean": "Score",
    "Disentanglement_Interpretability_Simplicity": "Score",
    "Disentanglement_Mutual_Information_Gap": "Score",
    "Disentanglement_Separated_Attribute_Predictability": "Score",
    "Disentanglement_Spearman's_Rank_Correlation": "Score",
    "accuracy_training": "Score",
    "accuracy_validation": "Score",
    "loss_KLD_training": "Loss",
    "loss_KLD_validation": "Loss",
    "loss_KLD_unscaled_training": "Loss",
    "loss_KLD_unscaled_validation": "Loss",
    "loss_reconstruction_training": "Loss",
    "loss_reconstruction_validation": "Loss",
    "loss_regularization_training": "Loss",
    "loss_regularization_validation": "Loss",
    "loss_sum_training": "Loss",
    "loss_sum_validation": "Loss"
}

# LaTeX table


def add_plot(path, options=[""], key="accuracy_training", dots=False):
    if dots:
        line = "dotted"
    else:
        line = "solid"
    options_str = f",\n{' '*8}".join(options+[line])
    plot = f"    \\addplot[\n        {options_str}\n        ]\n        table[x=step, y={key}] {{{str(path.relative_to(PATH_FIGURES.parent))}}};\n"
    return plot


def add_figures(path_list, options=[""], key="accuracy_training", legendentry="LEGEND"):
    axis = ""
    for i, path in enumerate(path_list):
        if i < 1:
            axis += add_plot(path, options, key)
            # axis += f"    \\addlegendentry{{{legendentry}}}\n"
        else:
            axis += add_plot(path, options, key, dots=True)
            # axis += "    \\addlegendentry{}\n"
    return axis


def get_figure_path(fig_label, key):
    fig_label = fig_label.replace(" ", "_")
    path_figure = Path(PATH_FIGURES, f"{fig_label}/{key}.tex")
    if not path_figure.parent.exists():
        path_figure.parent.mkdir(parents=True)
    return path_figure


def get_csv_path(fig_label, legend):
    fig_label = fig_label.replace(" ", "_")
    legend = legend.replace(" ", "_")
    path_csv = Path(PATH_FIGURES, f"data/{fig_label}_{legend}.csv")
    if not path_csv.parent.exists():
        path_csv.parent.mkdir(parents=True)
    return path_csv


def merge(paths, path_csv):
    path_mean = path_csv.with_stem(path_csv.stem+"_mean")
    path_min = path_csv.with_stem(path_csv.stem+"_min")
    path_max = path_csv.with_stem(path_csv.stem+"_max")
    if not (path_mean.exists() and path_min.exists() and path_max.exists()):
        dfs = [pd.read_csv(path, sep="\t", index_col="step") for path in paths]
        mean = pd.concat(dfs).groupby(level=0).mean()
        min = pd.concat(dfs).groupby(level=0).min()
        max = pd.concat(dfs).groupby(level=0).max()

        mean.to_csv(path_mean, sep="\t")
        min.to_csv(path_min, sep="\t")
        max.to_csv(path_max, sep="\t")
    return [path_mean, path_min, path_max]


def make_picture(tables):
    for fig_label, figure in tables.items():
        for key, title in keys.items():
            
            figures = []
            legend_str = ""
            path_figure = get_figure_path(fig_label, key)
            for legend, axis in figure["Axis"].items():
                path_csv = get_csv_path(fig_label, legend)
                options = axis.get("Options", [""])
                path_df = axis["Path"]
                if isinstance(path_df, list) and len(path_df)>1:
                    path_df = merge(path_df, path_csv)
                else:
                    if isinstance(path_df, list):
                        path_df = path_df[0]
                    path_df = Path(path_df)
                    path_csv = path_csv.with_stem(path_df.stem)
                    copyfile(path_df, path_csv)
                    path_df = [path_csv]
                legend_str += legend
                legend_str += ","*len(path_df)

                figures.append(add_figures(path_df, options, key, legend))
            
            xlabel = figure.get("xlabel", "epochs")
            ylabel = ylabels[key]
            xmax = figure.get("xmax", "50")
            ymax = figure.get("ymax", "% ymax=1")
            xtick = figure.get("xtick", "{0,10,20,30,40,50}")
            ytick = figure.get("ytick", "% ytick={0,0.2,0.4,0.6,0.8,1}")
            if not ymax.startswith("%"):
                ymax = f"ymax={ymax}"
            if not ytick.startswith("%"):
                ytick = f"ytick={ytick}"
            # legend_pos = figure.get("legend pos", "north west")

            figure_str = f"\\begin{{tikzpicture}}\n    \\begin{{axis}}[\n        xlabel={xlabel},\n        ylabel={ylabel},\n        xmin=0, xmax={xmax},\n        ymin=0, {ymax},\n        xtick={xtick},\n        {ytick},\n        legend entries={{{legend_str}}},\n        legend to name={{legend:{fig_label}_{key}}},\n        ymajorgrids=true,\n        grid style=dashed\n    ]\n\n"

            figure_str += "".join(figures)

            # legend_str = ",".join(figure["Axis"].keys())
            # figure_str += f"    \\legend{{{legend_str}}}\n"
            figure_str += "    \\end{axis}\n\\end{tikzpicture}"
            with open(path_figure, "w") as fp:
                fp.write(figure_str)


def main():
    # load data frame, calculate boxplot / mean, output it into (single) named thingy
    with open(PATH_TABLES) as fp:
        tables = json.load(fp)
    make_picture(tables)


if __name__ == "__main__":
    main()
