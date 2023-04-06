from tensorboard.backend.event_processing import event_accumulator
import tensorboard
from pathlib import Path
from multiprocessing import Pool
import os
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.chdir("..")

""" convert_tfevent, parse_tfevent, convert_tb_data
    Taken from https://gist.github.com/laszukdawid/62656cf7b34cac35b325ba21d46ecfcd
    https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
"""


def convert_tfevent(filepath, name_suffix=""):
    try:
        return pd.DataFrame([
            parse_tfevent(e, name_suffix) for e in summary_iterator(filepath) if len(e.summary.value)
        ])
    except tf.errors.DataLossError as err:
        print(f"Error with file:\n\t{filepath}")
        raise err


def parse_tfevent(tfevent, name_suffix=""):
    name = f"{tfevent.summary.value[0].tag} {name_suffix}" if name_suffix else tfevent.summary.value[0].tag
    return dict(
        wall_time=tfevent.wall_time,
        name=name,
        step=tfevent.step,
        value=float(tfevent.summary.value[0].simple_value),
    )


def convert_tb_data(root_dir):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    columns_order = ['wall_time', 'name', 'step', 'value']

    out = []
    for (root, folders, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue

            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))
            for folder in folders:
                name_suffix = folder.split("_")[-1]
                for path in Path(root, folder).iterdir():
                    out.append(convert_tfevent(str(path), name_suffix))
            break
        else:
            continue
        break

    # Concatenate (and sort) all partial individual dataframes
    df = pd.concat(out)[columns_order]

    epoch_dict = {}
    batch_training_dict = {}
    batch_validation_dict = {}
    for name in sorted(set(df["name"])):
        key = name.replace("/", "_")
        key = key.replace(" ", "_")
        values = list(df[df["name"] == name]["value"])
        if "batchwise" in name:
            if "training" in name:
                batch_training_dict[key] = values
            else:
                batch_validation_dict[key] = values
        else:
            epoch_dict[key] = values
    try:
        epoch_df = pd.DataFrame.from_dict(epoch_dict)
    except ValueError:
        print(f"Error in dir {root_dir}, while parsing normal set")
        print(list(map(lambda x: f"{x[0]} {len(x[1])}", epoch_dict.items())))
        exit()
    try:
        batch_training_df = pd.DataFrame.from_dict(batch_training_dict)
    except ValueError:
        print(f"Error in dir {root_dir}, while parsing batch training set")
        print(
            list(map(lambda x: f"{x[0]} {len(x[1])}", batch_training_dict.items())))
        exit()
    try:
        batch_validation_df = pd.DataFrame.from_dict(batch_validation_dict)
    except ValueError:
        print(f"Error in dir {root_dir}, while parsing batch validation set")
        print(
            list(map(lambda x: f"{x[0]} {len(x[1])}", batch_validation_dict.items())))
        exit()

    epoch_df.index += 1
    batch_training_df.index += 1
    batch_validation_df.index += 1

    return epoch_df, batch_training_df, batch_validation_df


def parallel(input_dir, output_file):
    epoch_df, batch_training_df, batch_validation_df = convert_tb_data(
        input_dir)

    epoch_df.to_csv(output_file, sep="\t", index_label="step")
    if not (batch_training_df.empty or batch_validation_df.empty):
        batch_training_df.to_csv(output_file.with_stem(
            output_file.stem + "_batchwise_training"), sep="\t", index_label="step")
        batch_validation_df.to_csv(output_file.with_stem(
            output_file.stem + "_batchwise_validation"), sep="\t", index_label="step")


def get_file_list():
    root_path = Path(os.environ["MASTER"], "save")
    dirs = [dir for dir in root_path.iterdir() if dir.name.startswith("20")]

    l = []
    for dir in dirs:
        for input_dir in Path(dir, "runs").iterdir():
            l.extend(input_dir.iterdir())
    return l


if __name__ == "__main__":
    input_dirs = get_file_list()

    l = []
    for input_dir in input_dirs:
        output_file = Path(os.environ["MASTER"],
                           "csv", input_dir.parents[2].name, f"{'_'.join(input_dir.parent.name.split('_')[:2])}_{input_dir.name}.csv")
        if output_file.exists():
            continue
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        l.append((input_dir, output_file))
    with Pool() as pool:
        pool.starmap(parallel, l)
