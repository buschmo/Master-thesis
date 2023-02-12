from tensorboard.backend.event_processing import event_accumulator
import tensorboard
from pathlib import Path
from multiprocessing import Pool
import os
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

os.chdir(Path(os.environ["MASTER"]))

"""
    Taken from https://gist.github.com/laszukdawid/62656cf7b34cac35b325ba21d46ecfcd
    https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
"""

def convert_tfevent(filepath):
    return pd.DataFrame([
        parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
    ])

def parse_tfevent(tfevent):
    return dict(
        wall_time=tfevent.wall_time,
        name=tfevent.summary.value[0].tag,
        step=tfevent.step,
        value=float(tfevent.summary.value[0].simple_value),
    )

def convert_tb_data(root_dir, sort_by=None):
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
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)

def parallel(input_dir, output_dir):
    df = convert_tb_data(input_dir)
    for name in set(df["name"]):
        output_file = Path(output_dir, name.replace("/", "_")+".csv")
        values = list(map(lambda x: (x[2],x[3]), df[df["name"] == name].to_numpy()))
        if not output_file.exists():
            with open(output_file, "w") as fp:
                fp.write(str(values)[1:-1])

if __name__ == "__main__":
    root_path = Path(os.environ["MASTER"], "save")
    dirs = [dir for dir in root_path.iterdir()]
    
    l = []
    for dir in dirs:
        for input_dir in Path(dir, "runs").iterdir():
            output_dir = Path(os.environ["MASTER"], "results", dir.name, input_dir.name)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            l.append((input_dir, output_dir))
    with Pool() as pool:
        pool.starmap(parallel, l)