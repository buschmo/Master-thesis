import torch
import datetime
import time

from code.naive_model import NaiveVAE
from utils.dataset import SimpleGermanDataset, SimpleWikipediaDataset
from utils.trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if device == "cpu":
        print("Cuda was not found.")
        return
    else:
        print("Cuda was found.")

    # dataset = SimpleWikipediaDataset()
    # dataset = SimpleGermanDataset()
    for Dataset in [SimpleGermanDataset, SimpleWikipediaDataset]:
        dataset = Dataset()
        for num_epochs in [100,1000]:
            ts = time.time()
            timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y-%m-%d_%H:%M:%S'
            )
            model = NaiveVAE(input_size=dataset.getInputSize(), foldername=dataset.__str__(), timestamp=timestamp)
            trainer = Trainer(dataset, model, checkpoint_index=50, timestamp=timestamp)

            trainer.train_model(
                batch_size=64,
                num_epochs=num_epochs
            )


if __name__ == "__main__":
    main()
