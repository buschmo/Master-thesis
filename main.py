import torch
import datetime
import time

from models.naive_model import NaiveVAE
from models.naive_trainer import NaiveTrainer
from models.tvae_model import TVAE
from models.tvae_trainer import TVAETrainer
from utils.datasets import SimpleGermanDatasetBERT, SimpleWikipediaDatasetBERT, SimpleGermanDatasetWordPiece, SimpleWikipediaDatasetWordPiece

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if device == "cpu":
        print("Cuda was not found.")
        return
    else:
        print("Cuda was found.")

    for Dataset in [SimpleGermanDatasetBERT, SimpleWikipediaDatasetBERT]:
        continue
        dataset = Dataset()
        for num_epochs in [1000]:
            for use_reg_loss in [True, False]:
                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                    '%Y-%m-%d_%H:%M:%S'
                )
                model = NaiveVAE(input_size=dataset.getInputSize(
                ), foldername=dataset.__str__(), timestamp=timestamp)
                trainer = NaiveTrainer(dataset=dataset, model=model, checkpoint_index=0,
                                       use_reg_loss=use_reg_loss, timestamp=timestamp)

                trainer.train_model(
                    batch_size=64,
                    num_epochs=num_epochs
                )

    for Dataset in [SimpleGermanDatasetWordPiece, SimpleWikipediaDatasetWordPiece]:
        dataset = Dataset()
        for num_epochs in [1000]:
            for use_reg_loss in [True, False]:
                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                    '%Y-%m-%d_%H:%M:%S'
                )
                model = TVAE(
                    ntoken=dataset.getInputSize(),
                    foldername=dataset.__str__(),
                    timestamp=timestamp)
                trainer = TVAETrainer(
                    dataset=dataset,
                    model=model,
                    checkpoint_index=0,
                    use_reg_loss=use_reg_loss,
                    timestamp=timestamp
                )

                trainer.train_model(
                    batch_size=32,
                    num_epochs=num_epochs
                )


if __name__ == "__main__":
    main()
