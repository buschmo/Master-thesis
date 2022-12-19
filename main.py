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
                trainer = NaiveTrainer(
                    dataset=dataset,
                    model=model,
                    checkpoint_index=0,
                    use_reg_loss=use_reg_loss,
                    timestamp=timestamp
                )

                trainer.train_model(
                    batch_size=64,
                    num_epochs=num_epochs
                )

    for Dataset in [SimpleGermanDatasetWordPiece, SimpleWikipediaDatasetWordPiece]:
        # continue
        dataset = Dataset()
        for num_epochs in [25]:
            for use_reg_loss in [True, False]:
                ts = time.time()
                timestamp = datetime.datetime.fromtimestamp(ts).strftime(
                    '%Y-%m-%d_%H:%M:%S'
                )
                model = TVAE(
                    ntoken=dataset.vocab_size,
                    d_model=256,
                    z_dim=64,
                    nhead_encoder=8,
                    nhead_decoder=8,
                    d_hid=512,
                    nlayers=1,
                    dropout=0.1,
                    use_gru=False,
                    foldername=dataset.__str__(),
                    timestamp=timestamp)
                trainer = TVAETrainer(
                    dataset=dataset,
                    model=model,
                    checkpoint_index=1,
                    use_reg_loss=use_reg_loss,
                    timestamp=timestamp
                )

                trainer.train_model(
                    batch_size=8,
                    num_epochs=num_epochs
                )


if __name__ == "__main__":
    main()
