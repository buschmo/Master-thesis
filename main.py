import torch

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
    dataset = SimpleGermanDataset()
    model = NaiveVAE(input_size=dataset.getInputSize(), filename=dataset.__str__())
    trainer = Trainer(dataset, model, checkpoint_index=50)

    trainer.train_model(
        batch_size=64,
        num_epochs=500
    )


if __name__ == "__main__":
    main()
