import torch

from code.naive_model import NaiveVAE
from utils.dataset import TextDataset
from utils.trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    if device == "cpu":
        print("Cuda was not found.")
        return
    else:
        print("Cuda was found.")

    dataset = TextDataset()
    model = NaiveVAE(input_size=dataset.getInputSize())
    trainer = Trainer(dataset, model)

    trainer.train_model(
        batch_size=64,
        num_epochs=100
    )


if __name__ == "__main__":
    main()
