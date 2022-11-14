import torch
from torch import nn
from pathlib import Path


class BaseModel(nn.Module):
    
    def __init__(self, filepath=None, filename=None):
        super().__init__()
        if filepath:
            self.filepath = filepath
        else:
            self.filename = filename if filename else self.__str__()
            self.update_filepath()

    def __repr__(self):
        return self.state_dict()

    def __str__(self):
        raise NotImplementedError

    def update_filepath(self):
        dir_name = Path("checkpoints").absolute()
        self.filepath = Path(dir_name, self.__str__(), self.filename + ".pt")

    def save(self):
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True)

        torch.save(self.state_dict(), self.filepath)
        print(f"Model {self} saved")

    def save_checkpoint(self, epoch_num):
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True)

        torch.save(self.state_dict(), self.filepath.with_stem(
            f"{self.filepath.stem}_{epoch_num}"))
        print(f"Model checkpoint {self} saved for epoch {epoch_num}.")

    def load(self):
        self.load_state_dict(torch.load(self.filepath))

    def getCheckpoints(self) -> dict[str, list[int]]:
        d = {}
        for file in self.filepath.parent.iterdir():
            if not "_" in file.stem:
                continue
            name = file.stem.split("_")[0]
            number = file.stem.split("_")[-1]
            d.setdefault(name, []).append(number)
        for k, v in d.items():
            d[k] = sorted(v)
        return d
