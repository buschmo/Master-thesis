import torch
from torch import nn
from pathlib import Path

class BaseModel(nn.Module):
    
    def __init__(self, filepath=None):
        super().__init__()
        if filepath:
            self.filepath = filepath
        else:
            self.update_filepath()
        
    def __repr__(self):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError
        
    def update_filepath(self):
        dir_name = Path("checkpoints").absolute()
        self.filepath = Path(dir_name, self.__str__(), self.__str__()+".pt")

    def save(self):
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True)

        torch.save(self.state_dict(), self.filepath)
        print(f"Model {self} saved")

    def save_checkpoint(self, epoch_num):
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True)
            
        filename = self.filepath.with_stem(f"{self.__str__()}_{epoch_num}")
        torch.save(self.state_dict(), filename)
        print(f"Model checkpoint {self} saved for epoch {epoch_num}.")
    
    def load(self):
        self.load_state_dict(torch.load(self.filepath))

