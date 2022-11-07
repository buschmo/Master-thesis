import torch

def to_numpy(tensor: torch.Tensor):
    """
    Converts torch Variable to numpy nd array
    :param variable: torch Variable, of any size
    :return: numpy nd array, of same size as variable
    """
    if torch.cuda.is_available():
        return tensor.detach().cpu().numpy()
    else:
        return tensor.numpy()