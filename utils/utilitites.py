from torch import Tensor

def to_numpy(tensor: Tensor):
    """
    Converts torch Variable to numpy nd array
    :param variable: torch Variable, of any size
    :return: numpy nd array, of same size as variable
    """
    if torch.cuda.is_available():
        return tensor.detech.cpu().numpy()
    else:
        return tensor.numpy()