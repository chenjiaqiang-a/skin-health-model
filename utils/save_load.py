import os
import pickle

import torch

__all__ = [
    "load_model", "save_model", "load_state_dict", "save_state_dict",
    "save_result"
]

_device = torch.device("cpu")


def load_model(path_model, device=_device):
    """load model from path_model 加载网络

    Args:
        path_model(str): the path of model file.
        device: the device to load file.
    """
    assert os.path.exists(path_model), "Model file doesn't exist!"
    model = torch.load(path_model, map_location=device)
    print('Load {} on {} successfully.'.format(path_model, device))
    return model


def save_model(model, path='.', filename='model.pkl'):
    """Save model to path/name 保存网络

    Args:
        model: the model to be saved.
        path(str): the path to save model.
        filename(str): the model file name.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    pth_model = os.path.join(path, filename)
    torch.save(model, pth_model)
    print('Model has been saved to {}'.format(pth_model))


def load_state_dict(model, dict_path, device=_device):
    """Load model parmas from state_dict 加载网络参数

    Args:
        model: the model to load params.
        dict_path(str): the path of state dict.
        device: the device to load model
    """
    assert os.path.exists(dict_path), "State dict file doesn't exist!"
    model.load_state_dict(torch.load(dict_path, map_location=device))
    return


def save_state_dict(model, path='.', filename='state_dict.pth'):
    """Save state dict to path/name 保存网络参数

    Args:
        model: the model to be saved.
        path(str): the path to save state dict.
        filename(str): the state dict file name.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    pth_dict = os.path.join(path, filename)
    torch.save(model.state_dict(), pth_dict)
    print('State dict has been saved to {}'.format(pth_dict))


def save_result(data, path='.', filename='result.pkl'):
    """Save result(a dict) using pickle

    Args:
        data(dict): the result to be saved.
        path(str): the path to save data.
        filename(str): the name of the file.
    """
    with open(os.path.join(path, filename), "wb") as fp:
        pickle.dump(data, fp)
