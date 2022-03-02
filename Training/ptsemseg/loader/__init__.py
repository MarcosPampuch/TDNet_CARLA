import json

from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.carla_data import Carla


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "CARLA": Carla,
    }[name]
    

