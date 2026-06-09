import torch


def load(filepath, map_location="cpu", **kwargs):
    return torch.load(filepath, map_location=map_location, **kwargs)


def load_from_pretrain(pretrain_path, map_location="cpu"):
    if not pretrain_path:
        return {}
    return load(pretrain_path, map_location=map_location)
