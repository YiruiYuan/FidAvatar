from train.dataset import IMAvatarDataset, InstaDataset
from train.loss import FidAvatarLoss
from fidavatar import FidAvatar

DatasetCallbacks = {
    '4dface': IMAvatarDataset,
    'imavatar': IMAvatarDataset,
    'insta': InstaDataset
}

ModelCallbacks = {
    'FidAvatar': FidAvatar
}

LossCallbacks = {
    'FidAvatar': FidAvatarLoss
}
