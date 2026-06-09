import torch.optim as optim

from fidavatar                         import FidAvatar

# ------------------------------------------------------------------------------- #

def register_optimizer_group_fidavatar(model: FidAvatar, cfg):

    optimizer_group = {}

    gs_l = [
        {'params': [model._opacity],        'lr': cfg.training.opacity_lr,     "name": "opacity"},
        {'params': [model._offset],         'lr': cfg.training.offset_lr,      "name": "offset"},
        {'params': [model._features_dc],    'lr': cfg.training.feature_dc_lr,  "name": "color"},
        {'params': [model._rotation],       'lr': cfg.training.rotation_lr,    "name": "rotation"},
        {'params': [model._scaling],        'lr': cfg.training.scaling_lr,     "name": "scaling"},
        {'params': [model.min_opacity],     'lr': 0.01,                        "name": "min_opacity"}
    ]

    gs_optimizer = optim.Adam(gs_l, lr=0.0)

    optimizer_group.update({'gs': gs_optimizer})

    bs_l = [
        {'params': [model.delta_shapedirs], 'lr': cfg.training.delta_shapedirs_lr, 'name': "delta_shapedirs"},
        {'params': [model.delta_posedirs],  'lr': cfg.training.delta_posedirs_lr,  'name': "delta_posedirs"},
        {'params': [model.delta_vertex], 'lr': 0.0001, 'name': "delta_vertex"},
    ]

    bs_optimizer = optim.Adam(bs_l, lr=0.0)

    optimizer_group.update({'bs': bs_optimizer})

    return optimizer_group
