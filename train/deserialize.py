import os
import glob
import torch

#-------------------------------------------------------------------------------#

def deserialize_checkpoints_fidavatar(self, checkpoint_dict):

    # --- separate gaussian attributes, since 'load_state_dict' can not handle it directly --- #
    gaussian_attributes = ['_offset', '_features_dc', '_features_rest', 
                            '_scaling','_rotation', '_opacity',
                            'face_index', 'bary_coords']
    missing_gaussian_attrs = [key for key in gaussian_attributes if key not in checkpoint_dict['model']]
    if len(missing_gaussian_attrs) > 0:
        raise KeyError(f"Checkpoint is missing gaussian attributes: {missing_gaussian_attrs}")
    gaussian_dict = {key: checkpoint_dict['model'].pop(key) for key in gaussian_attributes}

    # --- load other attributes (if have) --- #
    missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

    missing_keys_set = set(missing_keys)
    gaussian_attributes_set = set(gaussian_attributes)
    if not gaussian_attributes_set.issubset(missing_keys_set):
        required_missing = sorted(gaussian_attributes_set - missing_keys_set)
        raise RuntimeError(f"loaded manually, but these keys were not reported missing: {required_missing}")

    extra_missing_keys = sorted(missing_keys_set - gaussian_attributes_set)
    if len(extra_missing_keys) > 0:
        self.log("[WARN] missing non-gaussian keys in checkpoint (will use model defaults): "f"{extra_missing_keys}")

    # --- load gaussian attributes --- #
    for attr_name, attr_value in gaussian_dict.items():
        if attr_name not in ['face_index', 'bary_coords']:
            setattr(self.model, attr_name, torch.nn.Parameter(attr_value.requires_grad_(True)))
        else:
            setattr(self.model, attr_name, attr_value)
        missing_keys.remove(attr_name)
        if attr_name in missing_keys:
            missing_keys.remove(attr_name)

    # --- overwrite number of points --- #
    self.model.num_points = gaussian_dict['_offset'].shape[0]

    self.model.max_radii2D        = torch.zeros((self.model.num_points), device=self.device)
    self.model.xyz_gradient_accum = torch.zeros((self.model.num_points, 1), device=self.device)
    self.model.denom              = torch.zeros((self.model.num_points, 1), device=self.device)
    self.model.sample_flag        = torch.zeros((self.model.num_points), device=self.device)

    self.log("[INFO] loaded model.")
    if len(missing_keys) > 0:
        self.log(f"[WARN] missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        self.log(f"[WARN] unexpected keys: {unexpected_keys}")
