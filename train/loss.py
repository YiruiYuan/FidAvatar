from    __future__ import annotations

import  torch
from    torch import nn
import  torch.nn.functional as F
import  warnings
import  lpips

from    tools.loss_utils.dssim import d_ssim
from    tools.loss_utils.vgg_feature import VGGPerceptualLoss

from    pytorch3d.structures import     Meshes
from    pytorch3d.loss.mesh_laplacian_smoothing import   mesh_laplacian_smoothing
from    pytorch3d.loss.mesh_normal_consistency  import   mesh_normal_consistency

from    typing import Type, Union
from    dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13")

# ------------------------------------------------------------------------------- #

def parsing_loss_param(loss_class:  Type[BaseLoss],
                       cfg_loss:    dict):
    """
    Args:
        loss_class      : Loss function class, which should have a 
                            nested `Params` class with annotated parameters.

        cfg_loss (dict): A dictionary containing the configuration for the loss 
                            function, including possible weight values and other loss parameters.

    Returns:
        BaseLoss.Params: An instance of the `Params` class of `loss_class`, initialized 
                         with the extracted parameters from `cfg_loss`.
    """
    
    param_class = loss_class.Params

    # ugly fix
    param_map = {key.replace('_weight', '_loss'): key for key in param_class.__annotations__}

    param_dict = {}
    for key, mapped_key in param_map.items():
        if key in cfg_loss:
            param_dict[mapped_key] = cfg_loss[key]
        elif key in cfg_loss.weight:
            param_dict[mapped_key] = cfg_loss['weight'][key]

    return param_class(**param_dict)

# ------------------------------------------------------------------------------- #

class BaseLoss(nn.Module):

    @dataclass
    class Params:
        loss_weight:        float

    def accumulate_gradients(self, model_output, ground_truth, cur_step=None, cur_epoch=None): # to be overridden by subclass
        raise NotImplementedError
    
    def forward(self, model_output, ground_truth, cur_step=None, cur_epoch=None):
        return self.accumulate_gradients(model_output, ground_truth, cur_step, cur_epoch)
    
# ------------------------------------------------------------------------------- #

class FidAvatarLoss(BaseLoss):

    @dataclass
    class Params:
        rgb_type:           str     = 'l1'
        rgb_weight:         float   = 1
        vgg_weight:         float   = 0
        dssim_weight:       float   = 0
        scale_weight:       float   = 0
        lpips_weight:       float   = 0
        scale_threshold:    float   = 0
        rot_weight:         float   = 0
        laplacian_weight:   float   = 0
        normal_weight:      float   = 0
        flame_weight:       float   = 0
    
    def __init__(self, params: Params):
        super().__init__()
        
        self.params = params

        self.vgg_loss               = VGGPerceptualLoss()
        self.lpips_loss             = lpips.LPIPS(net='vgg').eval()
        self.l1_loss                = nn.L1Loss(reduction='mean')
        self.l2_loss                = nn.MSELoss(reduction='mean')

        self.laplacian_matrix       = None

    def get_dssim_loss(self, rgb_values, rgb_gt):
        return d_ssim(rgb_values, rgb_gt)

    def get_vgg_loss(self, rgb_values, rgb_gt):
        return self.vgg_loss(rgb_values, rgb_gt)

    def get_rgb_loss(self, rgb_values, rgb_gt):
        if self.params.rgb_type == 'l1':
            return self.l1_loss(rgb_values, rgb_gt)
        elif self.params.rgb_type == 'l2':
            return self.l2_loss(rgb_values, rgb_gt)
    
    def get_lpips_loss(self, rgb_values, rgb_gt, normalize=True):
        return self.lpips_loss(rgb_values, rgb_gt, normalize=normalize)

    def get_laplacian_smoothing_loss(self, verts_orig, verts):
        L = self.laplacian_matrix[None, ...].detach()

        basis_lap   = L.bmm(verts_orig).detach()
        offset_lap  = L.bmm(verts)

        diff = (offset_lap - basis_lap) ** 2
        diff = diff.sum(dim=-1, keepdim=True)

        return diff.mean()

    def accumulate_gradients(self, model_outputs, ground_truth, cur_step=None, cur_epoch=None):

        render_image = model_outputs['rgb_image']   # torch.Size([1, 3, 512, 512])
        gt_image     = ground_truth['rgb']          # torch.Size([1, 3, 512, 512])

        # Initialize the loss
        loss = self.get_rgb_loss(render_image, gt_image) * self.params.rgb_weight
        out = {'loss': loss, 'rgb_loss': loss}

        # vgg loss
        if self.params.vgg_weight > 0:
            vgg_loss = self.get_vgg_loss(render_image, gt_image)
            out['vgg_loss'] = vgg_loss
            out['loss'] += vgg_loss * self.params.vgg_weight

        # dssim loss
        if self.params.dssim_weight > 0:
            dssim_loss = self.get_dssim_loss(render_image, gt_image)
            out['dssim_loss'] = dssim_loss
            out['loss'] += dssim_loss * self.params.dssim_weight

        # scale loss
        if self.params.scale_weight > 0:
            scale = model_outputs['scale']
            scale_max, _ = torch.max(scale, dim=-1)
            scale_min, _ = torch.min(scale, dim=-1)
            scale_regu = F.relu(scale_max / scale_min - self.params.scale_threshold).mean()
            out['scale_loss'] = scale_regu
            out['loss'] += scale_regu * self.params.scale_weight

        # lpips loss
        if self.params.lpips_weight > 0:
            lpips_loss = self.get_lpips_loss(render_image, gt_image).squeeze()
            out['lpips_loss'] = lpips_loss
            out['loss'] += lpips_loss * self.params.lpips_weight

        # rotation loss
        if self.params.rot_weight > 0:
            raw_rot = model_outputs['raw_rot']
            rot_loss = torch.mean(raw_rot[..., 0] ** 2) + torch.mean(raw_rot[..., 2] ** 2)
            out['rot_loss'] = rot_loss
            out['loss'] += rot_loss * self.params.rot_weight

        # laplacian or normal loss
        if self.params.laplacian_weight > 0 or self.params.normal_weight > 0:
            verts = model_outputs['verts']  # [1, V, 3]
            faces = model_outputs['faces']  # [F, 3]
            meshes = Meshes(verts=verts, faces=faces[None, ...])

            if self.laplacian_matrix is None:
                self.laplacian_matrix = meshes.laplacian_packed().to_dense()

            if self.params.laplacian_weight > 0:
                verts = model_outputs['verts']
                verts_orig = model_outputs['verts_orig']
                laplacian_loss  = self.get_laplacian_smoothing_loss(verts_orig, verts)
                out['laplacian_loss'] = laplacian_loss
                out['loss'] += laplacian_loss * self.params.laplacian_weight

                # laplacian_loss = mesh_laplacian_smoothing(meshes)
                # out['laplacian_loss'] = laplacian_loss
                # out['loss'] += laplacian_loss * self.params.laplacian_weight

            if self.params.normal_weight > 0:
                normal_loss = mesh_normal_consistency(meshes)
                out['normal_loss'] = normal_loss
                out['loss'] += normal_loss * self.params.normal_weight

        # flame loss
        if self.params.flame_weight > 0:
            verts = model_outputs['verts']
            verts_orig = model_outputs['verts_orig']
            flame_loss = (verts - verts_orig) ** 2
            out['flame_loss'] = flame_loss.mean()
            out['loss'] += out['flame_loss'] * self.params.flame_weight

        return out


LossClass = FidAvatarLoss

