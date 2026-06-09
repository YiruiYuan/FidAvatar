import torch

from tools.util import EasyDict

from fidavatar                         import FidAvatar

from train.loss                         import FidAvatarLoss

#-------------------------------------------------------------------------------#

def iteration_step_fidavatar(input_data:       dict,
                              ground_truth:     dict,
                              model:            FidAvatar,
                              criterions:       FidAvatarLoss,
                              optimizers_group: dict,
                              cfg:              EasyDict,
                              global_step:      int,
                              cur_epoch:        int,
                              log = None,
                              **kwargs):
        
        if log is None:
            log = print

        output_data  = model(input_data)
        render_image = output_data['rgb_image']
        gt_image     = ground_truth['rgb']

        loss_output = criterions(output_data, ground_truth)

        loss = loss_output['loss']

        bs                  = output_data['bs']
        viewspace_points    = output_data['viewspace_points']
        visibility_filter   = output_data['visibility_filter']
        radii               = output_data['radii']

        #------------------------ zero grad ------------------------#
        for name, optimizer in optimizers_group.items():
            optimizer.zero_grad(set_to_none=True)

        loss.backward()

        #------------------------ do gaussian maintain ------------------------#
        for bs_ in range(bs):
            model._add_densification_stats(viewspace_points[bs_], visibility_filter[bs_])

        #------------------------ optimize ------------------------#
        for name, optimizer in optimizers_group.items():
            optimizer.step()

        # ------------------------ densify ------------------------
        if global_step % cfg.training.densify_interval == 0:

            # do uv densification
            old_num = model.num_points
            if old_num < cfg.training.max_points_num:
                
                model._uv_densify(optimizers_group['gs'],
                    increase_num = min(cfg.training.max_points_num - old_num, cfg.training.increase_num))
                
                log(f"Do UV densification, Guassian splats: {old_num} --> {model.num_points}.")
            else:
                log(f"Guassian splats: {old_num} has reached maximum number.")

        # ------------------------ prune ------------------------
        if global_step % cfg.training.prune_interval == 0:
            old_num = model.num_points
            model._prune_low_opacity_points(optimizers_group['gs'])
            
            log(f"Prune low opacity points, Guassian splats: {old_num} --> {model.num_points}.")
        # ------------------------ reset opacity ------------------------
        if global_step % cfg.training.opacity_reset_interval == 0 and global_step != 0:
            model._reset_opacity(optimizers_group['gs'])

        return {'loss_output': loss_output,
                'render_image': render_image,
                'gt_image': gt_image}
