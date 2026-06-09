import argparse
import os
import pickle

import torch


MODEL_NAME = "FidAvatar"
CONFIG_PATH = "./fidavatar.yaml"

NOVEL_VIEW = True
DLIB_KPS = True
AFFINE_TRANSFORM = True
INJECT_PRIOR = True
PANOHEAD_INVERSION = True
RENDER_INVERSION = True
INVERSE_TRANSFORM = True
RETRIEVE_MASK = True
HEATMAP_CHECK = True


def _env_or_arg(value, env_name):
    return value if value is not None else os.environ.get(env_name)


def resolve_default_config_path(config_path):
    if os.path.exists(config_path):
        return config_path

    root_config_path = os.path.join(os.path.dirname(__file__), os.path.basename(config_path))
    if os.path.exists(root_config_path):
        return root_config_path

    return config_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bg_color", type=str, default="white")
    parser.add_argument("--resume", action="store_true")
    opt = parser.parse_args()

    opt.root_path = _env_or_arg(opt.root_path, "DATASET_PATH")
    opt.workspace = _env_or_arg(opt.workspace, "EXP_DIR")
    opt.name = _env_or_arg(opt.name, "EXP_NAME")

    missing = [
        label
        for label, value in {
            "--root_path or DATASET_PATH": opt.root_path,
            "--workspace or EXP_DIR": opt.workspace,
            "--name or EXP_NAME": opt.name,
        }.items()
        if not value
    ]
    if missing:
        parser.error("missing required value(s): " + ", ".join(missing))

    opt.model_name = MODEL_NAME
    opt.config = resolve_default_config_path(CONFIG_PATH)
    return opt


def load_stage_config(opt, include_root_path=False):
    from common import load_config

    overrides = {
        "name": opt.name,
        "workspace": opt.workspace,
        "bg_color": opt.bg_color,
    }
    if include_root_path:
        overrides["root_path"] = opt.root_path
    return load_config(opt.config, overrides=overrides)


def run_mono_avatar(opt):
    from common import (
        construct_datasets,
        construct_loss,
        construct_metrics,
        construct_model,
        save_identity_info,
    )
    from tools.util import file_backup
    from train.trainer import Trainer

    cfg = load_stage_config(opt, include_root_path=True)

    datasets, dataset_name = construct_datasets(opt, cfg)
    save_identity_info(opt.workspace, datasets.train)

    model = construct_model(
        opt,
        cfg.model,
        cfg.dataset.canonical_pose,
        dataset=datasets.train,
    )
    criterion = construct_loss(opt, cfg.loss, datasets.train)
    metrics = construct_metrics(opt.device)

    file_backup(opt.workspace, opt.config)

    trainer = Trainer(
        opt.name,
        cfg,
        model,
        opt.device,
        train_dataset=datasets.train,
        test_dataset=datasets.test,
        criterions=criterion,
        metrics=metrics,
        workspace=opt.workspace,
        use_checkpoint="latest" if opt.resume else "scratch",
    )
    trainer.train(cfg.training.epochs[dataset_name])
    trainer.evaluate(mode="train", optim_epoch=cfg.training.epochs["finetune"])


def _remove_stale_affine_frames(generator):
    crop_dir = generator.media_save_path["affine_transform"]["folder"]
    quad_path = os.path.join(generator.media_save_path["aug_workspace"]["folder"], "quad.pkl")
    if not os.path.isdir(crop_dir) or not os.path.isfile(quad_path):
        return

    with open(quad_path, "rb") as f:
        quad_data = pickle.load(f, encoding="latin1")

    valid_names = {os.path.basename(str(key)) for key in quad_data.keys()}
    stale_names = []
    for image_name in os.listdir(crop_dir):
        lower_name = image_name.lower()
        if not lower_name.endswith((".png", ".jpg", ".jpeg")):
            continue
        if image_name not in valid_names:
            os.remove(os.path.join(crop_dir, image_name))
            stale_names.append(image_name)

    if stale_names:
        stale_names = sorted(stale_names)
        generator.log(
            f"[INFO] Removed {len(stale_names)} stale affine frames not in quad.pkl: "
            f"{', '.join(stale_names[:8])}"
        )


def run_generate_pseudo(opt):
    from common import construct_model, load_identity_info
    from tools.util import EasyDict
    from train.completion import PseudoGenerator

    cfg = load_stage_config(opt)

    identity_dict = load_identity_info(opt, cfg)
    model = construct_model(
        opt,
        cfg.model,
        0.0,
        identity_dict=identity_dict,
    )

    generator = PseudoGenerator(
        opt.name,
        cfg,
        model,
        opt.device,
        workspace=opt.workspace,
        use_checkpoints="latest",
    )
    generator.log(f"[INFO] Pseudo generation backend: {generator.pretrained_type}")

    orbit_frames = 30
    if generator.pretrained_type == "diffportrait":
        diff_cfg = getattr(cfg, "diffportrait", EasyDict())
        orbit_frames = int(getattr(diff_cfg, "step3_num_360_images", 32))
    generator.log(f"[INFO] Pseudo generation orbit_frames: {orbit_frames}")

    if NOVEL_VIEW:
        generator.render_novel_view(orbit_frames=orbit_frames)
    if DLIB_KPS:
        generator.detect_dlib_kps()
    if AFFINE_TRANSFORM:
        generator.execute_affine_transform()
        _remove_stale_affine_frames(generator)
    if INJECT_PRIOR:
        generator.inject_ffhq_prior()
    if PANOHEAD_INVERSION:
        generator.proceed_gan_inversion()
    if RENDER_INVERSION:
        generator.render_inversion_result(orbit_frames=orbit_frames)
    if INVERSE_TRANSFORM:
        generator.execute_inverse_transform()
    if RETRIEVE_MASK:
        generator.retrieve_image_mask()
        generator.retrieve_image_mask_modnet()
    if HEATMAP_CHECK:
        generator.heatmap_check()


def run_full_avatar(opt):
    from common import (
        construct_datasets,
        construct_loss,
        construct_metrics,
        construct_model,
        load_identity_info,
    )
    from train.completor import CompletionTrainer

    cfg = load_stage_config(opt, include_root_path=True)

    datasets, _ = construct_datasets(opt, cfg)
    identity_dict = load_identity_info(opt, cfg)
    model = construct_model(
        opt,
        cfg.model,
        0.0,
        identity_dict=identity_dict,
    )

    criterion = construct_loss(opt, cfg.loss, datasets.train)
    metrics = construct_metrics(opt.device)

    trainer = CompletionTrainer(
        opt.name,
        cfg,
        model,
        opt.device,
        train_dataset=datasets.train,
        test_dataset=datasets.test,
        criterions=criterion,
        metrics=metrics,
        workspace=opt.workspace,
        use_checkpoint="latest",
    )

    trainer.render_dynamic_novel_view(name="raw_dynamic_novel_view", mode="eval")
    trainer.render_dynamic_fixed_view(name="raw_dynamic_fixed_view", mode="eval", num_views=12)

    trainer.augmentation(finetune_epoch=1)
    full_head_ckpt = os.path.join(opt.workspace, "checkpoints_fullhead")

    trainer.evaluate_epoch(name="aug", mode="train_full_head")
    os.makedirs(full_head_ckpt, exist_ok=True)
    trainer.save_checkpoint(name="fullhead", remove_old=False, save_path=full_head_ckpt)

    trainer.render_dynamic_novel_view(name="dynamic_novel_view", mode="eval")
    trainer.render_dynamic_fixed_view(name="dynamic_fixed_view", mode="eval", num_views=12)


def main():
    opt = parse_args()
    import matplotlib
    matplotlib.use("Agg")

    from tools.util import seed_everything

    seed_everything(opt.seed)
    run_mono_avatar(opt)
    run_generate_pseudo(opt)
    run_full_avatar(opt)


if __name__ == "__main__":
    main()
