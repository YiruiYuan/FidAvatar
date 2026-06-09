"""Inference script for DiffPortrait360."""

import argparse
import os

import imageio
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image

from dataset import full_head_clean
from model_lib.ControlNet.cldm.model import create_model
from utils.checkpoint import load_from_pretrain
from utils.utils import count_param, print_peak_memory, set_seed


def save_360_images(generated_imgs, output_dir, image_name, num_360_images):
    image_stem = os.path.splitext(image_name)[0]
    image_output_dir = os.path.join(output_dir, image_stem)
    os.makedirs(image_output_dir, exist_ok=True)

    total_frames = len(generated_imgs)
    if total_frames == 0:
        return

    sample_count = max(1, int(num_360_images))
    if sample_count > total_frames:
        print(
            f"Warning: requested {sample_count} frames but only {total_frames} generated; "
            f"saving {total_frames} frames."
        )
        sample_count = total_frames
    elif sample_count < total_frames:
        print(
            f"Info: generated {total_frames} frames; keeping the first {sample_count} frames "
            "without resampling."
        )

    for idx in range(sample_count):
        frame = (generated_imgs[idx] * 255.0).clip(0, 255).astype("uint8")
        imageio.imwrite(f"{image_output_dir}/{idx + 1:04d}.png", frame)


def load_state_dict(model, ckpt_path, strict=True, map_location="cpu"):
    print(f"Loading model state dict from {ckpt_path} ...")
    state_dict = load_from_pretrain(ckpt_path, map_location=map_location)
    state_dict = state_dict.get("state_dict", state_dict)
    model.load_state_dict(state_dict, strict=strict)
    del state_dict


def visualize(args, batch_data, infer_model, n_sample):
    infer_model.eval()
    cond_imgs = batch_data["condition_image"].cuda()
    if n_sample == 1:
        conditions = batch_data["condition"].cuda()
        if args.denoise_from_guidance:
            fea_condition = batch_data["fea_condition"].cuda()
    else:
        try:
            conditions = torch.stack(batch_data["condition"]).squeeze().cuda()
        except Exception:
            conditions = batch_data["condition"].cuda()
        if args.denoise_from_guidance:
            fea_condition = batch_data["fea_condition"].cuda().squeeze()

    text = batch_data["text_blip"]
    c_cross = infer_model.get_learned_conditioning(text).repeat(n_sample, 1, 1)
    uc_cross = infer_model.get_unconditional_conditioning(n_sample)

    cond_img = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(cond_imgs))
    cond_img = cond_img.repeat(n_sample, 1, 1, 1)
    cond_img_cat = [cond_img]

    if "extra_appearance" in batch_data:
        more_cond_img = batch_data["extra_appearance"]
        more_cond_img = infer_model.get_first_stage_encoding(infer_model.encode_first_stage(more_cond_img.cuda()))
        more_cond_img = more_cond_img.repeat(n_sample, 1, 1, 1)

    gene_img_list = []
    generated_imgs = []
    for idx in range(conditions.shape[0] // n_sample):
        print(f"Generate Image {n_sample * idx} in {conditions.shape[0]} images")
        if args.denoise_from_guidance:
            fea_map_enc = infer_model.get_first_stage_encoding(
                infer_model.encode_first_stage(fea_condition[idx * n_sample : idx * n_sample + n_sample])
            )
            c = {
                "c_concat": [conditions[idx * n_sample : idx * n_sample + n_sample]],
                "c_crossattn": [c_cross],
                "image_control": cond_img_cat,
                "feature_control": fea_map_enc,
            }
        else:
            c = {
                "c_concat": [conditions[idx * n_sample : idx * n_sample + n_sample]],
                "c_crossattn": [c_cross],
                "image_control": cond_img_cat,
            }

        if args.control_mode == "controlnet_important":
            uc = {
                "c_concat": [conditions[idx * n_sample : idx * n_sample + n_sample]],
                "c_crossattn": [uc_cross],
            }
        else:
            uc = {
                "c_concat": [conditions[idx * n_sample : idx * n_sample + n_sample]],
                "c_crossattn": [uc_cross],
                "image_control": cond_img_cat,
            }

        c["wonoise"] = True
        uc["wonoise"] = True
        if "extra_appearance" in batch_data:
            c["more_image_control"] = [more_cond_img]

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            infer_model.to(args.device)
            infer_model.eval()
            gene_img, _ = infer_model.sample_log(
                cond=c,
                batch_size=n_sample,
                ddim=True,
                ddim_steps=50,
                eta=0.6,
                unconditional_guidance_scale=3,
                unconditional_conditioning=uc,
                inpaint=None,
            )

            decode_batch_size = max(1, int(args.decode_batch_size))
            decoded_chunks = []
            for decode_start in range(0, gene_img.shape[0], decode_batch_size):
                decode_end = decode_start + decode_batch_size
                decoded_chunks.append(infer_model.decode_first_stage(gene_img[decode_start:decode_end]))
            gene_img = torch.cat(decoded_chunks, dim=0)

            for frame_idx in range(n_sample):
                if "extra_appearance" in batch_data:
                    if "fea_condition" in batch_data and args.denoise_from_guidance:
                        cated = torch.cat(
                            [
                                fea_condition[idx * n_sample + frame_idx].cpu().squeeze(),
                                gene_img[frame_idx].cpu().squeeze(),
                                conditions[idx * n_sample + frame_idx].cpu().squeeze(),
                                cond_imgs.cpu().squeeze(),
                                batch_data["extra_appearance"].cpu().squeeze(),
                            ],
                            axis=2,
                        )
                    else:
                        cated = torch.cat(
                            [
                                gene_img[frame_idx].cpu().squeeze(),
                                conditions[idx * n_sample + frame_idx].cpu().squeeze(),
                                cond_imgs.cpu().squeeze(),
                                batch_data["extra_appearance"].cpu().squeeze(),
                            ],
                            axis=2,
                        )
                else:
                    cated = torch.cat(
                        [
                            gene_img[frame_idx].cpu().squeeze(),
                            conditions[idx * n_sample + frame_idx].cpu().squeeze(),
                            cond_imgs.cpu().squeeze(),
                        ],
                        axis=2,
                    )
                cated = cated.clamp(-1, 1).add(1).mul(0.5).permute(1, 2, 0).cpu().numpy()
                gene_img_list.append(cated)
                generated_imgs.append(
                    gene_img[frame_idx].squeeze().clamp(-1, 1).add(1).mul(0.5).permute(1, 2, 0).cpu().numpy()
                )

    if n_sample == 1:
        save_image(gene_img.clamp(-1, 1).add(1).mul(0.5), args.local_image_dir + "/" + batch_data["image_name"][0])
    else:
        save_360_images(generated_imgs, args.local_image_dir, batch_data["image_name"][0], args.num_360_images)
        if args.save_mp4:
            writer = imageio.get_writer(f"{args.local_image_dir}/{batch_data['image_name'][0]}.mp4", fps=10)
            for frame in generated_imgs:
                writer.append_data(frame)
            writer.close()


def main(args):
    args.device = torch.device("cuda")
    args.num_gpu = torch.cuda.device_count()
    args.use_gpu = torch.cuda.is_available() and args.num_gpu > 0
    os.makedirs(args.local_image_dir, exist_ok=True)

    print(args)
    set_seed(args.seed)

    model = create_model(args.model_config).cpu()
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.to(args.device)
    print("Total base  parameters {:.02f}M".format(count_param([model])))

    ckpt_path = args.resume_dir
    print(f"loading state dict from {ckpt_path} ...")
    load_state_dict(model, ckpt_path, strict=True)
    torch.cuda.empty_cache()

    image_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    if args.test_dataset == "back_head_generation":
        if args.use_initial_condition:
            print("Info: --use_initial_condition is ignored for back_head_generation; pose condition always comes from --condition_path.")
        test_dataset = full_head_clean.back_head_generation(
            image_transform=image_transform,
            inference_image_dataset=args.inference_image_path,
            condition_path=args.condition_path,
            initial_image_path=args.initial_image_path,
        )
    elif args.test_dataset == "full_head_clean_inference_final_face":
        test_dataset = full_head_clean.full_head_clean_inference_final_face(
            image_transform=image_transform,
            condition_path=args.condition_path,
            inference_image_dataset=args.inference_image_path,
            initial_image_path=args.initial_image_path,
        )
    else:
        print("find the appropriate dataset class!")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    test_iter = iter(test_loader)
    print(f"image dataloader created: dataset={args.test_dataset} batch_size={1}")

    infer_model = model.module if hasattr(model, "module") else model
    print("start inference loop!")
    first_print = True
    for step in range(len(test_loader)):
        test_batch_data = next(test_iter)
        with torch.no_grad():
            n_sample = int(args.nSample)
            visualize(args, test_batch_data, infer_model, n_sample=n_sample)
        if first_print or step % 200 == 0:
            torch.cuda.empty_cache()
            print_peak_memory(f"Max memory allocated After running {step} steps:", 0)
        first_print = False


if __name__ == "__main__":
    str2bool = lambda arg: bool(int(arg))

    parser = argparse.ArgumentParser(description="Control Net inference")
    parser.add_argument("--model_config", type=str, default="model_lib/ControlNet/models/cldm_v15_video.yaml")
    parser.add_argument("--sd_locked", type=str2bool, default=True)
    parser.add_argument("--only_mid_control", type=str2bool, default=False)
    parser.add_argument("--control_mode", type=str, default="balance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global_step", type=int, default=0)
    parser.add_argument("--local_image_dir", type=str, default=None, required=True)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--control_type", type=str, nargs="+", default=["pose"])
    parser.add_argument("--test_dataset", type=str, default=None)
    parser.add_argument("--inference_image_path", type=str, default=None)
    parser.add_argument("--denoise_from_guidance", action="store_true", default=False)
    parser.add_argument("--initial_image_path", type=str, default=None)
    parser.add_argument("--use_initial_condition", action="store_true", default=False)
    parser.add_argument("--condition_path", type=str, default=None)
    parser.add_argument("--nSample", type=int, default=None)
    parser.add_argument("--decode_batch_size", type=int, default=1)
    parser.add_argument("--num_360_images", type=int, default=32)
    parser.add_argument("--save_mp4", action="store_true", default=True)

    main(parser.parse_args())
