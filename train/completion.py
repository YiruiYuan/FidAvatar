import  glob
import  os
import  sys
import  math
import  tqdm
import  imageio
import  numpy as np
import  pickle
import  cv2
import  yaml
import  copy
import  json
import  PIL.Image
import  PIL.ImageChops
import  PIL.ImageOps
import  shutil
import  re
import  shlex
import  subprocess
from pathlib import Path

import  warnings
warnings.filterwarnings("ignore", category=UserWarning)

import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
import  torchvision
import  torchvision.transforms as transforms

from    typing import Type
from    fidavatar import ModelClass

from    tools.util         import EasyDict, load_to_gpu
from    tools.eg3d_utils.camera_eg3d import LookAtPoseSampler

from    pytorch3d.transforms import matrix_to_axis_angle

from    train.loader   import Loader

# ------------------------------------------------------------------------------- #

class PseudoGenerator(Loader):
    def __init__(
            self,
            name:              str,
            cfg:               EasyDict,
            model:             Type[ModelClass],
            device:            torch.device,
            workspace:         str,
            use_checkpoints:   str='latest',
            max_keep_ckpt:     int=2,
            bg_color:          str='white'
        ):
        super().__init__(name, cfg, model, device,
                         workspace, use_checkpoints, max_keep_ckpt)
        
        """
        The main goal of this class is to align images rendered using GS-based methods with
        those rendered by the pretrained 3D-aware GAN, which is a little bit tricky.
        """

        self.enhenced   = False
        self.bg_color   = bg_color

        self.dlib_threshold     = 1.0

        self.rotate_type        = 'camera'     # also try to be 'flame', but not good

        completion_cfg = getattr(self.cfg, "completion", EasyDict())
        self.pretrained_type = str(getattr(completion_cfg, "pretrained_type", "spherehead")).strip().lower()
        # supported: 'spherehead', 'panohead', 'arc2face', 'diffportrait'
        # optional manual override:
        # self.pretrained_type = 'diffportrait'

        self.pti_w_step         = 200
        self.pti_finetune_step  = 200

        self.rescale_scene      = True # To make the 3D-GAN output more complete, the nerf-scene can be scaled. However, this may result in a loss of image quality.
        self.rescale_factor     = 0.5

        # calculate flame rot joint position
        from flame.lbs import vertices2joints

        try:
            verts_cano, _, _ = self.model.flame.forward_with_delta_blendshape(
                expression_params   = self.model.flame.canonical_exp,
                full_pose           = self.model.flame.canonical_pose,
                delta_shapedirs     = self.model.delta_shapedirs if self.cfg.model.delta_blendshape else None,
                delta_posedirs      = self.model.delta_posedirs if self.cfg.model.delta_blendshape else None,
                delta_vertex        = self.model.delta_vertex if self.cfg.model.delta_vertex else None
            )
        except:
            verts_cano, _, _ = self.model.flame(
                expression_params   = self.model.flame.canonical_exp,
                full_pose           = self.model.flame.canonical_pose,
            )

        J = vertices2joints(self.model.flame.J_regressor, verts_cano)
        J_rot = J[0, 0]

        # carefully tune the last number...
        gs_camera_lookat_point = torch.tensor([0, 0, -0.02]).to(self.device)

        self.J = J
        self.gs_camera_lookat_point = gs_camera_lookat_point
        self.gs_camera_radius = self.cfg.camera_translation[-1]

        self.register_media_save()
        self.register_weight_path()

    def register_media_save(self):

        self.media_save_path = {
            "aug_workspace": {
                "folder": os.path.join(self.workspace, "augmentation")
            },
            "video": {
                "folder": os.path.join(self.workspace, "augmentation", "video")
            },
            "render_novel_view": {
                "folder": os.path.join(self.workspace, "augmentation", "novel_view")
            },
            "affine_transform": {
                "folder": os.path.join(self.workspace, "augmentation", "crop_images")
            },
            "inject_prior": {
                "folder": os.path.join(self.workspace, "augmentation", "crop_images_sr")
            },
            "run_pti":  {
                "folder": os.path.join(self.workspace, "augmentation", "gan_pti")
            },
            "inverse_transform":  {
                "folder": os.path.join(self.workspace, "augmentation", "paste_back")
            },
            "heatmap_check":    {
                "folder": os.path.join(self.workspace, "augmentation", "heatmap_check")
            }
        }

    def register_weight_path(self):
        supported_pretrained_types = {"spherehead", "panohead", "arc2face", "diffportrait"}
        if self.pretrained_type not in supported_pretrained_types:
            raise ValueError(
                f"Unsupported pretrained_type '{self.pretrained_type}'. "
                f"Expected one of {sorted(supported_pretrained_types)}"
            )

        diff_cfg = getattr(self.cfg, "diffportrait", EasyDict())

        self.weight_path = {
            "dlib": "./weights/shape_predictor_68_face_landmarks.dat",
            "gfpgan": "./weights/GFPGANv1.3.pth",
            "3d-gan": {
                "spherehead": "./weights/spherehead-ckpt-025000.pkl",
                "panohead": "./weights/easy-khair-180-gpc0.8-trans10-025000.pkl"
            },
            "arc2face": {
                "unet_dir": "./weights",
                "config": "./weights/config.json",
                "weights": "./weights/diffusion_pytorch_model.safetensors",
            },
            "diffportrait": {
                "panohead_model": getattr(
                    diff_cfg,
                    "panohead_model",
                    "./weights/easy-khair-180-gpc0.8-trans10-025000.pkl",
                ),
                "back_head_model": getattr(
                    diff_cfg,
                    "back_head_model",
                    "./weights/back_head-230000.th",
                ),
                "diff360_model": getattr(
                    diff_cfg,
                    "diff360_model",
                    "./weights/model_state-340000.th",
                ),
            },
            "bisenet": "./weights/79999_iter.pth",
            "modnet": "./weights/modnet_webcam_portrait_matting.ckpt",
        }

        for key, path in self.weight_path.items():
            if key == "3d-gan":
                if self.pretrained_type in path:
                    specific_path = path[self.pretrained_type]
                    if not os.path.exists(specific_path):
                        raise FileNotFoundError(f"Weight file for '{key}' ({specific_path}) is missing")
            elif key == "arc2face":
                if self.pretrained_type == "arc2face":
                    for _, p in path.items():
                        if not os.path.exists(p):
                            raise FileNotFoundError(f"Weight file for '{key}' ({p}) is missing")
            elif key == "diffportrait":
                if self.pretrained_type == "diffportrait":
                    for _, p in path.items():
                        if not os.path.exists(p):
                            raise FileNotFoundError(f"Weight file for '{key}' ({p}) is missing")
            else:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Weight file for '{key}' ({path}) is missing.")

    def _resolve_local_hf_snapshot(self, model_id_or_path, cache_dir=None):
        model_path = Path(str(model_id_or_path)).expanduser()
        if model_path.exists():
            return str(model_path.resolve())

        repo_id = str(model_id_or_path)
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"

        roots = []
        if cache_dir is not None:
            cache_root = Path(cache_dir).expanduser()
            roots.extend([
                cache_root,
                cache_root / "hub",
                cache_root / "huggingface" / "hub",
            ])

        home_cache = Path.home() / ".cache"
        roots.extend([
            home_cache / "huggingface" / "hub",
            home_cache / "hub",
            home_cache,
        ])

        seen = set()
        uniq_roots = []
        for root in roots:
            root_str = str(root)
            if root_str not in seen:
                seen.add(root_str)
                uniq_roots.append(root)

        for root in uniq_roots:
            repo_dir = root / repo_cache_name
            if not repo_dir.exists():
                continue

            ref_main = repo_dir / "refs" / "main"
            if ref_main.exists():
                commit = ref_main.read_text().strip()
                snapshot_dir = repo_dir / "snapshots" / commit
                if snapshot_dir.exists():
                    return str(snapshot_dir.resolve())

            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = [p for p in snapshots_dir.iterdir() if p.is_dir()]
                if snapshots:
                    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return str(snapshots[0].resolve())

        expected = Path.home() / ".cache" / "huggingface" / "hub" / repo_cache_name
        raise FileNotFoundError(
            f"Local base model '{repo_id}' not found. Expected cache repo dir like: {expected}"
        )

    def _run_arc2face_generation(self):
        arc_cfg = getattr(self.cfg, "arc2face", EasyDict())
        base_model = getattr(arc_cfg, "base_model", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        cache_dir = getattr(arc_cfg, "cache_dir", os.path.expanduser("~/.cache/huggingface/hub"))
        steps = int(getattr(arc_cfg, "num_inference_steps", 30))
        seed = int(getattr(arc_cfg, "seed", 42))
        generation_strength = float(getattr(arc_cfg, "generation_strength", 0.02))
        reference_generation_strength = float(getattr(arc_cfg, "reference_generation_strength", 0.002))

        prefer_reference = bool(getattr(arc_cfg, "prefer_reference", True))
        reference_dir = getattr(arc_cfg, "reference_dir", os.path.join(self.media_save_path["run_pti"]["folder"], "reference_aligned"))
        reference_output_mode = str(getattr(arc_cfg, "reference_output_mode", "copy")).strip().lower()
        reference_result_mix = float(getattr(arc_cfg, "reference_result_mix", 0.02))

        if reference_output_mode not in {"copy", "blend", "generate"}:
            raise ValueError(
                f"Invalid arc2face.reference_output_mode='{reference_output_mode}'. "
                "Expected one of: copy, blend, generate."
            )

        outdir = self.media_save_path["run_pti"]["folder"]
        out_image_dir = os.path.join(outdir, "image")
        os.makedirs(out_image_dir, exist_ok=True)
        reference_dir = os.path.abspath(os.path.expanduser(reference_dir))
        has_reference_dir = prefer_reference and os.path.isdir(reference_dir)

        self.log(
            f"++> Arc2Face reference mode: {reference_output_mode}, "
            f"prefer_reference={prefer_reference}, has_reference_dir={has_reference_dir}"
        )

        for stale_name in os.listdir(out_image_dir):
            if stale_name.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(out_image_dir, stale_name))

        novel_dir = self.media_save_path["render_novel_view"]["folder"]

        c2w_path = os.path.join(self.media_save_path["aug_workspace"]["folder"], "c2w.pkl")
        if not os.path.exists(c2w_path):
            raise FileNotFoundError(f"Missing camera trajectory file: {c2w_path}")

        with open(c2w_path, "rb") as f:
            c2w_data = pickle.load(f)

        c2w_by_name = {}
        for key, value in c2w_data.items():
            c2w_by_name[os.path.basename(str(key))] = np.array(value, dtype=np.float32)
        c2w_fallback = [c2w_by_name[k] for k in sorted(c2w_by_name.keys())]

        novel_files = sorted([
            n for n in os.listdir(novel_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not novel_files:
            raise FileNotFoundError(f"No target novel-view images found in {novel_dir}")

        reference_file_set = set()
        if has_reference_dir:
            reference_file_set = {
                n for n in os.listdir(reference_dir)
                if n.lower().endswith((".png", ".jpg", ".jpeg"))
            }
        num_reference_available = sum(1 for n in novel_files if n in reference_file_set)
        need_generation = reference_output_mode != "copy" or num_reference_available < len(novel_files)

        pipe = None
        if need_generation:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel
            except Exception as exc:
                raise ImportError(
                    "Arc2Face generation requires diffusers. Please install diffusers/transformers/accelerate in the current environment."
                ) from exc

            base_model = self._resolve_local_hf_snapshot(base_model, cache_dir=cache_dir)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            unet = UNet2DConditionModel.from_pretrained(
                self.weight_path["arc2face"]["unet_dir"],
                torch_dtype=dtype,
                local_files_only=True,
            )

            try:
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    base_model,
                    unet=unet,
                    torch_dtype=dtype,
                    safety_checker=None,
                    local_files_only=True,
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"Failed to load local base SD model '{base_model}'. "
                    "Please ensure it exists locally (or in cache_dir) because local_files_only=True is enabled."
                ) from exc
            pipe = pipe.to(self.device)

            if self.device.type == "cuda":
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
        else:
            self.log("++> Arc2Face diffusion bypassed: all frames will copy reference_aligned directly.")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        all_poses = {}
        video_frames = []
        num_reference_used = 0

        for idx, image_file in enumerate(novel_files):
            image_stem, _ = os.path.splitext(image_file)
            image_path = os.path.join(novel_dir, image_file)
            if not os.path.exists(image_path):
                self.log(f"++> [WARN] skip missing novel-view frame: {image_path}")
                continue

            c2w = c2w_by_name.get(image_file, None)
            if c2w is None and idx < len(c2w_fallback):
                c2w = c2w_fallback[idx]
            if c2w is None:
                c2w = np.eye(4, dtype=np.float32)

            world2cam = np.linalg.inv(c2w)
            all_poses[image_stem] = world2cam.tolist()

            init_image_path = image_path
            used_reference = False
            cur_strength = generation_strength
            if has_reference_dir:
                reference_path = os.path.join(reference_dir, image_file)
                if os.path.exists(reference_path):
                    init_image_path = reference_path
                    used_reference = True
                    num_reference_used += 1
                    cur_strength = reference_generation_strength

            cur_strength = float(np.clip(cur_strength, 0.0, 1.0))

            init_image = PIL.Image.open(init_image_path).convert("RGB").resize((512, 512), PIL.Image.Resampling.LANCZOS)

            if idx == 0:
                self.log(f"++> Arc2Face first-frame input image: {init_image_path}")
                self.log(
                    f"++> Arc2Face first-frame params: strength={cur_strength:.3f}, guidance_scale=1.000, "
                    f"steps={steps}, reference={used_reference}, mode={reference_output_mode}"
                )

            if used_reference and reference_output_mode == "copy":
                result = init_image.copy()
            else:
                result = pipe(
                    prompt="",
                    image=init_image,
                    num_inference_steps=steps,
                    strength=cur_strength,
                    guidance_scale=1.0,
                    generator=generator,
                ).images[0]

            if used_reference and reference_output_mode == "blend":
                result_mix = float(np.clip(reference_result_mix, 0.0, 1.0))
                result = PIL.Image.blend(init_image, result, result_mix)

            save_path = os.path.join(out_image_dir, image_file)
            result.save(save_path)
            video_frames.append(np.array(result))

        if not all_poses:
            raise RuntimeError("Arc2Face generation produced no valid view with trajectory")

        if has_reference_dir:
            self.log(f"++> Arc2Face used reference images for {num_reference_used}/{len(novel_files)} frames.")

        with open(os.path.join(outdir, "trajectory.json"), "w") as f:
            json.dump(all_poses, f, indent="\t")

        if len(video_frames) > 0:
            imageio.mimwrite(
                os.path.join(self.media_save_path["video"]["folder"], "pti.mp4"),
                video_frames,
                fps=25,
                quality=8,
                macro_block_size=1,
            )

    def _run_command(self, command, cwd=None, env=None):
        command = [str(arg) for arg in command]
        command_text = " ".join(shlex.quote(arg) for arg in command)
        self.log(f"++> Run command: {command_text}")
        subprocess.run(command, cwd=cwd, env=env, check=True)

    def _run_diffportrait_generation(self):
        diff_cfg = getattr(self.cfg, "diffportrait", EasyDict())

        def _as_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        def _to_abs(path_like):
            path = os.path.expanduser(str(path_like))
            if not os.path.isabs(path):
                path = os.path.join(repo_root, path)
            return os.path.abspath(path)

        repo_root = os.getcwd()
        diff_code_dir = os.path.join(repo_root, "diffportrait360", "code")
        if not os.path.isdir(diff_code_dir):
            raise FileNotFoundError(f"DiffPortrait code directory not found: {diff_code_dir}")

        condition_path = _to_abs(
            getattr(
                diff_cfg,
                "condition_path",
                os.path.join("diffportrait360", "sample_data", "cam_condition", "sphere32"),
            )
        )
        if not os.path.isdir(condition_path):
            raise FileNotFoundError(f"DiffPortrait condition path not found: {condition_path}")

        step1_num_steps = int(getattr(diff_cfg, "step1_num_steps", 200))
        step1_num_steps_pti = int(getattr(diff_cfg, "step1_num_steps_pti", 5))
        step1_fps = int(getattr(diff_cfg, "step1_fps", 30))
        step2_master_port = int(getattr(diff_cfg, "step2_master_port", 14020))
        step3_master_port = int(getattr(diff_cfg, "step3_master_port", 14031))
        step3_n_sample = int(getattr(diff_cfg, "step3_n_sample", 8))
        step3_decode_batch_size = int(getattr(diff_cfg, "step3_decode_batch_size", 1))
        step3_num_360_images = int(getattr(diff_cfg, "step3_num_360_images", 30))
        step3_denoise_from_guidance = _as_bool(getattr(diff_cfg, "step3_denoise_from_guidance", True))
        target_source_image = str(getattr(diff_cfg, "source_image_name", "0001.png"))

        outdir = _to_abs(self.media_save_path["run_pti"]["folder"])
        video_dir = _to_abs(self.media_save_path["video"]["folder"])
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        work_root = os.path.join(outdir, "diffportrait_work")
        input_image_dir = os.path.join(work_root, "input_image")
        noise_dir = os.path.join(work_root, "3DNoise")
        back_head_dir = os.path.join(work_root, "Back_Head")
        result_dir = os.path.join(work_root, "result")
        run_pti_image_dir = os.path.join(outdir, "image")

        def _reset_dir(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        _reset_dir(input_image_dir)
        _reset_dir(noise_dir)
        _reset_dir(back_head_dir)
        _reset_dir(result_dir)
        _reset_dir(run_pti_image_dir)

        source_dir = _to_abs(
            self.media_save_path["inject_prior"]["folder"] if self.enhenced else self.media_save_path["affine_transform"]["folder"]
        )
        source_dataset_path = os.path.join(source_dir, "dataset.json")
        if not os.path.exists(source_dataset_path):
            raise FileNotFoundError(f"Missing source camera json for DiffPortrait: {source_dataset_path}")

        source_images = sorted([
            name for name in os.listdir(source_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if len(source_images) == 0:
            raise FileNotFoundError(f"No source aligned images found for DiffPortrait in {source_dir}")

        image_lookup = {name.lower(): name for name in source_images}
        selected_source_name = image_lookup.get(target_source_image.lower(), source_images[0])
        selected_source_path = os.path.join(source_dir, selected_source_name)
        canonical_input_name = "0001.png"
        canonical_input_path = os.path.join(input_image_dir, canonical_input_name)
        diffportrait_meta = {
            "selected_source_name": selected_source_name,
            "canonical_input_name": canonical_input_name,
            "source_dir": source_dir,
        }
        shutil.copyfile(selected_source_path, canonical_input_path)
        self.log(
            f"++> DiffPortrait source frame: {selected_source_name} -> {canonical_input_name}"
        )

        with open(source_dataset_path, "r") as f:
            source_dataset_json = json.load(f)
        source_labels = source_dataset_json.get("labels", [])

        label_map = {str(item[0]): item[1] for item in source_labels}
        selected_label = label_map.get(selected_source_name)
        if selected_label is None:
            selected_png_name = os.path.splitext(selected_source_name)[0] + ".png"
            selected_label = label_map.get(selected_png_name)
        if selected_label is None:
            raise KeyError(
                f"Cannot find camera label for source image '{selected_source_name}' in {source_dataset_path}"
            )

        with open(os.path.join(input_image_dir, "dataset.json"), "w") as f:
            json.dump({"labels": [[canonical_input_name, selected_label]]}, f, indent="\t")

        python_exec = sys.executable
        diff_weight = self.weight_path["diffportrait"]
        env = os.environ.copy()
        cuda_visible_devices = getattr(diff_cfg, "cuda_visible_devices", None)
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        self.log("++> DiffPortrait Step1: PanoHead 3D-aware noise generation")
        self._run_command(
            [
                python_exec,
                "projector_withseg.py",
                "--outdir",
                noise_dir,
                "--num_steps",
                str(step1_num_steps),
                "--num_steps_pti",
                str(step1_num_steps_pti),
                "--fps",
                str(step1_fps),
                "--target_img_path",
                input_image_dir,
                "--network",
                diff_weight["panohead_model"],
                "--camera_json",
                os.path.join(input_image_dir, "dataset.json"),
            ],
            cwd=os.path.join(diff_code_dir, "3DNoise"),
            env=env,
        )

        def _pose_summary_from_c2w(c2w):
            c2w = np.array(c2w, dtype=np.float32).reshape(4, 4)
            t = c2w[:3, 3]
            r = float(np.linalg.norm(t))
            return c2w, t, r

        # Print pose diagnostics before entering DiffPortrait stage-2.
        selected_label_arr = np.array(selected_label, dtype=np.float32).reshape(-1)
        if selected_label_arr.size >= 16:
            diff_c2w, diff_t, diff_r = _pose_summary_from_c2w(selected_label_arr[:16])
            self.log(
                "++> [Pose@PreStep2] DiffPortrait source c2w:\n"
                + np.array2string(diff_c2w, precision=6, suppress_small=True)
            )
            self.log(
                f"++> [Pose@PreStep2] DiffPortrait source t={diff_t.tolist()}, radius={diff_r:.6f}"
            )
        else:
            self.log(
                f"[WARN] [Pose@PreStep2] selected_label has <16 values ({selected_label_arr.size}), cannot decode c2w."
            )

        novel_c2w_path = os.path.join(_to_abs(self.media_save_path["aug_workspace"]["folder"]), "c2w.pkl")
        if os.path.exists(novel_c2w_path):
            with open(novel_c2w_path, "rb") as f:
                novel_c2w_data = pickle.load(f)
            novel_by_name = {
                os.path.basename(str(k)): np.array(v, dtype=np.float32)
                for k, v in novel_c2w_data.items()
            }
            novel_names = sorted(novel_by_name.keys())
            if len(novel_names) > 0:
                ref_name = selected_source_name if selected_source_name in novel_by_name else novel_names[0]
                novel_ref_c2w, novel_ref_t, novel_ref_r = _pose_summary_from_c2w(novel_by_name[ref_name])
                all_r = [
                    float(np.linalg.norm(np.array(mat, dtype=np.float32).reshape(4, 4)[:3, 3]))
                    for mat in novel_by_name.values()
                ]
                self.log(
                    f"++> [Pose@PreStep2] Novel-view c2w (ref={ref_name}):\n"
                    + np.array2string(novel_ref_c2w, precision=6, suppress_small=True)
                )
                self.log(
                    f"++> [Pose@PreStep2] Novel-view ref t={novel_ref_t.tolist()}, radius={novel_ref_r:.6f}, "
                    f"orbit_radius(mean/min/max)={np.mean(all_r):.6f}/{np.min(all_r):.6f}/{np.max(all_r):.6f}, "
                    f"count={len(all_r)}"
                )
            else:
                self.log(f"[WARN] [Pose@PreStep2] Novel-view c2w.pkl is empty: {novel_c2w_path}")
        else:
            self.log(f"[WARN] [Pose@PreStep2] Novel-view c2w.pkl not found: {novel_c2w_path}")

        self.log("++> DiffPortrait Step2: Back-head generation")
        self._run_command(
            [
                python_exec,
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=1",
                "--master_port",
                str(step2_master_port),
                "inference.py",
                "--model_config",
                "./model_lib/ControlNet/models/cldm_v15_reference_only_pose_enable_PC.yaml",
                "--test_dataset",
                "back_head_generation",
                "--control_mode",
                "controlnet_important",
                "--local_image_dir",
                back_head_dir,
                "--resume_dir",
                diff_weight["back_head_model"],
                "--control_type",
                "GAN_Generated",
                "--inference_image_path",
                input_image_dir,
                "--nSample",
                "1",
                "--condition_path",
                condition_path,
                "--initial_image_path",
                noise_dir,
            ],
            cwd=diff_code_dir,
            env=env,
        )

        self.log("++> DiffPortrait Step3: 360-view sequence generation")
        step3_command = [
            python_exec,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--master_port",
            str(step3_master_port),
            "inference.py",
            "--model_config",
            "./model_lib/ControlNet/models/cldm_v15_reference_only_temporal_pose.yaml",
            "--test_dataset",
            "full_head_clean_inference_final_face",
            "--control_mode",
            "controlnet_important",
            "--local_image_dir",
            result_dir,
            "--resume_dir",
            diff_weight["diff360_model"],
            "--control_type",
            "GAN_Generated",
            "--inference_image_path",
            work_root,
            "--nSample",
            str(step3_n_sample),
            "--decode_batch_size",
            str(step3_decode_batch_size),
            "--num_360_images",
            str(step3_num_360_images),
            "--condition_path",
            condition_path,
            "--initial_image_path",
            noise_dir,
        ]
        if step3_denoise_from_guidance:
            step3_command.append("--denoise_from_guidance")
        self._run_command(step3_command, cwd=diff_code_dir, env=env)

        result_sequence_dir = os.path.join(result_dir, os.path.splitext(canonical_input_name)[0])
        if not os.path.isdir(result_sequence_dir):
            raise FileNotFoundError(
                f"DiffPortrait output sequence folder not found: {result_sequence_dir}"
            )

        generated_frames = sorted([
            name for name in os.listdir(result_sequence_dir)
            if name.lower().endswith(".png")
        ])
        if len(generated_frames) == 0:
            raise FileNotFoundError(
                f"No DiffPortrait frames generated under {result_sequence_dir}"
            )

        video_frames = []
        copied_frame_names = []
        for frame_idx, frame_name in enumerate(generated_frames, start=1):
            src_path = os.path.join(result_sequence_dir, frame_name)
            dst_name = f"{frame_idx:04d}.png"
            dst_path = os.path.join(run_pti_image_dir, dst_name)
            shutil.copyfile(src_path, dst_path)
            copied_frame_names.append(dst_name)
            video_frames.append(np.array(PIL.Image.open(dst_path).convert("RGB")))

        c2w_path = os.path.join(_to_abs(self.media_save_path["aug_workspace"]["folder"]), "c2w.pkl")
        if not os.path.exists(c2w_path):
            raise FileNotFoundError(f"Missing camera trajectory file: {c2w_path}")

        with open(c2w_path, "rb") as f:
            c2w_data = pickle.load(f)
        ordered_items = sorted(
            c2w_data.items(),
            key=lambda kv: os.path.basename(str(kv[0])),
        )
        ordered_c2w = [np.array(value, dtype=np.float32) for _, value in ordered_items]
        if len(ordered_c2w) == 0:
            raise RuntimeError("No camera trajectories found in c2w.pkl")

        all_poses = {}
        for idx, frame_name in enumerate(copied_frame_names):
            if idx < len(ordered_c2w):
                c2w = ordered_c2w[idx]
            else:
                c2w = ordered_c2w[-1]
            world2cam = np.linalg.inv(c2w)
            frame_stem, _ = os.path.splitext(frame_name)
            all_poses[frame_stem] = world2cam.tolist()

        with open(os.path.join(outdir, "trajectory.json"), "w") as f:
            json.dump(all_poses, f, indent="\t")

        if len(video_frames) > 0:
            imageio.mimwrite(
                os.path.join(video_dir, "pti.mp4"),
                np.stack(video_frames, axis=0),
                fps=25,
                quality=8,
                macro_block_size=1,
            )

        diffportrait_meta["generated_frames"] = len(copied_frame_names)
        with open(os.path.join(outdir, "diffportrait_meta.json"), "w") as f:
            json.dump(diffportrait_meta, f, indent="\t")

        self.log(
            f"++> DiffPortrait generation finished with {len(copied_frame_names)} frames."
        )
        
    @torch.no_grad()
    def render_novel_view(self, orbit_frames=40, ele_list=[0]):
        """
        Render novel view images from pretrained head avatar.
        """

        save_path = self.media_save_path["render_novel_view"]["folder"]
        os.makedirs(save_path, exist_ok=True)

        self.log('++> Render novel view images...')

        self.model.eval()

        total_len   = orbit_frames * len(ele_list)
        pbar        = tqdm.tqdm(total=total_len, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        input_data      = {}
        ground_truth    = {}

        input_data['expression'] = self.model.canonical_expression
        input_data['flame_pose'] = self.model.flame.canonical_pose
        
        input_data['fovx']  = [self.cfg.camera_fovx]
        input_data['fovy']  = [self.cfg.camera_fovy]

        #### insert a ugly fix ####
        if self.cfg.camera_rotation[1, 1] == 1.:
            self.cfg.camera_rotation[1, 1] = -1
            self.cfg.camera_rotation[2, 2] = -1

        input_data['cam_pose'] = torch.cat((self.cfg.camera_rotation, self.cfg.camera_translation[..., None]), dim=1)
        # erase translation in x, y
        input_data['cam_pose'][0, 3] = 0
        input_data['cam_pose'][1, 3] = 0
        input_data['cam_pose'] = input_data['cam_pose'][None, ...]

        results_cam_pose    = {}

        all_render_np   = []
        save_path_videos    = os.path.join(self.media_save_path["video"]["folder"], 'novel_view.mp4')
        os.makedirs(os.path.dirname(save_path_videos), exist_ok=True)

        for round, ele in enumerate(ele_list):
            for frame_idx in range(1, orbit_frames + 1):
                
                cam2world_pose  = LookAtPoseSampler.sample(
                    math.pi / 2 + 2 * math.pi * (frame_idx - 1) / orbit_frames,
                    math.pi / 2 - ele,
                    self.gs_camera_lookat_point,
                    radius  = self.gs_camera_radius,   # R
                    device  = self.device
                )

                # type-I: rotate camera
                if self.rotate_type == 'camera':
                    # input_data['cam_pose'][:, :3, :3] = cam2world_pose[:, :3, :3]

                    world2cam = torch.linalg.inv(cam2world_pose)
                    input_data['cam_pose'][:, :3, :4] = world2cam[:, :3, :4]

                # type-II: rotate head
                elif self.rotate_type == 'flame':
                    R = cam2world_pose[:, :3, :3]
                    R[:, 1] = R[:, 1] * -1
                    R[:, 2] = R[:, 2] * -1
                    rot_vec = matrix_to_axis_angle(R)
                    input_data["flame_pose"][:, :3] = rot_vec
                else:
                    raise

                load_to_gpu(input_data, ground_truth, self.device)
                output_data     = self.model(input_data)
                render_image    = output_data['rgb_image']
                render_np       = render_image[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                all_render_np.append((render_np * 255.).astype('uint8'))
                render_plot     = render_image[0].detach().cpu()
                image_name      = os.path.join(save_path, f'{frame_idx + round * orbit_frames:04d}.png')
                torchvision.utils.save_image(render_plot, image_name, normalize=True, value_range=(0, 1))

                results_cam_pose[image_name]    = cam2world_pose.squeeze().cpu().numpy()

                pbar.update(1)

        all_render_np = np.stack(all_render_np, axis=0)
        imageio.mimwrite(save_path_videos, all_render_np, fps=25, quality=10)

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'c2w.pkl'), 'wb') as f:
            pickle.dump(results_cam_pose, f)

        pbar.close()
        self.log_file_only(pbar)

        self.log('++> Render novel view finished.')

    def detect_dlib_kps(self):
        """
        Run dlib keypoints detection for further crop and filter out invalid images
        """
        import dlib
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']

        detector    = dlib.get_frontal_face_detector()
        predictor   = dlib.shape_predictor(self.weight_path["dlib"])

        self.log('++> Run dlib keypoints detection.')

        # load images
        img_dir     = self.media_save_path["render_novel_view"]["folder"]
        list_dir    = os.listdir(img_dir)

        # new dict for keypoints
        landmarks = {}
        pbar      = tqdm.tqdm(total=len(list_dir), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for img_name in list_dir:
            pbar.update(1)
            _, extension = os.path.splitext(img_name)
            if extension not in image_extensions:
                continue
            
            img_path    = os.path.join(img_dir, img_name)
            image       = cv2.imread(img_path)
            # gray scale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect face
            rects = detector(gray, 1)
            dets, scores, idx = detector.run(gray, 1)
            self.log(f"{img_path}: {scores}")

            if len(scores) == 0 or scores[0] < self.dlib_threshold:
                continue

            for (i, rect) in enumerate(rects):
                # get keypoints
                shape = predictor(gray, rect)
                # save kps to the dict
                landmarks[img_path] = [np.array([p.x, p.y]) for p in shape.parts()]

        # save the data.pkl pickle
        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'dlib_kps.pkl'), 'wb') as f:
            pickle.dump(landmarks, f)

        pbar.close()
        self.log_file_only(pbar)
        self.log('++> Dlib keypoints detection finished.')

    @torch.no_grad()
    def execute_affine_transform(self):
        """
        Align images with GAN preprocessing
        """
        root_dir = os.getcwd()

        if root_dir in sys.path:
            sys.path.remove(root_dir)

        TDDFA_LIB_PATH  = os.path.abspath(os.path.join(os.getcwd(), 'submodules/3DDFA_V2'))
        sys.path.insert(0, TDDFA_LIB_PATH)

        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA
        from tools.crop_utils.affine_util import (get_crop_bound, crop_image,
                                    find_center_bbox, crop_final,
                                    P2sRt, matrix2angle,
                                    eg3dcamparams)
        
        sys.path.insert(0, root_dir)

        self.log('++> Run Affine Align.')

        #----- load 3ddfa config -----#
        cur_dir = os.getcwd()
        os.chdir('./submodules/3DDFA_V2')
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        
        tddfa = TDDFA(gpu_mode='gpu', **cfg)
        face_boxes = FaceBoxes()
        
        os.chdir(cur_dir)
        #------------------------------#

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'dlib_kps.pkl'), "rb") as f:
            inputs = pickle.load(f, encoding="latin1").items()

        pbar = tqdm.tqdm(total=len(inputs), bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        size = 512
        results_quad = {}
        results_meta = {}
        results_orig_quad = {}
        for i, item in enumerate(inputs):
            pbar.update(1)

            # get initial cropping box (quad) using landmarks
            img_path, landmarks = item
            img_path = img_path
            img_orig = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
            if img_orig is None:
                print(f'Cannot load image')
                continue
            quad, quad_c, quad_x, quad_y = get_crop_bound(landmarks)

            results_orig_quad[img_path] = copy.deepcopy(quad)

            skip = False
            for iteration in range(1):
                bound = np.array([[0, 0], [0, size-1], [size-1, size-1], [size-1, 0]], dtype=np.float32)
                mat = cv2.getAffineTransform(quad[:3], bound[:3])
                img = crop_image(img_orig, mat, size, size)
                # img = img_orig
                h, w = img.shape[:2]

                # Detect faces, get 3DMM params and roi boxes
                boxes = face_boxes(img)
                # boxes = face_boxes(img_orig)
                if len(boxes) == 0:
                    print(f'No face detected')
                    skip = True
                    break

                param_lst, roi_box_lst = tddfa(img, boxes)
                # param_lst, roi_box_lst = tddfa(img_orig, boxes)
                box_idx = find_center_bbox(roi_box_lst, w, h)

                param = param_lst[box_idx]
                P = param[:12].reshape(3, -1)  # camera matrix
                s_relative, R, t3d = P2sRt(P)

                pose = matrix2angle(R)
                pose = [p * 180 / np.pi for p in pose]

                # Adjust z-translation in object space
                R_ = param[:12].reshape(3, -1)[:, :3]
                u = tddfa.bfm.u.reshape(3, -1, order='F')
                trans_z = np.array([ 0, 0, 0.5*u[2].mean() ]) # Adjust the object center
                trans = np.matmul(R_, trans_z.reshape(3,1))
                t3d += trans.reshape(3)

                ''' Camera extrinsic estimation for GAN training '''
                # Normalize P to fit in the original image (before 3DDFA cropping)
                sx, sy, ex, ey = roi_box_lst[0]
                scale_x = (ex - sx) / tddfa.size
                scale_y = (ey - sy) / tddfa.size
                t3d[0] = (t3d[0]-1) * scale_x + sx
                t3d[1] = (tddfa.size-t3d[1]) * scale_y + sy
                t3d[0] = (t3d[0] - 0.5*(w-1)) / (0.5*(w-1)) # Normalize to [-1,1]
                t3d[1] = (t3d[1] - 0.5*(h-1)) / (0.5*(h-1)) # Normalize to [-1,1], y is flipped for image space
                t3d[1] *= -1
                t3d[2] = 0 # orthogonal camera is agnostic to Z (the model always outputs 66.67)

                s_relative = s_relative * 2000
                scale_x = (ex - sx) / (w-1)
                scale_y = (ey - sy) / (h-1)
                s = (scale_x + scale_y) / 2 * s_relative
                # print(f"[{iteration}] s={s} t3d={t3d}")

                if s < 0.7 or s > 1.3:
                    print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} s={s}")
                    skip = True
                    break
                if abs(pose[0]) > 90 or abs(pose[1]) > 80 or abs(pose[2]) > 50:
                    print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} pose={pose}")
                    skip = True
                    break
                if abs(t3d[0]) > 1. or abs(t3d[1]) > 1.:
                    print(f"Skipping[{i+1-len(results_quad)}/{i+1}]: {img_path} pose={pose} t3d={t3d}")
                    skip = True
                    break

                quad_c = quad_c + quad_x * t3d[0]
                quad_c = quad_c - quad_y * t3d[1]
                quad_x = quad_x * s
                quad_y = quad_y * s
                c, x, y = quad_c, quad_x, quad_y
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)
                

            if skip:
                continue

            # final projection matrix
            s = 1
            t3d = 0 * t3d
            R[:,:3] = R[:,:3] * s
            P = np.concatenate([R,t3d[:,None]],1)
            P = np.concatenate([P, np.array([[0,0,0,1.]])],0)

            # Save cropped images

            cropped_img = crop_final(img_orig, size=size, quad=quad)
            # cropped_img = crop_final(img_orig, size=size, quad=quad, top_expand=0.0, left_expand=0.0, bottom_expand=0.0, right_expand=0.0)

            out_dir = self.media_save_path["affine_transform"]["folder"]

            if cropped_img is not None:
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), cropped_img)

                results_meta[img_path] = eg3dcamparams(P.flatten())
                results_quad[img_path] = quad

        pbar.close()
        self.log_file_only(pbar)

        self.log('++> Align Finished.')

        # if self.bg_color == 'black':
        #     # turn the background into white for inversion
        #     self.retrieve_image_mask_modnet(input_path = out_dir, output_path = out_dir.replace('multi_view_crop', 'multi_view_crop_mask'))

        #     for filename in os.listdir(out_dir):
        #         if filename.endswith('.json'):
        #             continue

        #         image = cv2.imread(os.path.join(out_dir, filename)).astype(np.float32) # [0, 255]
        #         mask  = cv2.imread(os.path.join(out_dir.replace('multi_view_crop', 'multi_view_crop_mask'), filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)  # [0, 255]

        #         image /= 255.
        #         mask  /= 255.

        #         image = image * mask[..., None] + (1 - mask[..., None])

        #         cv2.imwrite(os.path.join(out_dir, filename), (image * 255).astype(np.uint8))

        # Save quads
        self.log("results:", len(results_quad))
        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'quad.pkl'), 'wb') as f:
            pickle.dump(results_quad, f)

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'quad_orig.pkl'), 'wb') as f:
            pickle.dump(results_orig_quad, f)

        # Save meta data
        results_new = []
        for img, P  in results_meta.items():
            img = os.path.basename(img)
            res = [format(r, '.6f') for r in P]
            results_new.append((img,res))
        with open(os.path.join(self.media_save_path["affine_transform"]["folder"], 'dataset.json'), 'w') as outfile:
            json.dump({"labels": results_new}, outfile, indent="\t")

    @torch.no_grad()
    def inject_ffhq_prior(self):
        """
        Run pretrained face restore model to inject ffhq prior
        """

        GFPGAN_LIB_PATH = os.path.join(os.getcwd(), 'submodules/GFPGAN')
        sys.path.append(GFPGAN_LIB_PATH)

        from tools.sr_utils import GFPGANer
        from basicsr.utils import imwrite

        self.log('++> Run GFPGAN for enhencing.')

        orig_images     = self.media_save_path["affine_transform"]["folder"]
        final_images    = self.media_save_path["inject_prior"]["folder"]

        os.makedirs(final_images, exist_ok=True)

        orig_img_list = sorted(glob.glob(os.path.join(orig_images, '*')))
        orig_img_list = [img for img in orig_img_list if not img.endswith('.json')]

        bg_upsampler        = None
        arch                = 'clean'
        channel_multiplier  = 2
        model_path          = self.weight_path["gfpgan"]

        restorer = GFPGANer(
            model_path          = model_path,
            upscale             = 2,
            arch                = arch,
            channel_multiplier  = channel_multiplier,
            bg_upsampler        = bg_upsampler
        )

        for img_path in orig_img_list:
            img_name = os.path.basename(img_path)
            self.log(f'Processing {img_name} ...')

            basename, ext   = os.path.splitext(img_name)
            input_img       = cv2.imread(img_path, cv2.IMREAD_COLOR)

            # restore faces and background if necessary
            _, restored_faces, _ = restorer.enhance(
                input_img,
                has_aligned         = True,
                only_center_face    = False,
                paste_back          = False,
                weight              = 0.5)
            
            # save restored faces, which is a list (len == 1)
            if restored_faces is not None:
                extension = ext[1:]
                save_restore_path = os.path.join(final_images, f'{basename}.{extension}')
                imwrite(restored_faces.pop(), save_restore_path)

        self.log(f'Results are in the "{final_images}" folder.')

        # copy dataset.json
        shutil.copyfile(os.path.join(orig_images, 'dataset.json'), os.path.join(final_images, 'dataset.json'))
        self.enhenced   = True

        self.log('++> Enhencing finished.')

    def proceed_gan_inversion(self):
        """
        Run PTI
        """
        torch.cuda.empty_cache()

        if self.pretrained_type == 'arc2face':
            self.log('++> Proceed Arc2Face generation')
            self._run_arc2face_generation()
            self.log('++> Arc2Face generation finished')
            return
        if self.pretrained_type == 'diffportrait':
            self.log('++> Proceed DiffPortrait generation')
            self._run_diffportrait_generation()
            self.log('++> DiffPortrait generation finished')
            return
        if self.pretrained_type == 'spherehead':
            SPHEREHEAD_LIB_PAHT = os.path.join(os.getcwd(), 'submodules/SphereHead')
            sys.path.insert(1, SPHEREHEAD_LIB_PAHT)
            network_pkl = self.weight_path["3d-gan"]["spherehead"]
        elif self.pretrained_type == 'panohead':
            PANOHEAD_LIB_PATH = os.path.join(os.getcwd(), 'submodules/PanoHead')
            sys.path.insert(1, PANOHEAD_LIB_PATH)
            network_pkl = self.weight_path["3d-gan"]["panohead"]
        else:
            raise ValueError(f"Unsupported pretrained_type for GAN inversion: {self.pretrained_type}")

        import dnnlib
        import legacy
        from training.dataset import ImageFolderDataset

        from tools.eg3d_utils.pti import project_multi_view, project_pti_multi_view, save_optimization_video

        outdir  = self.media_save_path["run_pti"]["folder"]
        os.makedirs(outdir, exist_ok=True)

        self.log('++> Proceed GAN inversion')
        self.log('++> Load Networks from "%s"...' % network_pkl)

        with dnnlib.util.open_url(network_pkl) as fp:
            network_data = legacy.load_network_pkl(fp)
            G = network_data['G_ema'].requires_grad_(True).to(self.device)

        # hard code
        G.rendering_kwargs["ray_start"] = 2.35

        if self.enhenced:
            dataset_path    = self.media_save_path["inject_prior"]["folder"]
        else:
            dataset_path    = self.media_save_path["affine_transform"]["folder"]

        sphere_cfg = getattr(self.cfg, "spherehead", EasyDict())
        default_consistency_lambda = 0.5 if self.pretrained_type == 'spherehead' else 0.0
        side_back_consistency_lambda = float(
            getattr(sphere_cfg, "side_back_consistency_lambda", default_consistency_lambda)
        )
        side_back_pool_kernel = int(getattr(sphere_cfg, "side_back_pool_kernel", 8))
        side_back_delta_deg = float(getattr(sphere_cfg, "side_back_delta_deg", 30.0))

        self.log(
            f'++> Side/Back consistency: lambda={side_back_consistency_lambda:.4f}, '
            f'pool={side_back_pool_kernel}, back_delta_deg={side_back_delta_deg:.1f}'
        )

        dataset = ImageFolderDataset(
            path          = dataset_path,
            use_labels    = True,
            max_size      = None,
            xflip         = False
        )

        projected_w_steps   = project_multi_view(
            G,
            dataset,
            device      = self.device,
            log_fn      = self.log,
            num_steps   = self.pti_w_step,
            lambda_side_back_consistency = side_back_consistency_lambda,
            side_back_pool_kernel = side_back_pool_kernel,
            side_back_delta_deg = side_back_delta_deg,
        )

        G_steps = project_pti_multi_view(
            G,
            dataset,
            w_pivot     = projected_w_steps[-1:],
            device      = self.device,
            log_fn      = self.log,
            num_steps   = self.pti_finetune_step,
            lambda_side_back_consistency = side_back_consistency_lambda,
            side_back_pool_kernel = side_back_pool_kernel,
            side_back_delta_deg = side_back_delta_deg,
        )
        
        video = imageio.get_writer(
            os.path.join(self.media_save_path["video"]["folder"], 'optimization.mp4'),
            mode='I',
            fps=30,
            codec='libx264'
        )
        self.log(f'++> Saving optimization progress video ...')

        # save optimization video
        save_optimization_video(
            G,
            dataset,
            video,
            projected_w_steps,
            G_steps,
            device  = self.device
        )

        # Save final projected frame and W vector
        projected_w = projected_w_steps[-1]
        np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

        # Save network parameter
        with open(f'{outdir}/fintuned_generator.pkl', 'wb') as f:
            G_final = G_steps[-1].to(self.device)
            network_data["G_ema"] = G_final.eval().requires_grad_(False).cpu()
            pickle.dump(network_data, f)

        self.log('++> GAN inversion finished')

    def render_inversion_result(self, orbit_frames=40, ele_list=[0]):
        """
        Render PTI result
        """
        torch.cuda.empty_cache()

        if self.pretrained_type in {'arc2face', 'diffportrait'}:
            outdir = self.media_save_path["run_pti"]["folder"]
            img_dir = os.path.join(outdir, 'image')
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"{self.pretrained_type} output folder not found: {img_dir}")
            image_files = sorted([
                n for n in os.listdir(img_dir)
                if n.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            if len(image_files) == 0:
                raise FileNotFoundError(f"No {self.pretrained_type} generated images found in {img_dir}")

            frames = [
                np.array(PIL.Image.open(os.path.join(img_dir, n)).convert('RGB'))
                for n in image_files
            ]
            imageio.mimwrite(
                os.path.join(self.media_save_path["video"]["folder"], 'pti.mp4'),
                frames,
                fps=25,
                quality=8,
                macro_block_size=1,
            )
            self.log(f'++> {self.pretrained_type} render finished')
            return

        if self.pretrained_type == 'spherehead':
            SPHEREHEAD_LIB_PAHT = os.path.join(os.getcwd(), 'submodules/SphereHead')
            sys.path.insert(1, SPHEREHEAD_LIB_PAHT)
        elif self.pretrained_type == 'panohead':
            PANOHEAD_LIB_PATH = os.path.join(os.getcwd(), 'submodules/PanoHead')
            sys.path.insert(1, PANOHEAD_LIB_PATH)

        import dnnlib
        import legacy

        from tools.eg3d_utils.pti import gen_orbit_video

        outdir      = self.media_save_path["run_pti"]["folder"]

        self.log('++> Render PanoHead inversion')
        self.log(f'++> Load Networks from "{outdir}/fintuned_generator.pkl"...')

        ws = torch.tensor(np.load(f'{outdir}/projected_w.npz')['w']).to(self.device)
        with dnnlib.util.open_url(f'{outdir}/fintuned_generator.pkl') as fp:
            network_data = legacy.load_network_pkl(fp)
            G = network_data['G_ema'].requires_grad_(False).to(self.device) # type: ignore

        sampling_multiplier = 2
        G.rendering_kwargs["ray_start"]                     = 2.35 - self.rescale_factor / 2      # default: 2.35
        G.rendering_kwargs["ray_end"]                       = 3.3 + self.rescale_factor / 2      # default: 3.3
        G.rendering_kwargs['depth_resolution']              = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
        G.rendering_kwargs['depth_resolution_importance']   = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)

        gen_orbit_video(
            G,
            self.J,
            mp4_save_path       = os.path.join(self.media_save_path["video"]["folder"], 'pti.mp4'),
            save_path           = outdir,
            ws                  = ws,
            gs_lookat_point     = self.gs_camera_lookat_point,
            gs_radius           = self.gs_camera_radius,
            w_frames            = orbit_frames,
            ele_list            = ele_list,
            device              = self.device,
            rotate_type         = self.rotate_type,
            rescale_scene       = self.rescale_scene,
            rescale_factor      = self.rescale_factor,
        )

        self.log('++> Render finished')

    def execute_inverse_transform(self):
        """
        Paste aligned images into unaligned state
        """
        def natural_sort_key(s):
            return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

        def _as_pose_matrix(value):
            return np.array(value, dtype=np.float32).reshape(4, 4)

        def _pose_summary(c2w):
            c2w = _as_pose_matrix(c2w)
            t = c2w[:3, 3]
            return t, float(np.linalg.norm(t))

        def _rotation_angle_deg(rotation):
            rotation = np.array(rotation, dtype=np.float32).reshape(3, 3)
            cos_angle = (np.trace(rotation) - 1.0) * 0.5
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            return float(np.degrees(np.arccos(cos_angle)))

        def _rotation_to_rpy_deg(rotation):
            rotation = np.array(rotation, dtype=np.float32).reshape(3, 3)
            sy = math.sqrt(float(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
            singular = sy < 1e-6
            if not singular:
                roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
                pitch = math.atan2(float(-rotation[2, 0]), sy)
                yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
            else:
                roll = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
                pitch = math.atan2(float(-rotation[2, 0]), sy)
                yaw = 0.0
            return np.degrees(np.array([roll, pitch, yaw], dtype=np.float32))

        def _name_candidates(name):
            if name is None:
                return []
            base_name = os.path.basename(str(name))
            stem_name, ext = os.path.splitext(base_name)
            candidates = [base_name]
            if ext:
                candidates.extend([f"{stem_name}.png", f"{stem_name}.jpg", f"{stem_name}.jpeg"])
            else:
                candidates.extend([f"{base_name}.png", f"{base_name}.jpg", f"{base_name}.jpeg"])
            result = []
            seen = set()
            for candidate in candidates:
                lower_candidate = candidate.lower()
                if lower_candidate in seen:
                    continue
                seen.add(lower_candidate)
                result.append(candidate)
            return result

        def _find_label_by_name(labels, name):
            candidates = {candidate.lower() for candidate in _name_candidates(name)}
            if len(candidates) == 0:
                return None, None
            for item in labels:
                if len(item) < 2:
                    continue
                label_name = os.path.basename(str(item[0]))
                if label_name.lower() in candidates:
                    label_arr = np.array(item[1], dtype=np.float32).reshape(-1)
                    if label_arr.size >= 16:
                        return label_name, label_arr
            return None, None

        def _load_diffportrait_source_c2w():
            diff_meta_path = os.path.join(pti_dir, "diffportrait_meta.json")
            if not os.path.exists(diff_meta_path):
                return None, None, None
            with open(diff_meta_path, "r") as f:
                diff_meta = json.load(f)

            source_dir = diff_meta.get("source_dir")
            if not source_dir:
                return None, None, diff_meta
            source_dataset_path = os.path.join(source_dir, "dataset.json")
            if not os.path.exists(source_dataset_path):
                return None, None, diff_meta
            with open(source_dataset_path, "r") as f:
                source_dataset_json = json.load(f)
            source_labels = source_dataset_json.get("labels", [])

            preferred_names = [
                diff_meta.get("selected_source_name"),
                diff_meta.get("canonical_input_name"),
                reference_quad_name,
            ]
            for preferred_name in preferred_names:
                label_name, label_arr = _find_label_by_name(source_labels, preferred_name)
                if label_arr is not None:
                    return _as_pose_matrix(label_arr[:16]), label_name, diff_meta

            for item in source_labels:
                if len(item) < 2:
                    continue
                label_arr = np.array(item[1], dtype=np.float32).reshape(-1)
                if label_arr.size >= 16:
                    return _as_pose_matrix(label_arr[:16]), os.path.basename(str(item[0])), diff_meta
            return None, None, diff_meta

        def _load_novel_ref_c2w(diff_meta):
            c2w_path = os.path.join(self.media_save_path["aug_workspace"]["folder"], "c2w.pkl")
            if not os.path.exists(c2w_path):
                return None, None
            with open(c2w_path, "rb") as f:
                c2w_data = pickle.load(f, encoding="latin1")
            c2w_by_name = {
                os.path.basename(str(key)).lower(): (os.path.basename(str(key)), _as_pose_matrix(value))
                for key, value in c2w_data.items()
            }

            preferred_names = []
            if diff_meta:
                preferred_names.extend([
                    diff_meta.get("selected_source_name"),
                    diff_meta.get("canonical_input_name"),
                ])
            preferred_names.extend([reference_quad_name, images_files[0] if len(images_files) > 0 else None])

            for preferred_name in preferred_names:
                for candidate in _name_candidates(preferred_name):
                    hit = c2w_by_name.get(candidate.lower())
                    if hit is not None:
                        return hit[1], hit[0]
            if len(c2w_by_name) > 0:
                first_name = sorted(c2w_by_name.keys(), key=natural_sort_key)[0]
                hit = c2w_by_name[first_name]
                return hit[1], hit[0]
            return None, None

        def _foreground_bbox(image_path, threshold=245):
            if not os.path.exists(image_path):
                return None
            image = np.array(PIL.Image.open(image_path).convert("RGB"))
            mask = np.any(image < threshold, axis=2)
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                return None
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            width = x1 - x0 + 1
            height = y1 - y0 + 1
            center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            size = np.array([width, height], dtype=np.float32)
            return {
                "bbox": [x0, y0, x1, y1],
                "center": center,
                "size": size,
            }

        def _log_bbox_delta(frame_name, novel_bbox, paste_bbox):
            if novel_bbox is None or paste_bbox is None:
                self.log(
                    f"[WARN] [Image@PostInverse] frame={frame_name} "
                    f"novel_bbox={'missing' if novel_bbox is None else novel_bbox['bbox']} "
                    f"paste_bbox={'missing' if paste_bbox is None else paste_bbox['bbox']}"
                )
                return None, None
            center_delta = paste_bbox["center"] - novel_bbox["center"]
            size_delta = paste_bbox["size"] - novel_bbox["size"]
            size_ratio = paste_bbox["size"] / np.maximum(novel_bbox["size"], 1.0)
            self.log(
                f"++> [Image@PostInverse] frame={frame_name} "
                f"novel_bbox={novel_bbox['bbox']} paste_bbox={paste_bbox['bbox']}"
            )
            self.log(
                f"++> [Image@PostInverse] center_delta_xy={center_delta.tolist()} "
                f"size_delta_wh={size_delta.tolist()} size_ratio_wh={size_ratio.tolist()}"
            )
            return center_delta, size_ratio

        def _log_post_inverse_diagnostics(reference_bound, affine_matrix):
            if self.pretrained_type != "diffportrait":
                return

            diff_c2w, diff_source_name, diff_meta = _load_diffportrait_source_c2w()
            novel_c2w, novel_ref_name = _load_novel_ref_c2w(diff_meta)

            if diff_c2w is None:
                self.log("[WARN] [Pose@PostInverse] Cannot load DiffPortrait source c2w.")
            else:
                diff_t, diff_radius = _pose_summary(diff_c2w)
                self.log(
                    f"++> [Pose@PostInverse] DiffPortrait source c2w (ref={diff_source_name}):\n"
                    + np.array2string(diff_c2w, precision=6, suppress_small=True)
                )
                self.log(
                    f"++> [Pose@PostInverse] DiffPortrait source t={diff_t.tolist()}, radius={diff_radius:.6f}, "
                    f"rpy_deg={_rotation_to_rpy_deg(diff_c2w[:3, :3]).tolist()}"
                )

            if novel_c2w is None:
                self.log("[WARN] [Pose@PostInverse] Cannot load Novel-view ref c2w.")
            else:
                novel_t, novel_radius = _pose_summary(novel_c2w)
                self.log(
                    f"++> [Pose@PostInverse] Novel-view ref c2w (ref={novel_ref_name}):\n"
                    + np.array2string(novel_c2w, precision=6, suppress_small=True)
                )
                self.log(
                    f"++> [Pose@PostInverse] Novel-view ref t={novel_t.tolist()}, radius={novel_radius:.6f}, "
                    f"rpy_deg={_rotation_to_rpy_deg(novel_c2w[:3, :3]).tolist()}"
                )

            if diff_c2w is not None and novel_c2w is not None:
                relative_rotation = diff_c2w[:3, :3] @ novel_c2w[:3, :3].T
                relative_t = diff_c2w[:3, 3] - novel_c2w[:3, 3]
                diff_rpy = _rotation_to_rpy_deg(diff_c2w[:3, :3])
                novel_rpy = _rotation_to_rpy_deg(novel_c2w[:3, :3])
                self.log(
                    f"++> [Pose@PostInverse] relative_rotation_angle_deg="
                    f"{_rotation_angle_deg(relative_rotation):.6f}"
                )
                self.log(
                    f"++> [Pose@PostInverse] relative_t_xyz={relative_t.tolist()}, "
                    f"xy_shift={relative_t[:2].tolist()}, z_shift={float(relative_t[2]):.6f}"
                )
                self.log(
                    f"++> [Pose@PostInverse] delta_rpy_deg={(diff_rpy - novel_rpy).tolist()}, "
                    f"relative_rpy_deg={_rotation_to_rpy_deg(relative_rotation).tolist()}"
                )

            affine_2x3 = np.array(affine_matrix, dtype=np.float32).reshape(2, 3)
            affine_linear = affine_2x3[:, :2]
            affine_scale = [
                float(np.linalg.norm(affine_linear[:, 0])),
                float(np.linalg.norm(affine_linear[:, 1])),
            ]
            self.log(
                "++> [Affine@PostInverse] reference_quad:\n"
                + np.array2string(reference_quad, precision=6, suppress_small=True)
            )
            self.log(
                "++> [Affine@PostInverse] bound:\n"
                + np.array2string(reference_bound, precision=6, suppress_small=True)
            )
            self.log(
                "++> [Affine@PostInverse] M:\n"
                + np.array2string(affine_2x3, precision=6, suppress_small=True)
            )
            self.log(
                f"++> [Affine@PostInverse] scale_xy={affine_scale}, "
                f"det={float(np.linalg.det(affine_linear)):.6f}, "
                f"translation_xy={affine_2x3[:, 2].tolist()}"
            )

            novel_image_dir = self.media_save_path["render_novel_view"]["folder"]
            diagnostic_frame = reference_quad_name if reference_quad_name in images_files else None
            if diagnostic_frame is None:
                for image_file in images_files:
                    if os.path.exists(os.path.join(novel_image_dir, image_file)):
                        diagnostic_frame = image_file
                        break
            if diagnostic_frame is not None:
                novel_bbox = _foreground_bbox(os.path.join(novel_image_dir, diagnostic_frame))
                paste_bbox = _foreground_bbox(os.path.join(save_image_dir, diagnostic_frame))
                _log_bbox_delta(diagnostic_frame, novel_bbox, paste_bbox)

            center_deltas = []
            size_ratios = []
            for image_file in images_files:
                novel_path = os.path.join(novel_image_dir, image_file)
                paste_path = os.path.join(save_image_dir, image_file)
                if not os.path.exists(novel_path) or not os.path.exists(paste_path):
                    continue
                novel_bbox = _foreground_bbox(novel_path)
                paste_bbox = _foreground_bbox(paste_path)
                if novel_bbox is None or paste_bbox is None:
                    continue
                center_deltas.append(paste_bbox["center"] - novel_bbox["center"])
                size_ratios.append(paste_bbox["size"] / np.maximum(novel_bbox["size"], 1.0))
            if len(center_deltas) > 0:
                center_deltas_np = np.stack(center_deltas, axis=0)
                size_ratios_np = np.stack(size_ratios, axis=0)
                self.log(
                    f"++> [Image@PostInverse] bbox_summary count={len(center_deltas)} "
                    f"mean_center_delta_xy={center_deltas_np.mean(axis=0).tolist()} "
                    f"mean_abs_center_delta_xy={np.abs(center_deltas_np).mean(axis=0).tolist()} "
                    f"mean_size_ratio_wh={size_ratios_np.mean(axis=0).tolist()}"
                )
        
        self.log('++> Run affine inverse.')
        
        pti_dir         = self.media_save_path["run_pti"]["folder"]
        paste_dir       = self.media_save_path["inverse_transform"]["folder"]

        pti_image_dir   = os.path.join(pti_dir, 'image')
        # pano_mask_dir   = os.path.join(pano_dir, 'mask')

        images_files = [
            name for name in os.listdir(pti_image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        # mask_files      = os.listdir(pano_mask_dir)

        # for proper visualzation
        images_files = sorted(images_files, key=natural_sort_key)

        save_image_dir  = os.path.join(paste_dir, 'image')
        # save_mask_dir   = os.path.join(paste_dir, 'mask')
        save_video_dir  = os.path.join(paste_dir, 'paste_novel_view.mp4')

        os.makedirs(save_image_dir, exist_ok=True)
        # os.makedirs(save_mask_dir, exist_ok=True)

        with open(os.path.join(self.media_save_path["aug_workspace"]["folder"], 'quad.pkl'), "rb") as f:
            inputs = pickle.load(f, encoding="latin1")

        quad_by_name = {
            os.path.basename(str(key)): np.array(value, dtype=np.float32)
            for key, value in inputs.items()
        }
        if len(quad_by_name) == 0:
            raise ValueError("No valid quad entries found in quad.pkl.")

        available_items = sorted(quad_by_name.items(), key=lambda kv: natural_sort_key(kv[0]))
        quad_by_lower_name = {
            key.lower(): (key, value)
            for key, value in quad_by_name.items()
        }

        def _try_get_quad(quad_name):
            if not quad_name:
                return None
            base_name = os.path.basename(str(quad_name))
            stem_name, ext = os.path.splitext(base_name)
            name_candidates = [base_name]
            if ext:
                name_candidates.extend([f"{stem_name}.png", f"{stem_name}.jpg", f"{stem_name}.jpeg"])
            else:
                name_candidates.extend([f"{base_name}.png", f"{base_name}.jpg", f"{base_name}.jpeg"])

            visited = set()
            for candidate in name_candidates:
                lower_candidate = candidate.lower()
                if lower_candidate in visited:
                    continue
                visited.add(lower_candidate)
                hit = quad_by_lower_name.get(lower_candidate)
                if hit is not None:
                    return hit
            return None

        preferred_quad_names = []
        if self.pretrained_type == "diffportrait":
            diff_cfg = getattr(self.cfg, "diffportrait", EasyDict())
            preferred_quad_names.append(getattr(diff_cfg, "source_image_name", None))
            diff_meta_path = os.path.join(pti_dir, "diffportrait_meta.json")
            if os.path.exists(diff_meta_path):
                with open(diff_meta_path, "r") as f:
                    diff_meta = json.load(f)
                preferred_quad_names.append(diff_meta.get("selected_source_name"))
                preferred_quad_names.append(diff_meta.get("canonical_input_name"))
        preferred_quad_names.extend(["0001.png", images_files[0] if len(images_files) > 0 else None])

        reference_quad_name = None
        reference_quad = None
        for candidate_name in preferred_quad_names:
            hit = _try_get_quad(candidate_name)
            if hit is not None:
                reference_quad_name, reference_quad = hit
                break

        if reference_quad is None:
            reference_quad_name, reference_quad = available_items[0]
            self.log(
                f"[WARN] Cannot match preferred reference quad names {preferred_quad_names}; "
                f"fallback to '{reference_quad_name}'."
            )
        self.log(f"++> Inverse reference quad: {reference_quad_name}")
        
        size = 512

        # magic numbers
        top_expand      = 0.1
        left_expand     = 0.05
        bottom_expand   = 0.0
        right_expand    = 0.05

        crop_w      = int(size * (1 + left_expand + right_expand))
        crop_h      = int(size * (1 + top_expand + bottom_expand))
        crop_size   = (crop_w, crop_h)

        top     = int(size * top_expand)
        left    = int(size * left_expand)

        bound = np.array([
                        [left, top],
                        [left, top + size - 1],
                        [left + size - 1, top + size - 1],
                        [left + size - 1, top]
                        ], dtype=np.float32)

        delta_bound = bound - 256
        apply_rescale_compensation = self.pretrained_type in {"spherehead", "panohead"} and self.rescale_scene
        if apply_rescale_compensation:
            ratio = 2.7 / (2.7 + self.rescale_factor)
            delta_bound *= ratio
        else:
            delta_bound *= 1.0
            if self.pretrained_type in {"arc2face", "diffportrait"} and self.rescale_scene:
                self.log(
                    f"++> Skip inverse rescale compensation for backend '{self.pretrained_type}'."
                )
        bound = delta_bound + 256

        M = cv2.getAffineTransform(reference_quad[:3], bound[:3]).flatten()

        all_images_np = []
        valid_mask_src = PIL.Image.new("L", crop_size, 255)
        novel_image_dir = self.media_save_path["render_novel_view"]["folder"]
        diff_cfg = getattr(self.cfg, "diffportrait", EasyDict())
        affine_shift_weight = float(getattr(diff_cfg, "affine_shift_weight", 0.35))
        affine_shift_max_px = float(getattr(diff_cfg, "affine_shift_max_px", 24.0))
        shift_reference_name = images_files[0] if len(images_files) > 0 else None
        shift_reference_bbox = None
        if shift_reference_name is not None:
            shift_reference_bbox = _foreground_bbox(os.path.join(novel_image_dir, shift_reference_name))
        if self.pretrained_type == "diffportrait":
            if shift_reference_bbox is None:
                self.log(
                    f"[WARN] [AffineShift] Cannot compute reference bbox for {shift_reference_name}; "
                    "per-frame shift compensation disabled."
                )
            else:
                self.log(
                    f"++> [AffineShift] reference={shift_reference_name}, "
                    f"center={shift_reference_bbox['center'].tolist()}, "
                    f"weight={affine_shift_weight:.3f}, max_px={affine_shift_max_px:.3f}"
                )
        affine_shift_deltas = []
        affine_shift_applied = []

        for i in range(len(images_files)):

            rgb_image   = PIL.Image.new("RGB", (size, size), "white")
            # mask_image  = PIL.Image.new("L", (size, size), "black")

            image_file  = images_files[i]
            # mask_file   = mask_files[i]

            image   = PIL.Image.open(os.path.join(pti_image_dir, image_file)).convert('RGB')
            # mask    = PIL.Image.open(os.path.join(pano_mask_dir, mask_file)).convert('L')

            # mask = mask.point(lambda x: 255 if x > 128 else 0, mode='1')

            image   = image.resize(crop_size)
            # mask    = mask.resize(crop_size)
        
            unalign_img = image.transform(crop_size, PIL.Image.Transform.AFFINE, M, PIL.Image.Resampling.BICUBIC)
            valid_mask = valid_mask_src.transform(
                crop_size,
                PIL.Image.Transform.AFFINE,
                M,
                PIL.Image.Resampling.NEAREST,
                fillcolor=0,
            )
            if (
                self.pretrained_type == "diffportrait"
                and shift_reference_bbox is not None
                and image_file != shift_reference_name
            ):
                current_bbox = _foreground_bbox(os.path.join(novel_image_dir, image_file))
                if current_bbox is not None:
                    shift_delta = current_bbox["center"] - shift_reference_bbox["center"]
                    affine_shift_deltas.append(shift_delta)
                    shift_apply = np.clip(
                        shift_delta * affine_shift_weight,
                        -affine_shift_max_px,
                        affine_shift_max_px
                    )
                    affine_shift_applied.append(shift_apply)
                    dx, dy = float(shift_apply[0]), float(shift_apply[1])
                    unalign_img = unalign_img.transform(
                        crop_size,
                        PIL.Image.Transform.AFFINE,
                        (1, 0, -dx, 0, 1, -dy),
                        PIL.Image.Resampling.BICUBIC,
                        fillcolor=(255, 255, 255),
                    )
                    valid_mask = valid_mask.transform(
                        crop_size,
                        PIL.Image.Transform.AFFINE,
                        (1, 0, -dx, 0, 1, -dy),
                        PIL.Image.Resampling.NEAREST,
                        fillcolor=0,
                    )
                else:
                    self.log(f"[WARN] [AffineShift] Missing current bbox for {image_file}; no shift applied.")
            # unalign_mask = mask.transform(crop_size, PIL.Image.Transform.AFFINE, M, PIL.Image.Resampling.BICUBIC)

            rgb_image.paste(unalign_img, (0, 0), mask=valid_mask)
            # mask_image.paste(unalign_mask, (0, 0),  mask=PIL.ImageOps.invert(mask_ops))

            rgb_image.save(os.path.join(save_image_dir, image_file))
            # mask_image.save(os.path.join(save_mask_dir, mask_file))

            all_images_np.append(np.array(rgb_image))

        if self.pretrained_type == "diffportrait" and len(affine_shift_deltas) > 0:
            affine_shift_deltas_np = np.stack(affine_shift_deltas, axis=0)
            affine_shift_applied_np = np.stack(affine_shift_applied, axis=0)
            self.log(
                f"++> [AffineShift] applied_count={len(affine_shift_deltas)}, "
                f"mean_delta_xy={affine_shift_deltas_np.mean(axis=0).tolist()}, "
                f"mean_abs_delta_xy={np.abs(affine_shift_deltas_np).mean(axis=0).tolist()}, "
                f"max_abs_delta_xy={np.abs(affine_shift_deltas_np).max(axis=0).tolist()}, "
                f"mean_applied_xy={affine_shift_applied_np.mean(axis=0).tolist()}, "
                f"mean_abs_applied_xy={np.abs(affine_shift_applied_np).mean(axis=0).tolist()}, "
                f"max_abs_applied_xy={np.abs(affine_shift_applied_np).max(axis=0).tolist()}"
            )

        imageio.mimwrite(save_video_dir, all_images_np, fps=25, quality=8, macro_block_size=1)
        shutil.copyfile(os.path.join(pti_dir, 'trajectory.json'), os.path.join(paste_dir, 'trajectory.json'))
        _log_post_inverse_diagnostics(bound, M)

        self.log('++> Affine inverse finished.')

    @torch.no_grad()
    def retrieve_image_mask(self, input_path = None, output_path = None):
        """
        Get mask from parsing net
        """
        PARSING_LIB_PATH = os.path.join(os.getcwd(), 'submodules/face-parsing.PyTorch')
        sys.path.append(PARSING_LIB_PATH)

        import importlib.util
        spec = importlib.util.spec_from_file_location("BiSeNetModule", os.path.join(PARSING_LIB_PATH, 'model.py'))
        bi_se_net_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bi_se_net_module)

        BiSeNet = bi_se_net_module.BiSeNet

        self.log('++> Matting pasted images.')
                
        paste_dir           = self.media_save_path["inverse_transform"]["folder"]
        paste_image_dir     = os.path.join(paste_dir, 'image')
        paste_mask_dir      = os.path.join(paste_dir, 'mask_bisenet')
        os.makedirs(paste_mask_dir, exist_ok=True)

        if input_path is None or output_path is None:
            input_path  = paste_image_dir
            output_path = paste_mask_dir
        else:
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

        n_classes = 19
        net = BiSeNet(n_classes=n_classes).to(self.device)
        ckpt_path = self.weight_path["bisenet"]
        net.load_state_dict(torch.load(ckpt_path))
        net.eval()

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])

        im_names = os.listdir(input_path)
        im_names = [img for img in im_names if not img.endswith('.json')]
        for im_name in im_names:
            self.log('Matte image: {0}'.format(im_name))

            image = PIL.Image.open(os.path.join(input_path, im_name))
            info = {}

            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.to(self.device)

            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            single_label_mask = np.zeros_like(parsing)
            head_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17])
            # head_array = np.array([16])
            index = np.where(np.isin(parsing, head_array))     # neckhead
            single_label_mask[index] = 1
            single_label_mask_image = PIL.Image.fromarray((single_label_mask).astype(np.uint8) * 255)

            matte_name = im_name
            single_label_mask_image.save(os.path.join(output_path, matte_name))

            # image_w_mask = PIL.Image.composite(image, PIL.Image.new('RGB', image.size, (255, 255, 255)), single_label_mask_image)
            # image_w_mask.save(os.path.join(input_path, matte_name))

        self.log('++> Matted finished.')

    @torch.no_grad()
    def retrieve_image_mask_modnet(self, input_path = None, output_path = None):
        """
        Get mask from MODNet
        """

        MODNET_LIB_PATH = os.path.join(os.getcwd(), 'submodules/MODNet')
        sys.path.append(MODNET_LIB_PATH)

        from src.models.modnet import MODNet

        self.log('++> Matting pasted images.')
                
        paste_dir           = self.media_save_path["inverse_transform"]["folder"]
        paste_image_dir     = os.path.join(paste_dir, 'image')
        paste_mask_dir      = os.path.join(paste_dir, 'mask_modnet')
        os.makedirs(paste_mask_dir, exist_ok=True)

        if input_path is None or output_path is None:
            input_path  = paste_image_dir
            output_path = paste_mask_dir
        else:
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)

        ckpt_path           = './weights/modnet_webcam_portrait_matting.ckpt'

        # define hyper-parameters
        ref_size = 512

        # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)

        modnet = modnet.to(self.device)
        weights = torch.load(ckpt_path)
        modnet.load_state_dict(weights)
        modnet.eval()

        # inference images
        im_names = os.listdir(input_path)
        im_names = [img for img in im_names if not img.endswith('.json')]
        for im_name in im_names:
            self.log('Matte image: {0}'.format(im_name))

            # read image
            im = PIL.Image.open(os.path.join(input_path, im_name))

            # unify image channels to 3
            im = np.asarray(im)
            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            elif im.shape[2] == 4:
                im = im[:, :, 0:3]

            # convert image to PyTorch tensor
            im = PIL.Image.fromarray(im)
            im = im_transform(im)

            # add mini-batch dim
            im = im[None, :, :, :]

            # resize image for input
            im_b, im_c, im_h, im_w = im.shape
            if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                if im_w >= im_h:
                    im_rh = ref_size
                    im_rw = int(im_w / im_h * ref_size)
                elif im_w < im_h:
                    im_rw = ref_size
                    im_rh = int(im_h / im_w * ref_size)
            else:
                im_rh = im_h
                im_rw = im_w
            
            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

            # inference
            _, _, matte = modnet(im.to(self.device), True)

            # resize and save matte
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            matte_name = im_name
            PIL.Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(output_path, matte_name))


        mask_files = [f for f in os.listdir(paste_mask_dir) if f.endswith('.png')]
        result = None

        for mask_file in mask_files:
            img = PIL.Image.open(os.path.join(paste_mask_dir, mask_file))
            img = np.array(img)

            if result is None:
                result = img
            else:
                result = np.logical_and(result, img)

        result_img = PIL.Image.fromarray(result.astype(np.uint8) * 255)

        threshold = 10

        boundary_height = None
        for row in range(result.shape[1]):
            row_ = 512 - row - 1
            white_count = np.sum(result[row_, :])
            if white_count >= threshold:
                boundary_height = row_
                break

        if boundary_height is not None:
            boundary_height = row_
        else:
            raise

        width = ref_size
        height = ref_size

        new_img = np.zeros((height, width), dtype=np.uint8)

        new_img[boundary_height:, :] = 0
        new_img[:boundary_height, :] = 255

        new_img_pil = PIL.Image.fromarray(new_img)

        new_img_pil.save(os.path.join(self.media_save_path["inverse_transform"]["folder"], 'torsor_boundary.png'))

        self.log('++> Matted finished.')

    def heatmap_check(self):
        """
        Check misalignment via heatmap
        """

        from tools.util import colorize_weights_map

        self.log('++> Run heatmap check.')

        paste_dir           = self.media_save_path["inverse_transform"]["folder"]
        paste_image_dir     = os.path.join(paste_dir, 'image')
        gs_image_dir        = self.media_save_path["render_novel_view"]["folder"]

        heatmap_dir         = self.media_save_path["heatmap_check"]["folder"]
        os.makedirs(heatmap_dir, exist_ok=True)

        img_names = sorted([
            n for n in os.listdir(gs_image_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        for img_name in img_names:
            
            gs_img      = PIL.Image.open(os.path.join(gs_image_dir,     img_name))
            paste_img   = PIL.Image.open(os.path.join(paste_image_dir,  img_name))

            gs_img_tensor       = transform(gs_img)[None, ...]
            paste_img_tensor    = transform(paste_img)[None, ...]

            err = (gs_img_tensor - paste_img_tensor).abs().max(dim=1)[0].clip(0, 1)
            err_plot = colorize_weights_map(err, min_val=0, max_val=1)

            grid   = torchvision.utils.make_grid(err_plot, nrow=1, normalize=True, value_range=(0, 1))
            torchvision.utils.save_image(grid, os.path.join(heatmap_dir, img_name))

        self.log('++> Heatmap check done.')
