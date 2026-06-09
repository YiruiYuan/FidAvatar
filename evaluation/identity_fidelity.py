#!/usr/bin/env python3
"""
Evaluate identity fidelity between one real image and multiple rendered views
from different methods using ArcFace embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from insightface.app import FaceAnalysis

try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_REAL_IMAGE = Path("/home/eric/Desktop/mycode/Fatediffavatar/data/insta/bala/images/00000.png")
DEFAULT_METHOD_ROOTS = [
    "mycode=/home/eric/Desktop/mycode/Fatediffavatar/workspace/insta/bala/completion/eval_render",
    "baseline=/home/eric/Desktop/mycode/Fatediffavatar/workspace/insta/bala/completion_geo/eval_render",
]
DEFAULT_OUTPUT = Path("/home/eric/Desktop/mycode/Fatediffavatar/workspace/insta/bala/identity_fidelity_geo.json")
SKIP_VIEW_IDS = {4, 5, 6}


def parse_named_paths(entries: List[str], flag_name: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for raw in entries:
        if "=" not in raw:
            raise ValueError(f"Each {flag_name} entry must be name=path")
        name, path_str = raw.split("=", 1)
        key = name.strip()
        if not key:
            raise ValueError(f"Each {flag_name} entry must include a non-empty method name")
        parsed[key] = Path(path_str.strip())
    return parsed


def detect_embedding(face_app: FaceAnalysis, image_rgb: np.ndarray, device: torch.device) -> torch.Tensor | None:
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    image_bgr = image_rgb[..., ::-1].copy()
    try:
        faces = face_app.get(image_bgr)
    except Exception:
        return None
    if not faces:
        return None
    faces = sorted(
        faces,
        key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]),
        reverse=True,
    )
    embedding = np.array(faces[0]["embedding"], dtype=np.float32)
    emb_tensor = torch.from_numpy(embedding).to(device)
    emb_tensor = F.normalize(emb_tensor, dim=0)
    return emb_tensor


def compute_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float32)
    return {
        "ISD_mean": float(arr.mean()),
        "ISD_std": float(arr.std(ddof=0)),
    }


def save_histogram(values: List[float], out_path: Path) -> None:
    if not _HAS_MATPLOTLIB or not values:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 3.0))
    plt.hist(values, bins=50, range=(-1.0, 1.0), color="seagreen", edgecolor="black")
    plt.xlabel("Cosine similarity (real ↔ render)")
    plt.ylabel("Count")
    plt.title("Identity fidelity distribution")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def collect_render_images(method_root: Path, view_count: int, image_name: str) -> List[Tuple[int, Path]]:
    render_images: List[Tuple[int, Path]] = []
    for view_idx in range(view_count):
        if view_idx in SKIP_VIEW_IDS:
            continue
        view_root = method_root / f"view_{view_idx}"
        image_dir = view_root / "dynamic_fixed_view"
        if not image_dir.exists():
            image_dir = view_root / "raw_dynamic_fixed_view"
        image_path = image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing render image: {image_path}")
        render_images.append((view_idx, image_path))
    return render_images


def summarize_method(values: List[float]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "values": values,
    }
    if values:
        stats = compute_stats(values)
        summary.update(
            {
                "ISD_mean": stats["ISD_mean"],
                "ISD_std": stats["ISD_std"],
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )
    else:
        summary.update(
            {
                "ISD_mean": None,
                "ISD_std": None,
                "min": None,
                "max": None,
            }
        )
    return summary


def build_comparison_result(per_method: Dict[str, Dict[str, object]]) -> str | None:
    method_names = sorted(per_method.keys())
    if len(method_names) < 2:
        return None

    if len(method_names) != 2:
        ranked = sorted(
            (
                (name, result["ISD_mean"])
                for name, result in per_method.items()
                if result["ISD_mean"] is not None
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(ranked) < 2:
            return None
        higher_name, higher_mean = ranked[0]
        lower_name, lower_mean = ranked[1]
        return f"{higher_name} is higher than {lower_name} by {higher_mean - lower_mean:.4f}"

    first_name, second_name = method_names
    first_mean = per_method[first_name]["ISD_mean"]
    second_mean = per_method[second_name]["ISD_mean"]
    if first_mean is None or second_mean is None:
        return None
    if abs(first_mean - second_mean) < 1e-8:
        return f"{first_name} and {second_name} are equal"
    if first_mean > second_mean:
        return f"{first_name} is higher than {second_name} by {first_mean - second_mean:.4f}"
    return f"{second_name} is higher than {first_name} by {second_mean - first_mean:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identity fidelity (ArcFace) between one real image and rendered views from multiple methods."
    )
    parser.add_argument(
        "--real-image",
        type=Path,
        default=DEFAULT_REAL_IMAGE,
        help="Path to the real image used as the reference identity.",
    )
    parser.add_argument(
        "--method-roots",
        nargs="+",
        default=DEFAULT_METHOD_ROOTS,
        help="Entries of the form name=eval_render_root.",
    )
    parser.add_argument(
        "--view-count",
        type=int,
        default=12,
        help="Number of view_i directories to evaluate. Default: 12.",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="0000.png",
        help="Image filename inside each view_i/dynamic_fixed_view directory. Default: 0000.png.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for ArcFace embeddings.",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=256,
        help="ArcFace detector input size (square).",
    )
    parser.add_argument(
        "--ctx_id",
        type=int,
        default=0,
        help="InsightFace device index (-1 for CPU).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save metrics as JSON.",
    )
    parser.add_argument(
        "--hist-out",
        type=Path,
        default=None,
        help="Optional path to save a histogram of cosine similarities.",
    )
    args = parser.parse_args()

    method_roots = parse_named_paths(args.method_roots, "--method-roots")
    if not args.real_image.exists():
        raise FileNotFoundError(f"Real image not found: {args.real_image}")
    for method_name, method_root in method_roots.items():
        if not method_root.exists():
            raise FileNotFoundError(f"Method root not found for {method_name}: {method_root}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    providers = ["CPUExecutionProvider"] if args.ctx_id < 0 else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    face_app = FaceAnalysis(name="antelopev2", root=str(EVALUATION_ROOT), providers=providers)
    face_app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))

    real_img = np.array(Image.open(args.real_image).convert("RGB"), dtype=np.uint8)
    real_emb = detect_embedding(face_app, real_img, device)
    if real_emb is None:
        raise RuntimeError(f"No face detected in real image: {args.real_image}")

    all_similarities: List[float] = []

    per_method: Dict[str, Dict[str, object]] = {}
    for method_name, method_root in method_roots.items():
        render_images = collect_render_images(method_root, args.view_count, args.image_name)
        method_values: List[float] = []

        for view_idx, image_path in render_images:
            rendered_img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            rend_emb = detect_embedding(face_app, rendered_img, device)
            if rend_emb is None:
                cosine = 0.0
                print(
                    f"Warning: no face detected in render image, using ISD=0.0: {image_path} (view_{view_idx})",
                    file=sys.stderr,
                )
            else:
                cosine = float(torch.dot(real_emb, rend_emb).item())
            all_similarities.append(cosine)
            method_values.append(cosine)

        per_method[method_name] = {
            "root": str(method_root),
            **summarize_method(method_values),
        }

    identity_id = args.real_image.parents[1].name
    metrics: Dict[str, object] = {
        "ID": identity_id,
        "comparison_result": build_comparison_result(per_method),
        "per_method": per_method,
    }

    if all_similarities and args.hist_out is not None:
        save_histogram(all_similarities, args.hist_out)

    print("\nIdentity fidelity (real ↔ render) report:")

    for method_name, method_result in per_method.items():
        print(
            f"  {method_name}: "
            f"ISD_mean={method_result['ISD_mean']:.4f}, ISD_std={method_result['ISD_std']:.4f}, "
            f"min={method_result['min']:.4f}, "
            f"max={method_result['max']:.4f}, values={method_result['values']}"
        )
    if metrics["comparison_result"] is not None:
        print(f"  Comparison: {metrics['comparison_result']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
