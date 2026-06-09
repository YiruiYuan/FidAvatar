#!/usr/bin/env python3
"""
Evaluate identity similarity using view-specific reference images and ArcFace embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from insightface.app import FaceAnalysis

    _HAS_INSIGHTFACE = True
except ImportError:
    FaceAnalysis = Any  # type: ignore[assignment]
    _HAS_INSIGHTFACE = False

DEFAULT_LEFT_IMAGE = Path("/home/eric/Desktop/mycode/Fatediffavatar/data/insta/yyr/isd_image/left.png")
EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_RIGHT_IMAGE = Path("/home/eric/Desktop/mycode/Fatediffavatar/data/insta/yyr/isd_image/right.png")
DEFAULT_FRONT_IMAGE = Path("/home/eric/Desktop/mycode/Fatediffavatar/data/insta/yyr/isd_image/front.png")
DEFAULT_METHOD_ROOTS = [
    "mycode=/home/eric/Desktop/mycode/Fatediffavatar/workspace/insta/yyr/completion/eval_render",
    "baseline=/home/eric/Desktop/mycode/Fatediffavatar/workspace/insta/yyr/completion_arc/eval_render",
]
DEFAULT_OUTPUT = Path("/home/eric/Desktop/mycode/Fatediffavatar/workspace/insta/yyr/identity_fidelity_arc.json")

SKIP_VIEW_IDS = {4, 5, 6}
LEFT_VIEW_IDS = {0, 1, 2, 3}
RIGHT_VIEW_IDS = {7, 8, 9, 10}
FRONT_VIEW_IDS = {11}


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
    if not values:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 3.0))
    plt.hist(values, bins=50, range=(-1.0, 1.0), color="seagreen", edgecolor="black")
    plt.xlabel("Cosine similarity (reference ↔ render)")
    plt.ylabel("Count")
    plt.title("Identity similarity distribution")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def collect_render_images(method_root: Path, view_count: int, image_name: str) -> List[Tuple[int, Path]]:
    render_images: List[Tuple[int, Path]] = []
    for view_idx in range(view_count):
        if view_idx in SKIP_VIEW_IDS:
            continue
        image_path = method_root / f"view_{view_idx}" / "dynamic_fixed_view" / image_name
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


def build_comparison_result(per_method: Dict[str, Dict[str, object]], score_key: str) -> str | None:
    method_names = sorted(per_method.keys())
    if len(method_names) < 2:
        return None

    ranked = sorted(
        (
            (name, result[score_key])
            for name, result in per_method.items()
            if result.get(score_key) is not None
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked) < 2:
        return None

    higher_name, higher_score = ranked[0]
    lower_name, lower_score = ranked[1]
    if abs(higher_score - lower_score) < 1e-8:
        if len(method_names) == 2:
            first_name, second_name = method_names
            return f"{first_name} and {second_name} are equal"
        return f"{higher_name} and {lower_name} are equal"
    return f"{higher_name} is higher than {lower_name} by {higher_score - lower_score:.4f}"


def reference_name_for_view(view_idx: int) -> str:
    if view_idx in LEFT_VIEW_IDS:
        return "left"
    if view_idx in RIGHT_VIEW_IDS:
        return "right"
    if view_idx in FRONT_VIEW_IDS:
        return "front"
    raise ValueError(f"View {view_idx} does not have a configured reference image")


def load_reference_embeddings(
    face_app: FaceAnalysis,
    device: torch.device,
    reference_images: Dict[str, Path],
) -> Dict[str, Dict[str, object]]:
    reference_infos: Dict[str, Dict[str, object]] = {}
    for name, image_path in reference_images.items():
        if not image_path.exists():
            raise FileNotFoundError(f"Reference image not found for {name}: {image_path}")
        image_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        embedding = detect_embedding(face_app, image_rgb, device)
        if embedding is None:
            raise RuntimeError(f"No face detected in reference image for {name}: {image_path}")
        reference_infos[name] = {
            "path": str(image_path),
            "embedding": embedding,
        }
    return reference_infos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identity similarity (ArcFace) using view-specific reference images and rendered views."
    )
    parser.add_argument(
        "--left-image",
        type=Path,
        default=DEFAULT_LEFT_IMAGE,
        help="Reference image for view_0-3.",
    )
    parser.add_argument(
        "--right-image",
        type=Path,
        default=DEFAULT_RIGHT_IMAGE,
        help="Reference image for view_7-10.",
    )
    parser.add_argument(
        "--front-image",
        type=Path,
        default=DEFAULT_FRONT_IMAGE,
        help="Reference image for view_11.",
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

    if not _HAS_INSIGHTFACE:
        raise ImportError("insightface is required to run this script. Please install it in the current environment.")

    method_roots = parse_named_paths(args.method_roots, "--method-roots")
    for method_name, method_root in method_roots.items():
        if not method_root.exists():
            raise FileNotFoundError(f"Method root not found for {method_name}: {method_root}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    providers = ["CPUExecutionProvider"] if args.ctx_id < 0 else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    face_app = FaceAnalysis(name="antelopev2", root=str(EVALUATION_ROOT), providers=providers)
    face_app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))

    reference_images = {
        "left": args.left_image,
        "right": args.right_image,
        "front": args.front_image,
    }
    reference_infos = load_reference_embeddings(face_app, device, reference_images)

    all_similarities: List[float] = []
    per_method: Dict[str, Dict[str, object]] = {}
    front_per_method: Dict[str, Dict[str, object]] = {}

    for method_name, method_root in method_roots.items():
        render_images = collect_render_images(method_root, args.view_count, args.image_name)
        method_values: List[float] = []
        per_view: List[Dict[str, object]] = []
        front_value: float | None = None

        for view_idx, image_path in render_images:
            reference_name = reference_name_for_view(view_idx)
            rendered_img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            rend_emb = detect_embedding(face_app, rendered_img, device)
            if rend_emb is None:
                raise RuntimeError(f"No face detected in render image: {image_path} (view_{view_idx})")

            reference_emb = reference_infos[reference_name]["embedding"]
            cosine = float(torch.dot(reference_emb, rend_emb).item())
            all_similarities.append(cosine)
            method_values.append(cosine)
            per_view.append(
                {
                    "view_id": view_idx,
                    "reference": reference_name,
                    "reference_path": reference_infos[reference_name]["path"],
                    "value": cosine,
                }
            )

            if reference_name == "front":
                front_value = cosine

        per_method[method_name] = {
            "root": str(method_root),
            **summarize_method(method_values),
            "per_view": per_view,
        }
        front_per_method[method_name] = {
            "view_id": 11,
            "reference_path": reference_infos["front"]["path"],
            "ISD": front_value,
        }

    identity_id = args.front_image.parents[1].name
    metrics: Dict[str, object] = {
        "ID": identity_id,
        "reference_images": {name: info["path"] for name, info in reference_infos.items()},
        "view_reference_mapping": {
            "view_0-3": "left",
            "view_7-10": "right",
            "view_11": "front",
        },
        "comparison_result": build_comparison_result(per_method, "ISD_mean"),
        "front_comparison_result": build_comparison_result(front_per_method, "ISD"),
        "front_per_method": front_per_method,
        "per_method": per_method,
    }

    if all_similarities and args.hist_out is not None:
        save_histogram(all_similarities, args.hist_out)

    print("\nIdentity similarity (reference ↔ render) report:")

    for method_name, method_result in per_method.items():
        print(
            f"  {method_name}: "
            f"ISD_mean={method_result['ISD_mean']:.4f}, ISD_std={method_result['ISD_std']:.4f}, "
            f"min={method_result['min']:.4f}, "
            f"max={method_result['max']:.4f}, values={method_result['values']}"
        )
    if metrics["comparison_result"] is not None:
        print(f"  Comparison: {metrics['comparison_result']}")

    print("\nFront-view (front.png ↔ view_11) report:")
    for method_name, method_result in front_per_method.items():
        front_score = method_result["ISD"]
        if front_score is None:
            print(f"  {method_name}: front_ISD=None")
            continue
        print(f"  {method_name}: front_ISD={front_score:.4f}")
    if metrics["front_comparison_result"] is not None:
        print(f"  Front comparison: {metrics['front_comparison_result']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
