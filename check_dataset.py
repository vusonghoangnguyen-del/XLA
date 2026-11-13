"""Kiểm tra chương trình trên bộ ảnh tổng hợp và ground truth đi kèm.

Lệnh chạy từ thư mục code:
    python check_dataset.py --images ../images --ground-truth ../ground_truth
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from gaussian_multiscale_features import extract_features


SIGMAS = (1.0, 2.0, 3.0, 5.0)
FEATURE_FIELDS = (
    "smoothed", "lx", "ly", "lxx", "lxy", "lyy", "gradient",
    "edge_strength", "edge_mask", "ridge", "blob",
)


def check_dataset(images_dir: Path, gt_dir: Path) -> tuple[Counter[str], int, int]:
    image_paths = sorted(images_dir.glob("*.png"))
    gt_paths = sorted(gt_dir.glob("*.png"))
    image_names = {path.name for path in image_paths}
    gt_names = {path.name for path in gt_paths}
    if image_names != gt_names:
        raise ValueError(
            f"Tên không khớp: thiếu GT {sorted(image_names - gt_names)}, "
            f"thừa GT {sorted(gt_names - image_names)}"
        )
    if not image_paths:
        raise ValueError("Không tìm thấy ảnh PNG đầu vào")

    counts: Counter[str] = Counter()
    runs = 0
    for image_path in image_paths:
        kind = image_path.stem.split("_", 1)[0]
        if kind not in {"edge", "ridge", "blob"}:
            raise ValueError(f"Nhóm ảnh không nhận diện được: {image_path.name}")
        source = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_dir / image_path.name), cv2.IMREAD_UNCHANGED)
        if source is None or gt is None:
            raise ValueError(f"Không đọc được cặp ảnh: {image_path.name}")
        if source.shape != gt.shape or source.ndim != 2 or gt.ndim != 2:
            raise ValueError(f"Sai kích thước hoặc số kênh: {image_path.name}")
        if gt.dtype != np.uint8 or not np.isin(np.unique(gt), (0, 255)).all():
            raise ValueError(f"GT phải là mask nhị phân uint8: {image_path.name}")
        if not np.any(gt):
            raise ValueError(f"GT rỗng: {image_path.name}")

        image = source.astype(np.float32) / 255.0
        for sigma in SIGMAS:
            result = extract_features(image, sigma)
            for field in FEATURE_FIELDS:
                output = getattr(result, field)
                if output.shape != source.shape or not np.isfinite(output).all():
                    raise ValueError(f"Đầu ra {field} lỗi ở {image_path.name}, σ={sigma}")
            if not np.isin(np.unique(result.edge_mask), (0, 255)).all():
                raise ValueError(f"edge_mask không nhị phân: {image_path.name}, σ={sigma}")
            runs += 1
        counts[kind] += 1

    centers_path = gt_dir / "blob_centers.csv"
    with centers_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    centers: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        name = f"{row['sample_id']}.png"
        x, y = int(row["x"]), int(row["y"])
        centers[name].append((x, y))
    if set(centers) != {path.name for path in image_paths if path.stem.startswith("blob_") }:
        raise ValueError("Danh sách ảnh blob trong CSV không khớp")
    for name, points in centers.items():
        gt = cv2.imread(str(gt_dir / name), cv2.IMREAD_GRAYSCALE)
        if gt is None or len(points) != 6 or len(set(points)) != 6:
            raise ValueError(f"Tọa độ tâm blob không hợp lệ: {name}")
        for x, y in points:
            if not (0 <= x < gt.shape[1] and 0 <= y < gt.shape[0] and gt[y, x] == 255):
                raise ValueError(f"Tâm blob không khớp mask: {name}, ({x}, {y})")
    return counts, runs, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("../images"))
    parser.add_argument("--ground-truth", type=Path, default=Path("../ground_truth"))
    args = parser.parse_args()
    counts, runs, centers = check_dataset(args.images, args.ground_truth)
    print(f"Images passed: {sum(counts.values())} (edge={counts['edge']}, "
          f"ridge={counts['ridge']}, blob={counts['blob']})")
    print(f"Algorithm runs passed: {runs} (sigmas={SIGMAS})")
    print(f"Blob centers matching masks: {centers}")


if __name__ == "__main__":
    main()
