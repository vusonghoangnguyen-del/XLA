from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ScaleFeatures:
    """Cac anh trung gian va dap ung dac trung tai mot ty le sigma."""

    sigma: float
    smoothed: np.ndarray
    lx: np.ndarray
    ly: np.ndarray
    lxx: np.ndarray
    lxy: np.ndarray
    lyy: np.ndarray
    gradient: np.ndarray
    edge_strength: np.ndarray
    edge_mask: np.ndarray
    ridge: np.ndarray
    blob: np.ndarray


def load_grayscale(path: Path) -> np.ndarray:
    """Doc anh xam va dua ve float32 trong khoang [0, 1]."""

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Khong the doc anh: {path}")
    return image.astype(np.float32) / 255.0


def gaussian_derivative_kernels(
    sigma: float,
    truncate: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tao kernel Gaussian 1D va dao ham bac 1, bac 2.

    g1 va g2 duoc nhan lan luot voi sigma va sigma^2 de chuan hoa theo
    ty le. Viec ep tong kernel dao ham ve 0 lam giam sai so roi rac.
    """

    if sigma <= 0:
        raise ValueError("Sigma phai lon hon 0.")

    radius = max(1, int(math.ceil(truncate * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)

    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g /= g.sum()

    g1 = -(x / (sigma * sigma)) * g
    g2 = ((x * x - sigma * sigma) / (sigma**4)) * g

    # Dieu chinh sai so do cat ngan va roi rac hoa kernel.
    g1 -= g1.mean()
    g2 -= g2.mean()

    # Dao ham chuan hoa theo ty le: sigma^n * L_(bac n).
    g1 *= sigma
    g2 *= sigma * sigma

    return (
        g.astype(np.float32),
        g1.astype(np.float32),
        g2.astype(np.float32),
    )


def separable_filter(
    image: np.ndarray,
    kernel_x: np.ndarray,
    kernel_y: np.ndarray,
) -> np.ndarray:
    """Tich chap tach bien voi dieu kien bien phan xa."""

    return cv2.sepFilter2D(
        image,
        cv2.CV_32F,
        kernel_x,
        kernel_y,
        borderType=cv2.BORDER_REFLECT101,
    )


def non_maximum_suppression(
    magnitude: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
) -> np.ndarray:
    """Lam manh duong bien bang cuc dai cuc bo theo huong gradient."""

    angle = (np.rad2deg(np.arctan2(ly, lx)) + 180.0) % 180.0
    padded = np.pad(magnitude, 1, mode="constant")

    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    up_left = padded[:-2, :-2]
    up_right = padded[:-2, 2:]
    down_left = padded[2:, :-2]
    down_right = padded[2:, 2:]

    keep = np.zeros(magnitude.shape, dtype=bool)

    horizontal = (angle < 22.5) | (angle >= 157.5)
    diagonal_45 = (angle >= 22.5) & (angle < 67.5)
    vertical = (angle >= 67.5) & (angle < 112.5)
    diagonal_135 = (angle >= 112.5) & (angle < 157.5)

    keep |= horizontal & (center >= left) & (center >= right)
    keep |= diagonal_45 & (center >= up_right) & (center >= down_left)
    keep |= vertical & (center >= up) & (center >= down)
    keep |= diagonal_135 & (center >= up_left) & (center >= down_right)

    result = np.where(keep, magnitude, 0.0)
    result[[0, -1], :] = 0.0
    result[:, [0, -1]] = 0.0
    return result.astype(np.float32)


def threshold_edges(edge_strength: np.ndarray, percentile: float) -> np.ndarray:
    """Nguong hoa dap ung bien theo phan vi cua cac gia tri duong."""

    # Anh phang van co the tao dap ung ~1e-8 do sai so float32.
    # Voi anh dau vao da chuan hoa [0, 1], bo cac dap ung duoi 1e-6.
    positive = edge_strength[edge_strength > 1e-6]
    if positive.size == 0:
        return np.zeros(edge_strength.shape, dtype=np.uint8)

    threshold = float(np.percentile(positive, percentile))
    return np.where(edge_strength >= threshold, 255, 0).astype(np.uint8)


def hessian_eigenvalues(
    lxx: np.ndarray,
    lxy: np.ndarray,
    lyy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tinh hai tri rieng Hessian, sap xep theo tri tuyet doi tang dan."""

    trace = lxx + lyy
    discriminant = np.sqrt(np.maximum((lxx - lyy) ** 2 + 4.0 * lxy**2, 0.0))
    eigen_a = 0.5 * (trace - discriminant)
    eigen_b = 0.5 * (trace + discriminant)

    swap = np.abs(eigen_a) > np.abs(eigen_b)
    lambda_small = np.where(swap, eigen_b, eigen_a)
    lambda_large = np.where(swap, eigen_a, eigen_b)
    return lambda_small, lambda_large


def ridge_response(
    lambda_small: np.ndarray,
    lambda_large: np.ndarray,
    polarity: str = "bright",
    beta: float = 0.5,
) -> np.ndarray:
    """Tinh dap ung duong go tu tri rieng Hessian (dang Frangi 2D)."""

    epsilon = np.finfo(np.float32).eps
    ratio = np.abs(lambda_small) / (np.abs(lambda_large) + epsilon)
    structure = np.sqrt(lambda_small**2 + lambda_large**2)
    c = max(float(np.percentile(structure, 95)), epsilon)

    response = np.exp(-(ratio**2) / (2.0 * beta**2))
    response *= 1.0 - np.exp(-(structure**2) / (2.0 * c**2))

    if polarity == "bright":
        response[lambda_large >= 0] = 0.0
    elif polarity == "dark":
        response[lambda_large <= 0] = 0.0
    elif polarity != "both":
        raise ValueError("polarity phai la 'bright', 'dark' hoac 'both'.")

    return response.astype(np.float32)


def extract_features(
    image: np.ndarray,
    sigma: float,
    edge_percentile: float = 75.0,
    ridge_polarity: str = "bright",
) -> ScaleFeatures:
    """Tinh toan tat ca dac trung tai mot ty le."""

    g, g1, g2 = gaussian_derivative_kernels(sigma)

    smoothed = separable_filter(image, g, g)
    lx = separable_filter(image, g1, g)
    ly = separable_filter(image, g, g1)
    lxx = separable_filter(image, g2, g)
    lxy = separable_filter(image, g1, g1)
    lyy = separable_filter(image, g, g2)

    gradient = cv2.magnitude(lx, ly)
    edge_strength = non_maximum_suppression(gradient, lx, ly)
    edge_mask = threshold_edges(edge_strength, edge_percentile)

    lambda_small, lambda_large = hessian_eigenvalues(lxx, lxy, lyy)
    ridge = ridge_response(lambda_small, lambda_large, ridge_polarity)

    # Laplacian of Gaussian chuan hoa theo ty le; lay tri tuyet doi de bat
    # ca dom sang va dom toi.
    blob = np.abs(lxx + lyy).astype(np.float32)

    return ScaleFeatures(
        sigma=sigma,
        smoothed=smoothed,
        lx=lx,
        ly=ly,
        lxx=lxx,
        lxy=lxy,
        lyy=lyy,
        gradient=gradient,
        edge_strength=edge_strength,
        edge_mask=edge_mask,
        ridge=ridge,
        blob=blob,
    )


def normalize_uint8(image: np.ndarray, absolute: bool = False) -> np.ndarray:
    """Chuan hoa mot anh float de luu/quan sat duoi dang uint8."""

    data = np.abs(image) if absolute else image
    minimum = float(data.min())
    maximum = float(data.max())
    if maximum - minimum <= np.finfo(np.float32).eps:
        return np.zeros(data.shape, dtype=np.uint8)
    normalized = (data - minimum) / (maximum - minimum)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def sigma_label(sigma: float) -> str:
    """Tao nhan sigma an toan cho ten file."""

    return f"{sigma:g}".replace(".", "p")


def save_scale_results(result: ScaleFeatures, output_dir: Path) -> None:
    """Luu cac anh ket qua cua mot ty le."""

    label = sigma_label(result.sigma)
    images = {
        "gaussian": normalize_uint8(result.smoothed),
        "gradient": normalize_uint8(result.gradient),
        "edge_strength": normalize_uint8(result.edge_strength),
        "edge_mask": result.edge_mask,
        "ridge": normalize_uint8(result.ridge),
        "blob_log": normalize_uint8(result.blob),
        "lx": normalize_uint8(result.lx, absolute=True),
        "ly": normalize_uint8(result.ly, absolute=True),
        "lxx": normalize_uint8(result.lxx, absolute=True),
        "lxy": normalize_uint8(result.lxy, absolute=True),
        "lyy": normalize_uint8(result.lyy, absolute=True),
    }
    for name, image in images.items():
        cv2.imwrite(str(output_dir / f"sigma_{label}_{name}.png"), image)


def save_summary(
    original: np.ndarray,
    results: list[ScaleFeatures],
    output_path: Path,
) -> None:
    """Tao bang tong hop de so sanh cac ty le."""

    columns = len(results)
    figure, axes = plt.subplots(5, columns, figsize=(4 * columns, 16), squeeze=False)
    row_titles = ["Gaussian", "Gradient", "Edge", "Ridge", "Blob (|LoG|)"]

    for column, result in enumerate(results):
        views = [
            result.smoothed,
            result.gradient,
            result.edge_mask,
            result.ridge,
            result.blob,
        ]
        for row, view in enumerate(views):
            axes[row, column].imshow(view, cmap="gray")
            if row == 0:
                axes[row, column].set_title(f"Sigma = {result.sigma:g}")
            if column == 0:
                axes[row, column].set_ylabel(row_titles[row])
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    figure.suptitle("Phat hien dac trung anh da ty le bang dao ham Gaussian")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    cv2.imwrite(
        str(output_path.parent / "original_grayscale.png"),
        normalize_uint8(original),
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phat hien edge, ridge va blob da ty le bang dao ham Gaussian."
    )
    parser.add_argument("input", type=Path, help="Duong dan anh dau vao.")
    parser.add_argument(
        "--sigmas",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 3.0, 5.0],
        help="Danh sach sigma (mac dinh: 1 2 3 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multiscale_results"),
        help="Thu muc luu ket qua.",
    )
    parser.add_argument(
        "--edge-percentile",
        type=float,
        default=75.0,
        help="Phan vi nguong bien, trong khoang [0, 100] (mac dinh: 75).",
    )
    parser.add_argument(
        "--ridge-polarity",
        choices=["bright", "dark", "both"],
        default="bright",
        help="Phat hien duong go sang, toi hoac ca hai (mac dinh: bright).",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if not arguments.sigmas or any(sigma <= 0 for sigma in arguments.sigmas):
        raise ValueError("Moi gia tri sigma phai lon hon 0.")
    if not 0.0 <= arguments.edge_percentile <= 100.0:
        raise ValueError("edge-percentile phai nam trong khoang [0, 100].")


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)

    image = load_grayscale(arguments.input)
    arguments.output.mkdir(parents=True, exist_ok=True)

    results = []
    for sigma in arguments.sigmas:
        print(f"Dang xu ly sigma = {sigma:g} ...")
        result = extract_features(
            image,
            sigma,
            edge_percentile=arguments.edge_percentile,
            ridge_polarity=arguments.ridge_polarity,
        )
        save_scale_results(result, arguments.output)
        results.append(result)

    summary_path = arguments.output / "multiscale_summary.png"
    save_summary(image, results, summary_path)

    print(f"Hoan tat. Ket qua duoc luu tai: {arguments.output.resolve()}")
    print(f"Anh tong hop: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
