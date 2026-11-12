"""Kiểm tra chức năng cơ bản, không cần bộ ảnh ground truth."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from gaussian_multiscale_features import (
    extract_features,
    gaussian_derivative_kernels,
    load_grayscale,
    save_scale_results,
)


class GaussianMultiscaleSmokeTests(unittest.TestCase):
    def test_kernels_have_expected_sums(self) -> None:
        for sigma in (1.0, 2.0, 3.0, 5.0):
            g, g1, g2 = gaussian_derivative_kernels(sigma)
            self.assertEqual(len(g) % 2, 1)
            self.assertAlmostEqual(float(g.sum()), 1.0, places=5)
            self.assertAlmostEqual(float(g1.sum()), 0.0, places=5)
            self.assertAlmostEqual(float(g2.sum()), 0.0, places=5)

    def test_feature_shapes_ranges_and_saved_files(self) -> None:
        image = np.zeros((64, 80), dtype=np.float32)
        image[16:48, 20:60] = 1.0
        result = extract_features(image, sigma=2.0)
        names = (
            "smoothed", "lx", "ly", "lxx", "lxy", "lyy",
            "gradient", "edge_strength", "edge_mask", "ridge", "blob",
        )
        for name in names:
            value = getattr(result, name)
            self.assertEqual(value.shape, image.shape, name)
            self.assertTrue(np.isfinite(value).all(), name)
        self.assertEqual(result.edge_mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(result.edge_mask)).issubset({0, 255}))
        self.assertGreater(int(np.count_nonzero(result.edge_mask)), 0)

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            save_scale_results(result, output)
            files = list(output.glob("sigma_2_*.png"))
            self.assertEqual(len(files), 11)
            for path in files:
                saved = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(saved, path.name)
                self.assertEqual(saved.shape, image.shape, path.name)

    def test_input_error_and_constant_image(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_grayscale(Path("this_file_does_not_exist.png"))
        with self.assertRaises(ValueError):
            gaussian_derivative_kernels(0.0)
        result = extract_features(np.full((32, 32), 0.5, np.float32), 2.0)
        self.assertEqual(int(np.count_nonzero(result.edge_mask)), 0)


if __name__ == "__main__":
    unittest.main()
