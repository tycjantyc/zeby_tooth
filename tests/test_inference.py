"""Integration checks for geometry, boundaries, validation and binary output."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn

from inference.pipeline import SegmentationPipeline, _load_model
from inference.volume import predict_volume, read_volume, require_same_geometry


class IntensityModel(nn.Module):
    def forward(self, x):
        assert not torch.is_grad_enabled()
        return x[:, :1] * 8 - 4


class CentroidModel(nn.Module):
    def forward(self, x):
        return x[:, :1] * 12 - 6


class SeparatorModel(nn.Module):
    def forward(self, x):
        return torch.cat((torch.zeros_like(x), torch.full_like(x, -10), x * 12 - 6), dim=1)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, array, origin=(13.0, -8.0, 22.0)):
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.3, 0.4, 0.5))
        image.SetOrigin(origin)
        image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        path = self.root / name
        sitk.WriteImage(image, str(path))
        return path

    def pipeline(self):
        pipeline = SegmentationPipeline(device="cpu")
        pipeline._models = {"centroid": (CentroidModel(), 1, 1),
                            "separator": (SeparatorModel(), 1, 3),
                            "canals": (IntensityModel(), 1, 1)}
        return pipeline

    def test_sliding_windows_cover_small_odd_and_tail_shapes(self):
        rng = np.random.default_rng(4)
        for shape in ((3, 5, 7), (8, 8, 8), (13, 10, 11)):
            volume = rng.random((1, *shape), dtype=np.float32)
            result = predict_volume(IntensityModel(), volume, (8,) * 3, (5,) * 3, 1, device="cpu")
            expected = torch.from_numpy(volume * 8 - 4).sigmoid().numpy()
            np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_coordinate_channels_and_dynamic_class_count(self):
        class Coordinates(nn.Module):
            def forward(self, x):
                self.last = x
                return x[:, 1:]
        model = Coordinates()
        result = predict_volume(model, np.zeros((1, 7, 7, 7), np.float32),
                                (8,) * 3, (4,) * 3, 3, device="cpu", coordinates=True)
        self.assertEqual(model.last.shape[1], 4)
        self.assertEqual(result.shape, (3, 7, 7, 7))
        self.assertTrue(np.isfinite(result).all())

    def test_single_class_legacy_evaluate_returns_foreground(self):
        from evaluation.evaluate import evaluate
        path = self.write("binary.nii.gz", np.ones((3, 5, 7), np.float32))
        labels = evaluate(IntensityModel(), path, (8,) * 3, (4,) * 3,
                          num_classes=1, normalization_function=lambda x: x, device="cpu")
        self.assertEqual(labels.shape, (3, 5, 7))
        self.assertTrue(labels.all())

    def test_agh_legacy_respects_class_count_and_coordinate_inputs(self):
        from evaluation.evaluate_agh_segmentor import evaluate_agh
        class ThreeClasses(nn.Module):
            def forward(self, x):
                if x.shape[1] != 4:
                    raise ValueError("Expected intensity and three coordinates")
                return torch.cat((x[:, :1] - 10, x[:, :1] - 5, x[:, :1] + 10), dim=1)
        path = self.write("classes.nii.gz", np.ones((3, 5, 7), np.float32))
        labels = evaluate_agh(ThreeClasses(), path, (8,) * 3, (4,) * 3,
                              num_classes=3, normalization_function=lambda x: x, device="cpu")
        self.assertTrue((labels == 2).all())

    def test_pipeline_separates_and_preserves_geometry(self):
        shape = (17, 15, 13)
        mask = np.zeros(shape, np.uint8)
        mask[0:12, 2:13, 3:11] = 1
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        cbct = self.write("cbct.nii.gz", image)
        teeth = self.write("teeth.nii", mask)
        results = self.pipeline().run(cbct, self.root / "out", teeth_mask_path=teeth)
        self.assertEqual(set(results), {"teeth_mask", "teeth_instances", "canals_mask"})
        reference, _ = read_volume(cbct)
        for key, path in results.items():
            self.assertTrue(Path(path).is_absolute())
            result_image, array = read_volume(path)
            require_same_geometry(reference, result_image)
            self.assertEqual(array.shape, shape)
            self.assertFalse(array[mask == 0].any())
            self.assertTrue(array.any(), key)
        np.testing.assert_array_equal(read_volume(results["teeth_instances"])[1], mask)
        self.assertEqual(read_volume(results["teeth_instances"])[1].dtype, np.uint16)

    def test_large_tooth_and_existing_instances_keep_ids_and_tail(self):
        shape = (103, 9, 11)
        instances = np.full(shape, 42, np.uint16)
        image = np.broadcast_to(np.arange(shape[0], dtype=np.float32)[:, None, None], shape).copy()
        cbct = self.write("cbct.nii.gz", image)
        mask = self.write("instances.nii.gz", instances)
        pipeline = self.pipeline()
        results = pipeline.run(cbct, self.root / "out", instances_path=mask)
        np.testing.assert_array_equal(read_volume(results["teeth_instances"])[1], instances)
        canals = read_volume(results["canals_mask"])[1]
        self.assertTrue(canals[-1].all())
        self.assertFalse(canals[0].any())
        with self.assertRaises(FileExistsError):
            pipeline.run(cbct, self.root / "out", instances_path=mask)

    def test_empty_mask_requires_no_models(self):
        image = np.zeros((5, 7, 9), np.float32)
        cbct = self.write("cbct.nii.gz", image)
        mask = self.write("empty.nii.gz", image.astype(np.uint8))
        pipeline = SegmentationPipeline(weights_dir=self.root / "missing", device="cpu")
        results = pipeline.run(cbct, self.root / "empty-out", teeth_mask_path=mask)
        self.assertTrue(all(not read_volume(p)[1].any() for p in results.values()))

    def test_invalid_inputs_and_missing_weights_fail(self):
        cbct = self.write("cbct.nii.gz", np.zeros((5, 7, 9), np.float32))
        mismatch = self.write("mask.nii.gz", np.ones((5, 7, 9), np.uint8), origin=(0.0, 0.0, 0.0))
        pipeline = self.pipeline()
        with self.assertRaises(ValueError):
            pipeline.run(cbct, self.root / "missing-mask")
        with self.assertRaises(ValueError):
            pipeline.run(cbct, self.root / "wrong-geometry", teeth_mask_path=mismatch)
        with self.assertRaises(FileNotFoundError):
            _load_model(self.root / "missing.pt", "cpu")
        with self.assertRaises(ValueError):
            predict_volume(IntensityModel(), np.zeros((1, 5, 5, 5), np.float32),
                           (8,) * 3, (9,) * 3, 1, device="cpu")


if __name__ == "__main__":
    unittest.main()
