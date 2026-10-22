import unittest
from unittest.mock import patch

import numpy as np
import torch

from utils.utils import (
    init_seeds_in_voxel,
    interpolate_volume_in_neighborhood_torch,
    is_within_mask,
    fit_from_model,
    minmax_normalize,
    zero_mean_unit_variance_normalize,
)


class UtilityGeometryTests(unittest.TestCase):
    def test_mask_membership_uses_nearest_voxel_center_and_half_voxel_bounds(self):
        mask = np.zeros((3, 1, 1), dtype=bool)
        mask[1, 0, 0] = True
        positions = np.array([
            [-0.49, 0., 0.], [-0.5, 0., 0.], [-0.51, 0., 0.],
            [0.49, 0., 0.], [0.5, 0., 0.], [1.49, 0., 0.],
            [2.49, 0., 0.], [2.5, 0., 0.], [np.nan, 0., 0.],
        ])
        np.testing.assert_array_equal(
            is_within_mask(positions, mask),
            [False, False, False, False, True, True, False, False, False],
        )

    def test_empty_voxel_seed_mask_returns_n_by_three(self):
        seeds = init_seeds_in_voxel(np.zeros((2, 3, 4), dtype=bool), 2)
        self.assertEqual(seeds.shape, (0, 3))
        self.assertEqual(seeds.dtype, np.float32)

    def test_neighborhood_interpolation_preserves_xyz_and_offset_order(self):
        x, y, z = np.mgrid[:4, :5, :6]
        values = (100 * x + 10 * y + z).astype(np.float32)
        volume = torch.from_numpy(values).unsqueeze(0).unsqueeze(0)
        coordinates = torch.tensor([[1.25, 2.5, 3.75]], dtype=torch.float32)
        offsets = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)

        samples = interpolate_volume_in_neighborhood_torch(
            volume, coordinates, [3., 4., 5.], offsets)
        expected = np.array([[253.75], [163.75], [154.75]], dtype=np.float32)
        np.testing.assert_allclose(samples.numpy(), expected, atol=1e-5)

    def test_interpolation_preserves_singleton_point_and_channel_dimensions(self):
        volume = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
        samples = interpolate_volume_in_neighborhood_torch(
            volume, torch.tensor([[1., 1., 1.]]), [1., 1., 1.])
        self.assertEqual(samples.shape, (1, 1))
        self.assertEqual(float(samples[0, 0]), 7.)

    def test_interpolation_handles_singleton_spatial_axis(self):
        values = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        volume = torch.from_numpy(values).unsqueeze(0).unsqueeze(0)
        samples = interpolate_volume_in_neighborhood_torch(
            volume, torch.tensor([[0., 1., 2.]]), [0., 2., 2.])
        self.assertEqual(float(samples[0, 0]), 5.)


class NormalizationTests(unittest.TestCase):
    def test_minmax_normalize_does_not_mutate_input(self):
        source = np.array([[[[2., 4.], [4., 8.]]]], dtype=np.float32)
        original = source.copy()
        normalized = minmax_normalize(source, np.ones(source.shape[:3], dtype=bool))
        np.testing.assert_array_equal(source, original)
        np.testing.assert_allclose(normalized, [[[[0., 1.], [0., 1.]]]])

    def test_minmax_normalize_returns_fractional_values_for_integer_input(self):
        source = np.array([[[[0, 1, 3]]]], dtype=np.int16)
        normalized = minmax_normalize(source, np.ones(source.shape[:3], dtype=bool))
        np.testing.assert_allclose(normalized, [[[[0., 1 / 3, 1.]]]])
        self.assertTrue(np.issubdtype(normalized.dtype, np.floating))

    def test_zero_mean_normalize_rejects_empty_mask(self):
        data = np.ones((2, 2, 2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'contains no voxels'):
            zero_mean_unit_variance_normalize(data, np.zeros((2, 2, 2), dtype=bool))

    def test_fodf_fit_does_not_mutate_supplied_mask(self):
        class Model:
            n = np.array([0])

            def fit(self, samples):
                return float(np.sum(samples))

        class InlinePool:
            def __init__(self, _count):
                pass

            @staticmethod
            def map(function, items):
                return [function(item) for item in items]

            @staticmethod
            def close():
                pass

            @staticmethod
            def join():
                pass

        data = np.ones((2, 2, 2, 3), dtype=np.float32)
        mask = np.ones((2, 2, 2), dtype=bool)
        mask[0] = False
        original = mask.copy()
        with patch('utils.utils.multiprocessing.Pool', InlinePool):
            fit_from_model(Model(), data, mask=mask, nbr_processes=1)
        np.testing.assert_array_equal(mask, original)


if __name__ == '__main__':
    unittest.main()
