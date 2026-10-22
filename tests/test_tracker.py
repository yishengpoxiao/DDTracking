import logging
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tracker.tracker import Tracker


class TrackerTests(unittest.TestCase):
    def test_lengths_use_physical_affine_for_half_voxel_steps(self):
        angle = np.deg2rad(37)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([2.0, 1.0, 3.0])
        points = np.array([[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0, 0]])

        np.testing.assert_allclose(Tracker._streamline_lengths_mm([points], affine), [3.0])

    def test_step_cap_uses_oblique_affine_and_step_size(self):
        angle = np.deg2rad(37)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        affine = np.eye(4)
        affine[:3, :3] = rotation @ np.diag([2.0, 1.0, 3.0])

        self.assertEqual(Tracker._max_tracking_steps(10.0, 0.5, affine), 20)

    def test_checkpoint_load_error_is_raised(self):
        tracker = Tracker.__new__(Tracker)
        tracker.logger = logging.getLogger("tracker-load-test")
        tracker.device = "cpu"
        tracker.model = object()
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            tracker.model_file = checkpoint.name
            with patch.object(torch, "load", side_effect=RuntimeError("invalid checkpoint")) as load:
                with self.assertRaisesRegex(RuntimeError, "invalid checkpoint"):
                    tracker.load_model()

        load.assert_called_once_with(tracker.model_file, map_location="cpu", weights_only=True)
        self.assertIsNone(tracker.model)

    def test_empty_seed_batch_returns_empty_result(self):
        tracker = Tracker.__new__(Tracker)
        tracker.track_data = SimpleNamespace(
            wm_mask=None,
            brain_mask=np.ones((3, 3, 3), dtype=bool),
            FA=np.ones((3, 3, 3)),
            volume=np.zeros((3, 3, 3, 1), dtype=np.float32),
            dwi=SimpleNamespace(affine=np.eye(4)),
        )
        tracker.seed_points = np.empty((0, 3))
        tracker.track_batch_size = 4
        tracker.device = "cpu"
        tracker.logger = logging.getLogger("tracker-empty-batch-test")

        self.assertEqual(len(tracker.streamline_tracking()), 0)

    def test_reverse_warmup_preserves_all_previous_directions(self):
        tracker = Tracker.__new__(Tracker)
        tracker.use_previous_dire = 3
        tracker.device = 'cpu'
        tracker.model = SimpleNamespace(_apply_rnn_layers=lambda volume, points, state:
                                        (None, torch.ones((1, points.shape[0], 2))))
        reversed_path = np.array([[[0., 0., 0.], [1., 0., 0.],
                                  [1., 1., 0.], [1., 1., 1.]]], dtype=np.float32)
        _, history = tracker._prepare_reverse_state(reversed_path, torch.empty(0))
        np.testing.assert_allclose(history[0], [[0., 0., 1.], [0., 1., 0.], [1., 0., 0.]], atol=2e-6)

    def test_reverse_rejects_sharp_turn_at_seed(self):
        tracker = Tracker.__new__(Tracker)
        tracker.use_previous_dire = 1
        tracker.device = 'cpu'
        tracker.look_ahead = False
        tracker.step_size = .5
        tracker.max_angle = 45
        tracker.logger = logging.getLogger('reverse-angle-test')
        tracker.model = SimpleNamespace(generate=lambda volume, points, state:
                                        (torch.tensor([[-1., 0., 0.]]), torch.ones((1, 1, 2))))
        result = tracker._reverse_tracking(np.array([[2., 2., 2.]], dtype=np.float32),
                                          torch.empty(0), np.ones((5, 5, 5), dtype=bool),
                                          None, np.array([[[1., 0., 0.]]]), np.array([2]))
        self.assertTrue(np.isnan(result).all())


if __name__ == "__main__":
    unittest.main()
