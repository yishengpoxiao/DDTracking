import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.conditional_unet1d import ConditionalUnet1D
from model.ddtracking import DDTracking
from model.gaussion_diffusion import D2MP_OB, VarianceSchedule


class PredictionNet(nn.Module):
    def forward(self, sample, timesteps, global_cond, local_cond=None):
        return global_cond[:, :3]


class ModelTests(unittest.TestCase):
    def test_masked_losses_match_selected_points_and_unmasked_mean(self):
        target = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        prediction = torch.tensor([[.2, .4, .3], [4., -5., 9.]], requires_grad=True)
        times = torch.tensor([.25, .75])
        with patch('torch.randn_like', side_effect=torch.zeros_like):
            for loss_type in ('l1', 'l2', 'cosine', 'w_cosine'):
                with self.subTest(loss_type=loss_type):
                    diffusion = D2MP_OB(loss_type, PredictionNet(), VarianceSchedule(10))
                    full = diffusion(target, prediction, t=times)
                    all_selected = diffusion(target, prediction, t=times, mask=torch.ones(2))
                    torch.testing.assert_close(full, all_selected)
                    selected = diffusion(target, prediction, t=times, mask=torch.tensor([1., 0.]))
                    expected = diffusion(target[:1], prediction[:1], t=times[:1])
                    torch.testing.assert_close(selected, expected)
                    empty = diffusion(target, prediction, t=times, mask=torch.zeros(2))
                    self.assertEqual(empty.item(), 0.)
                    self.assertTrue(torch.isfinite(empty))

    def test_sampling_returns_trajectory_without_mutating_schedule(self):
        schedule = VarianceSchedule(100)
        diffusion = D2MP_OB('l1', PredictionNet(), schedule)
        condition = torch.tensor([[.2, -.4, .6]])
        trajectory = diffusion.sample(condition, bestof=False, ret_traj=True)
        self.assertEqual(schedule.num_steps, 100)
        self.assertEqual(set(trajectory), {0, 1})
        torch.testing.assert_close(trajectory[0], -condition)
        torch.testing.assert_close(diffusion.sample(condition, bestof=False), -condition)

    def test_unet_accepts_scalar_time_and_no_global_condition(self):
        model = ConditionalUnet1D(3, down_dims=(4, 8))
        sample = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        scalar = model(sample, .25, global_cond=None)
        vector = model(sample, torch.full((2,), .25), global_cond=None)
        self.assertEqual(scalar.shape, sample.shape)
        torch.testing.assert_close(scalar, vector)

    def test_neighborhood_offsets_follow_model_device(self):
        with patch('model.ddtracking.ConditionalUnet1D', return_value=PredictionNet()):
            model = DDTracking(1, 1, 4, 'GRU', 0., 3, 'l1').to('meta')
        self.assertEqual(model.offset.device.type, 'meta')
        self.assertEqual(tuple(model.offset.shape), (27, 3))
        self.assertNotIn('offset', model.state_dict())


if __name__ == '__main__':
    unittest.main()
