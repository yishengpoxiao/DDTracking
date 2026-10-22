import logging
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from trainer.trainer import Trainer, VolumeBatchSampler, VolumeSampler


def dataset(volume_ids, volume_count=None):
    volume_count = volume_count or (max(volume_ids, default=-1) + 1)
    return SimpleNamespace(
        volumes=[np.zeros((1, 1, 1, 1)) for _ in range(volume_count)],
        streamline_id_to_volume_id=np.asarray(volume_ids, dtype=np.int64),
    )


class TinyTrainingDataset:
    def __init__(self):
        self.volumes = [np.zeros((4, 4, 4, 1), dtype=np.float32)]
        self.streamline_id_to_volume_id = np.asarray([0, 0], dtype=np.int64)
        self.streamlines = [
            np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
            np.asarray([[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32),
        ]

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, index):
        points = self.streamlines[index]
        targets = np.diff(points, axis=0)
        targets /= np.linalg.norm(targets, axis=-1, keepdims=True)
        return self.volumes[0], points[:-1], targets


class SamplerTests(unittest.TestCase):
    def test_empty_volumes_are_skipped_and_uneven_validation_shards_are_exact(self):
        data = dataset([0, 0, 0, 2, 2], volume_count=3)
        shards = [
            list(VolumeBatchSampler(data, 2, num_replicas=2, rank=rank,
                                    shuffle=False, pad_distributed=False, pad_last=False))
            for rank in range(2)
        ]

        self.assertEqual([len(shard) for shard in shards], [2, 1])
        flattened = [item for shard in shards for batch in shard for item in batch]
        self.assertCountEqual(flattened, range(5))

    def test_training_sampler_pads_full_batches_even_when_world_exceeds_batches(self):
        data = dataset([0])
        shards = [list(VolumeBatchSampler(data, 4, num_replicas=3, rank=rank,
                                          shuffle=False)) for rank in range(3)]

        self.assertEqual([len(shard) for shard in shards], [1, 1, 1])
        self.assertTrue(all(len(shard[0]) == 4 for shard in shards))

    def test_empty_dataset_fails_without_division_by_zero(self):
        data = dataset([], volume_count=2)
        with self.assertRaisesRegex(ValueError, "no streamlines"):
            VolumeSampler(data, 2, num_replicas=1, rank=0)
        with self.assertRaisesRegex(ValueError, "no streamlines"):
            VolumeBatchSampler(data, 2, num_replicas=1, rank=0)


class TrainerCheckpointTests(unittest.TestCase):
    def test_config_does_not_require_unused_model_type_and_sizes_rish_channels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            params = {
                "num_layers": 1, "num_cells": 2, "cell_type": "GRU",
                "dropout_prob": 0.0, "learning_rate": 1e-3, "batch_size": 2,
                "epochs": 1, "decay_LR_patience": 2, "decay_factor": 0.5,
                "early_stopping_patience": 3, "loss_type": "l1",
                "model_weights_save_dir": temp_dir, "model_name": "test",
                "dataset_path": temp_dir, "sh_order": 6, "flip_streamline": False,
                "use_previous_dire": 0, "input_type": "rish", "num_workers": 0,
            }

            trainer = Trainer(params=params)

            self.assertEqual(trainer.grad_directions, 4)

    @staticmethod
    def trainer(path, model):
        trainer = Trainer.__new__(Trainer)
        trainer.checkpoint_path = path
        trainer.model = model
        trainer.distributed = False
        trainer.rank = 0
        trainer.device = torch.device("cpu")
        trainer.grad_directions = 1
        trainer.num_layers = 1
        trainer.num_cells = 2
        trainer.cell_type = "GRU"
        trainer.dropout_prob = 0.0
        trainer.logger = logging.getLogger("trainer-checkpoint-test")
        return trainer

    def test_resume_restores_latest_optimizer_scheduler_and_stopping_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = f"{temp_dir}/model.pt"
            first_model = torch.nn.Linear(2, 1)
            first = self.trainer(path, first_model)
            optimizer = torch.optim.AdamW(first_model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            with torch.no_grad():
                first_model.weight.fill_(3.0)
            first.save_checkpoint(4, optimizer, scheduler, 0.7, is_best=False,
                                  gradscaler=scaler, best_val_loss=0.5, bad_epochs=2)

            resumed_model = torch.nn.Linear(2, 1)
            resumed = self.trainer(path, resumed_model)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.5)
            resumed_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(resumed_optimizer)
            resumed_scaler = torch.amp.GradScaler("cuda", enabled=False)

            best_loss, bad_epochs = resumed.load_checkpoint(
                resumed_optimizer, resumed_scheduler, resumed_scaler)

            torch.testing.assert_close(resumed_model.weight, first_model.weight)
            self.assertEqual(resumed.current_epoch, 5)
            self.assertEqual(best_loss, 0.5)
            self.assertEqual(bad_epochs, 2)
            self.assertEqual(resumed_optimizer.param_groups[0]["lr"], 0.01)

    def test_best_checkpoint_path_is_based_on_file_stem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = f"{temp_dir}/directory.pt"
            path = f"{folder}/model.pt"
            import os
            os.makedirs(folder)
            model = torch.nn.Linear(2, 1)
            trainer = self.trainer(path, model)
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            scaler = torch.amp.GradScaler("cuda", enabled=False)

            trainer.save_checkpoint(0, optimizer, scheduler, 1.0, is_best=True,
                                    gradscaler=scaler, best_val_loss=1.0, bad_epochs=0)

            self.assertTrue(os.path.isfile(f"{folder}/model_best.pt"))


class ValidationTests(unittest.TestCase):
    class LossModel:
        def __init__(self, losses):
            self.losses = iter(losses)

        def eval(self):
            return self

        def __call__(self, volume, inputs, targets, length):
            return torch.tensor(next(self.losses))

    @staticmethod
    def batch(length):
        return (torch.zeros((1, 1)), torch.zeros((1, 1, 3)),
                torch.zeros((1, length, 3)), [length])

    def test_validation_weights_batch_losses_by_target_steps(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = self.LossModel([2.0, 4.0])
        trainer.distributed = False
        trainer.rank = 0
        trainer.device = torch.device("cpu")

        value = trainer.validate([self.batch(1), self.batch(3)])

        self.assertEqual(value, 3.5)

    def test_empty_validation_rank_reduces_float_loss_and_integer_count(self):
        trainer = Trainer.__new__(Trainer)
        trainer.model = SimpleNamespace(module=self.LossModel([]), eval=lambda: None)
        trainer.distributed = True
        trainer.rank = 1
        trainer.device = torch.device("cpu")
        tensors = []

        def all_reduce(tensor, op):
            tensors.append(tensor)
            tensor.fill_(8.0 if tensor.dtype == torch.float64 else 4)

        with patch("trainer.trainer.dist.all_reduce", side_effect=all_reduce):
            value = trainer.validate([])

        self.assertEqual(value, 2.0)
        self.assertEqual([tensor.dtype for tensor in tensors], [torch.float64, torch.int64])

    def test_early_stop_uses_global_broadcast(self):
        trainer = Trainer.__new__(Trainer)
        trainer.distributed = True
        trainer.rank = 1
        trainer.device = torch.device("cpu")

        def broadcast(tensor, src):
            tensor.fill_(1)

        with patch("trainer.trainer.dist.broadcast", side_effect=broadcast) as call:
            self.assertTrue(trainer._sync_early_stop(False))

        call.assert_called_once()


class TinyTrainResumeTests(unittest.TestCase):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, volume, inputs, targets, length):
            return self.scale.square()

    def test_cpu_training_and_resume_with_zero_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            params = {
                "num_layers": 1, "num_cells": 2, "cell_type": "GRU",
                "dropout_prob": 0.0, "learning_rate": 0.01, "batch_size": 2,
                "epochs": 1, "decay_LR_patience": 1, "decay_factor": 0.5,
                "early_stopping_patience": 5, "loss_type": "l1",
                "model_weights_save_dir": temp_dir, "model_name": "tiny",
                "trainset_path": "/train", "validset_path": "/valid",
                "sh_order": 6, "flip_streamline": False, "use_previous_dire": 0,
                "input_type": "sh", "num_workers": 0,
            }

            def set_tiny_model(trainer):
                trainer.model = self.Model().to(trainer.device)

            with patch("torch.cuda.is_available", return_value=False), \
                 patch("trainer.trainer.load_dataset", side_effect=lambda **kwargs: TinyTrainingDataset()), \
                 patch.object(Trainer, "set_model", set_tiny_model):
                Trainer(params=params).train()
                params["resume"] = True
                params["epochs"] = 2
                Trainer(params=params).train()

            checkpoint = torch.load(f"{temp_dir}/tiny.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)


class DistributedDatasetLoadTests(unittest.TestCase):
    def test_nonzero_rank_waits_for_rank_zero_cache_creation(self):
        trainer = Trainer.__new__(Trainer)
        trainer.distributed = True
        trainer.rank = 1
        trainer.device = torch.device("cpu")
        trainer.input_type = "sh"
        trainer.sh_order = 6
        trainer.flip_streamline = False
        trainer.vox_step = 0.5
        trainer.train_path = "/train"
        trainer.valid_path = "/valid"
        events = []

        def broadcast(status, src, device):
            events.append("broadcast")
            status[0] = None

        def load(**kwargs):
            events.append("load")
            return object()

        with patch("trainer.trainer.dist.broadcast_object_list", side_effect=broadcast), \
             patch("trainer.trainer.dist.barrier", side_effect=lambda: events.append("barrier")), \
             patch("trainer.trainer.load_dataset", side_effect=load):
            train, valid = trainer._load_datasets()

        self.assertEqual(events, ["broadcast", "barrier", "load", "load"])
        self.assertIsNot(train, valid)


if __name__ == "__main__":
    unittest.main()
