"""Two-process Gloo check for uneven validation shards and stop synchronization."""

import datetime
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer.trainer import Trainer  # noqa: E402


class ConstantLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_value = torch.nn.Parameter(torch.tensor(3.5))

    def forward(self, volume, inputs, targets, length):
        return self.loss_value


def _worker(rank, rendezvous_file):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        trainer = Trainer.__new__(Trainer)
        trainer.model = DistributedDataParallel(ConstantLoss())
        trainer.distributed = True
        trainer.rank = rank
        trainer.device = torch.device("cpu")

        batch = (
            torch.zeros((1, 1)),
            torch.zeros((1, 4, 3)),
            torch.zeros((1, 4, 3)),
            [4],
        )
        validation_loader = [batch] if rank == 0 else []
        validation_loss = trainer.validate(validation_loader)
        if abs(validation_loss - 3.5) > 1e-6:
            raise AssertionError(f"rank {rank} got validation loss {validation_loss}, expected 3.5")

        stop = trainer._sync_early_stop(rank == 0)
        if not stop:
            raise AssertionError(f"rank {rank} did not receive rank 0's early-stop signal")
    finally:
        dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous_file = os.path.join(temp_dir, "gloo-rendezvous")
        mp.spawn(_worker, args=(rendezvous_file,), nprocs=2, join=True)
    print("Two-process Gloo validation and early-stop checks passed.")


if __name__ == "__main__":
    main()
