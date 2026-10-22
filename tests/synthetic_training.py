"""Train and resume the actual network on tiny synthetic HDF5 datasets."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import torch

from trainer.trainer import Trainer
from utils.dataset import TractographyDataset


def main():
    root = Path(tempfile.mkdtemp(prefix='ddtracking-training-'))
    for split, position in (('train', 1.), ('valid', 2.)):
        lines = [np.array([[position, 1, 1], [position, 1.5, 1], [position, 2, 1]], dtype=np.float32),
                 np.array([[1, position, 1], [1.5, position, 1], [2, position, 1]], dtype=np.float32)]
        subject = SimpleNamespace(volume=np.ones((4, 4, 4, 1), dtype=np.float32),
                                  tractogram=nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)))
        TractographyDataset(subjects=[subject]).save_to_h5(root / f'{split}.h5')
    params = dict(num_layers=1, num_cells=8, cell_type='GRU', dropout_prob=0.,
                  learning_rate=0.001, batch_size=2, epochs=1, decay_LR_patience=1,
                  decay_factor=.5, early_stopping_patience=5, loss_type='l1',
                  model_weights_save_dir=str(root), model_name='synthetic',
                  trainset_path=str(root / 'train.h5'), validset_path=str(root / 'valid.h5'),
                  sh_order=0, input_type='sh', flip_streamline=True, use_previous_dire=2,
                  look_ahead=True, vox_step=.5, num_workers=0)
    torch.manual_seed(42)
    first = Trainer(params=params).train()
    params.update(resume=True, epochs=2)
    resumed = Trainer(params=params).train()
    checkpoint = torch.load(root / 'synthetic.pt', map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 2
    assert np.isfinite(first) and np.isfinite(resumed)
    assert checkpoint['optimizer']['state']
    assert all(state['step'].item() == 4 for state in checkpoint['optimizer']['state'].values())
    print(f'Actual-network training and resume passed; checkpoints in {root}')


if __name__ == '__main__':
    main()
