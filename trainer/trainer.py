import logging
import datetime
import math
import os
import random

import numpy as np
from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader, Sampler
import torch.nn.utils.rnn as rnn_utils
import torch.distributed as dist

from utils.dataset import load_dataset
from model.ddtracking import DDTracking


def _distributed_batches(batches, num_replicas, rank, pad=True):
    if not batches:
        raise ValueError("Cannot sample a dataset with no streamlines.")
    if pad:
        total_size = math.ceil(len(batches) / num_replicas) * num_replicas
        batches = batches + [batches[index % len(batches)] for index in range(total_size - len(batches))]
    return batches[rank::num_replicas]


class VolumeSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True):
        """
        Samples batches of volumes.
        """
        
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.shuffle = shuffle

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        self.num_replicas = int(num_replicas)

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.rank = int(rank)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be between zero and num_replicas - 1")

        sid2vid = np.array(self.dataset.streamline_id_to_volume_id)
        self.vol2stream = {
            vidx: np.where(sid2vid == vidx)[0]
            for vidx in range(len(self.dataset.volumes))
        }
        
        self.volume_indices = [index for index in range(len(dataset.volumes)) if len(self.vol2stream[index])]
        self.num_volumes = len(self.volume_indices)
        if not self.num_volumes:
            raise ValueError("Cannot sample a dataset with no streamlines.")
        self.num_samples = int(math.ceil(self.num_volumes / self.num_replicas))
        
        self.epoch = 0
    def __iter__(self):
        rng = np.random.default_rng(self.epoch)
        
        all_batches = []
        
        volume_indices = np.asarray(self.volume_indices)
        if self.shuffle:
            rng.shuffle(volume_indices)
            
        for v_idx in volume_indices:
            streamline_indices = self.vol2stream[v_idx].copy()
            if self.shuffle:
                rng.shuffle(streamline_indices)

            batch = np.resize(streamline_indices, self.batch_size)
            
            all_batches.append(batch.tolist())

        if self.shuffle:
            rng.shuffle(all_batches)

        sampled_batches = _distributed_batches(all_batches, self.num_replicas, self.rank)
        return iter(sampled_batches)

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class VolumeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True,
                 pad_distributed=True, pad_last=True):
        """
        Samples batches of volumes.
        """
        
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.shuffle = shuffle
        self.pad_distributed = pad_distributed
        self.pad_last = pad_last

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        self.num_replicas = int(num_replicas)

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.rank = int(rank)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank must be between zero and num_replicas - 1")
        
        self.stream_batch = 0

        sid2vid = np.array(self.dataset.streamline_id_to_volume_id)
        self.vol2stream = {
            vidx: np.where(sid2vid == vidx)[0]
            for vidx in range(len(self.dataset.volumes))
        }
        
        self.stream_batch = sum(math.ceil(len(v_inds) / self.batch_size)
                                for v_inds in self.vol2stream.values() if len(v_inds))
        if self.stream_batch == 0:
            raise ValueError("Cannot sample a dataset with no streamlines.")

        self.num_samples = (int(math.ceil(self.stream_batch / self.num_replicas)) if pad_distributed
                            else max(0, (self.stream_batch - self.rank + self.num_replicas - 1) // self.num_replicas))
        
        self.epoch = 0
    
    def __iter__(self):
        rng = np.random.default_rng(self.epoch)
        
        all_batches = []
        
        volume_indices = np.arange(len(self.dataset.volumes))
        if self.shuffle:
            rng.shuffle(volume_indices)
            
        for v_idx in volume_indices:
            streamline_indices = self.vol2stream[v_idx].copy()
            if self.shuffle:
                rng.shuffle(streamline_indices)

            for i in range(0, len(streamline_indices), self.batch_size):
                batch = streamline_indices[i:i+self.batch_size]
                if self.pad_last and len(batch) < self.batch_size:
                    batch = np.concatenate([batch, np.resize(streamline_indices, self.batch_size - len(batch))])
                all_batches.append(batch.tolist())

        if self.shuffle:
            rng.shuffle(all_batches)

        sampled_batches = _distributed_batches(all_batches, self.num_replicas, self.rank,
                                               pad=self.pad_distributed)
        
        return iter(sampled_batches)
            
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    
def collate_fn(batch):
    volume = torch.from_numpy(batch[0][0][np.newaxis, ...])
    
    batch.sort(key=lambda x: len(x[1]), reverse=True)

    # batch_x = [torch.from_numpy(item[1]) for item in batch]
    # batch_y = [torch.from_numpy(item[2]) for item in batch]
    # batch_length = [y.size(0) for y in batch_y]
    batch_x = []
    batch_y = []
    batch_length = []
    for _, sl, targ in batch:
        batch_x.append(torch.from_numpy(sl))
        batch_y.append(torch.from_numpy(targ))
        batch_length.append(len(targ))
        
    batch_x = rnn_utils.pad_sequence(batch_x, batch_first=True, padding_value=0.0)
    batch_y = rnn_utils.pad_sequence(batch_y, batch_first=True, padding_value=0.0)
    
    return volume, batch_x, batch_y, batch_length


class Trainer:
    def __init__(self, logger=None, **args):
        self.params = args['params']
        if logger is None:
            logging.basicConfig(format='%(asctime)s %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger

        self.num_layers = int(self.params['num_layers'])
        self.num_cells = int(self.params['num_cells'])
        self.cell_type = self.params['cell_type']
        self.dropout_prob = float(self.params['dropout_prob'])
        self.learning_rate = float(self.params['learning_rate'])
        self.batch_size = int(self.params['batch_size'])
        self.epochs = int(self.params['epochs'])
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        self.num_workers = int(self.params.get('num_workers', 6))
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        self.decay_LR_patience = int(self.params['decay_LR_patience'])
        self.decay_factor = float(self.params['decay_factor'])
        self.early_stopping_patience = int(self.params['early_stopping_patience'])
        self.loss_type = self.params['loss_type']
        
        self.T_max = int(self.params.get('T_max', 100))  # 默认周期为总 epoch 数
        self.eta_min = float(self.params.get('eta_min', 1e-6))

        self.model_weights_save_dir = self.params['model_weights_save_dir']
        self.model_name = self.params['model_name']
        has_trainset = 'trainset_path' in self.params
        has_validset = 'validset_path' in self.params
        if has_trainset != has_validset:
            raise ValueError("trainset_path and validset_path must be provided together.")
        if not has_trainset:
            self.dataset_path = self.params['dataset_path']
        else:
            self.train_path = self.params['trainset_path']
            self.valid_path = self.params['validset_path']

        self.sh_order = int(self.params.get('sh_order', 6))
        self.flip_streamline = bool(self.params['flip_streamline'])
        
        self.use_previous_dire = int(self.params['use_previous_dire'])

        self.input_type = self.params['input_type']
        
        self.look_ahead = bool(self.params.get('look_ahead', False))
        self.vox_step = float(self.params.get('vox_step', 0.5))

        self.resume = bool(self.params.get('resume', False))
        
        if self.input_type in ['sh', 'fodf']:
            self.grad_directions = (self.sh_order + 1) * (self.sh_order + 2) // 2
        elif self.input_type == 'rish':
            self.grad_directions = self.sh_order // 2 + 1
        else:
            self.grad_directions = 100

        self.model = None
        
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = True
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f'cuda:{self.local_rank}')
        
            timeout = datetime.timedelta(hours=2)
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, timeout=timeout)
            if self.rank == 0:
                self.logger.info(f"Distributed training initialized. World size: {self.world_size}")

            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        else:
            self.world_size = 1
            self.distributed = False
            self.rank = 0
            self.local_rank = 0
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.logger.info("Single GPU or CPU training")

        self.current_epoch = None
        os.makedirs(self.model_weights_save_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_weights_save_dir, self.model_name + '.pt')

    def set_model(self):
        self.model = DDTracking(
            input_size=self.grad_directions,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            dropout_rate=self.dropout_prob,
            target_dim=3,
            loss_type=self.loss_type,
            previous_dire=self.use_previous_dire,
            look_ahead=self.look_ahead,
            vox_step=self.vox_step
        ).to(self.device)

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            
    def destroy_ddp(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def _load_datasets(self):
        def load():
            options = dict(input_type=self.input_type, sh_order=self.sh_order,
                           flip_streamline=self.flip_streamline, vox_step=self.vox_step)
            if hasattr(self, 'train_path'):
                train = load_dataset(path=self.train_path, mode='trainset', **options)
                valid = load_dataset(path=self.valid_path, mode='validset', **options)
                return train, valid

            dataset = load_dataset(path=self.dataset_path, mode='dataset', **options)
            return dataset.split_into_train_valid(split_frac=0.8)

        if not self.distributed:
            return load()

        datasets = None
        error = None
        if self.rank == 0:
            try:
                datasets = load()
            except Exception as exc:
                error = exc

        status = [None if error is None else (type(error).__name__, str(error))]
        dist.broadcast_object_list(status, src=0, device=self.device)
        if status[0] is not None:
            if error is not None:
                raise error
            name, message = status[0]
            raise RuntimeError(f"Rank 0 failed to load training datasets ({name}): {message}")

        dist.barrier()
        return datasets if self.rank == 0 else load()

    def train(self):
        self.set_model()
        
        train_dataset, valid_dataset = self._load_datasets()
            
        if self.flip_streamline:
            train_dataset.fun_flip_streamline()
            valid_dataset.fun_flip_streamline()
        
        # train_sampler = VolumeSampler(train_dataset, self.batch_size, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        train_sampler = VolumeBatchSampler(train_dataset, self.batch_size, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        valid_sampler = VolumeBatchSampler(valid_dataset, self.batch_size * 2,
                                           num_replicas=self.world_size, rank=self.rank,
                                           shuffle=False, pad_distributed=False, pad_last=False)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=self.decay_factor,
                                                     patience=self.decay_LR_patience, threshold=1e-6, min_lr=self.eta_min)
        gradscaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        if self.resume:
            best_val_loss, bad_epochs = self.load_checkpoint(optimizer, scheduler, gradscaler)
        else:
            self.current_epoch = 0
            best_val_loss = float('inf')
            bad_epochs = 0

        loader_options = dict(num_workers=self.num_workers, pin_memory=self.device.type == 'cuda',
                             collate_fn=collate_fn, persistent_workers=self.num_workers > 0)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_options)
        valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, **loader_options)

        for epoch in range(self.current_epoch, self.epochs):
            train_loader.batch_sampler.set_epoch(epoch)
            valid_loader.batch_sampler.set_epoch(epoch)

            self.model.train()
            train_loss = 0
            train_steps = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", unit="batch", disable=self.rank != 0) as tepoch:
                for batch_idx, (volume, inputs, targets, length) in enumerate(tepoch):       
                    volume = volume.float().to(self.device)
                    inputs = inputs.float().to(self.device)
                    targets = targets.float().to(self.device)

                    optimizer.zero_grad()
                    
                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                        loss = self.model(volume, inputs, targets, length)
                    
                    gradscaler.scale(loss).backward()
                    gradscaler.step(optimizer)
                    gradscaler.update()

                    step_count = sum(length)
                    train_loss += loss.item() * step_count
                    train_steps += step_count
                    avg_loss = train_loss / train_steps
                    tepoch.set_postfix(train_loss=avg_loss)
                        
            if self.distributed:
                dist.barrier()
                
            if self.rank == 0:
                old_lr = optimizer.param_groups[0]['lr']

            val_loss = self.validate(valid_loader)
            scheduler.step(val_loss)
            
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1

            if self.rank == 0:
                new_lr = optimizer.param_groups[0]['lr']

                if old_lr == new_lr:
                    self.logger.info(f'Epoch {epoch+1}: Learning rate remains the same at {old_lr}')
                else:
                    self.logger.info(f'Epoch {epoch+1}: Learning rate changed from {old_lr} to {new_lr}')

                self.logger.info(f'Epoch {epoch+1}: Training loss {avg_loss:.7f}, Validation loss {val_loss:.7f}')

                if improved:
                    self.logger.info(f'Epoch {epoch+1}: New best model saved with validation loss {best_val_loss}')
                else:
                    self.logger.info(f'Epoch {epoch+1}: No improvement in validation loss.')

            self.save_checkpoint(epoch, optimizer, scheduler, val_loss, is_best=improved,
                                 gradscaler=gradscaler, best_val_loss=best_val_loss, bad_epochs=bad_epochs)

            should_stop = self._sync_early_stop(self.rank == 0 and bad_epochs > self.early_stopping_patience)
            if should_stop:
                if self.rank == 0:
                    self.logger.info('Early stopping triggered.')
                break

        self.destroy_ddp()
        if self.rank == 0:
            self.logger.info("Training completed with performance: {}".format(best_val_loss))

        return best_val_loss

    def validate(self, loader):
        self.model.eval()
        validation_model = self.model.module if self.distributed else self.model
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            with tqdm(loader, desc="Validating", unit="batch", disable=self.rank != 0) as vepoch:
                for volume, inputs, targets, length in vepoch:
                    volume = volume.float().to(self.device)
                    inputs = inputs.float().to(self.device)
                    targets = targets.float().to(self.device)

                    with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                        loss = validation_model(volume, inputs, targets, length)

                    target_steps = sum(length)
                    total_loss += loss.item() * target_steps
                    total_samples += target_steps

                    avg_loss = total_loss / total_samples if total_samples else float('nan')

                    if self.rank == 0:
                        vepoch.set_postfix(val_loss=avg_loss)

            total_loss_tensor = torch.tensor(total_loss, dtype=torch.float64, device=self.device)
            total_samples_tensor = torch.tensor(total_samples, dtype=torch.int64, device=self.device)
            if self.distributed:
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
                total_loss = total_loss_tensor.item()
                total_samples = total_samples_tensor.item()

            if total_samples == 0:
                raise ValueError("Validation dataset produced no target steps.")
            avg_loss = total_loss / total_samples
            return avg_loss

    def _sync_early_stop(self, stop):
        if not self.distributed:
            return bool(stop)
        stop_tensor = torch.tensor(int(stop) if self.rank == 0 else 0, device=self.device)
        dist.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())

    def save_checkpoint(self, epoch, optimizer, scheduler, val_loss, is_best,
                        gradscaler, best_val_loss, bad_epochs):
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'gradscaler': gradscaler.state_dict(),
            'grad_directions': self.grad_directions,
            'num_layers': self.num_layers,
            'num_cells': self.num_cells,
            'cell_type': self.cell_type,
            'dropout_prob': self.dropout_prob,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'bad_epochs': bad_epochs,
        }

        torch.save(checkpoint, self.checkpoint_path)
        self.logger.info(f"Checkpoint saved at epoch {epoch+1} with validation loss {val_loss}")
        if is_best:
            path, extension = os.path.splitext(self.checkpoint_path)
            best_path = f"{path}_best{extension}"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model checkpoint saved at epoch {epoch+1} with validation loss {val_loss}")

    def load_checkpoint(self, optimizer, scheduler, gradscaler):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        gradscaler.load_state_dict(checkpoint['gradscaler'])
        self.current_epoch = int(checkpoint['epoch'])
        best_val_loss = float(checkpoint['best_val_loss'])
        bad_epochs = int(checkpoint['bad_epochs'])
        self.logger.info(f"Resumed training from epoch {self.current_epoch} with best validation loss {best_val_loss}")
        return best_val_loss, bad_epochs
