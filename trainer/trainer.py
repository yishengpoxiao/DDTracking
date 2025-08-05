import logging
import datetime
import math

from tqdm import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader, Sampler
import torch.nn.utils.rnn as rnn_utils
import torch.distributed as dist

from utils.dataset import load_dataset
from utils.utils import *
from model.ddtracking import DDTracking  # Ensure this import matches your model's actual class name

import nibabel as nib


class VolumeSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True):
        """
        Samples batches of volumes.
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        self.num_replicas = num_replicas

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.rank = rank

        sid2vid = np.array(self.dataset.streamline_id_to_volume_id)
        self.vol2stream = {
            vidx: np.where(sid2vid == vidx)[0]
            for vidx in range(len(self.dataset.volumes))
        }
        
        self.num_volumes = len(self.dataset.volumes)
        self.num_samples = int(math.ceil(self.num_volumes / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        
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

            if len(streamline_indices) >= self.batch_size:
                batch = streamline_indices[:self.batch_size]
            else:
                repeat_count = math.ceil(self.batch_size / len(streamline_indices))
                extended = np.tile(streamline_indices, repeat_count)
                batch = extended[:self.batch_size]
            
            all_batches.append(batch.tolist())

        padding_size = self.total_size - len(all_batches)
        if padding_size > 0:
            all_batches += all_batches[:padding_size]

        if self.shuffle:
            rng.shuffle(all_batches)

        sampled_batches = all_batches[self.rank:self.total_size:self.num_replicas]
        return iter(sampled_batches)

    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class VolumeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True):
        """
        Samples batches of volumes.
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        self.num_replicas = num_replicas

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.rank = rank
        
        self.stream_batch = 0

        sid2vid = np.array(self.dataset.streamline_id_to_volume_id)
        self.vol2stream = {
            vidx: np.where(sid2vid == vidx)[0]
            for vidx in range(len(self.dataset.volumes))
        }
        
        self.stream_batch = sum(math.ceil(len(v_inds)/batch_size) for v_inds in self.vol2stream.values())

        self.num_samples = int(math.ceil(self.stream_batch / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        
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
                if len(batch) < self.batch_size:
                    batch = np.concatenate([batch, streamline_indices[:self.batch_size-len(batch)]])
                all_batches.append(batch.tolist())
        
        padding_size = self.total_size - len(all_batches)
        if padding_size > 0:
            all_batches += all_batches[:padding_size]

        if self.shuffle:
            rng.shuffle(all_batches)

        sampled_batches = all_batches[self.rank:self.total_size:self.num_replicas]
        
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
        self.decay_LR_patience = int(self.params['decay_LR_patience'])
        self.decay_factor = float(self.params['decay_factor'])
        self.early_stopping_patience = int(self.params['early_stopping_patience'])
        self.loss_type = self.params['loss_type']
        
        self.T_max = int(self.params.get('T_max', 100))  # 默认周期为总 epoch 数
        self.eta_min = float(self.params.get('eta_min', 1e-6))

        self.model_weights_save_dir = self.params['model_weights_save_dir']
        self.model_name = self.params['model_name']
        if 'trainset_path' not in self.params and 'validset_path' not in self.params:
            self.dataset_path = self.params['dataset_path']
        else:
            self.train_path = self.params['trainset_path']
            self.valid_path = self.params['validset_path']

        self.sh_order = int(self.params.get('sh_order', 6))
        self.flip_streamline = bool(self.params['flip_streamline'])
        
        self.use_previous_dire = int(self.params['use_previous_dire'])

        self.model_type = self.params['model_type']
        self.input_type = self.params['input_type']
        
        self.look_ahead = bool(self.params.get('look_ahead', False))
        self.vox_step = float(self.params.get('vox_step', 0.5))

        self.resume = bool(self.params.get('resume', False))
        
        self.grad_directions = int((self.sh_order + 1) * (self.sh_order + 2) / 2) if self.input_type in ['sh', 'fodf'] else 100

        self.model = None
        
        if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = True
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.device = torch.device(f'cuda:{self.local_rank}')
        
            timeout = datetime.timedelta(hours=2)
            dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size, timeout=timeout)
            torch.cuda.set_device(self.rank)
            self.logger.info(f"Distributed training initialized. Rank: {self.rank}, Local Rank: {self.local_rank}")

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

        self.model.offset = self.model.offset.to(self.device)

        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            
    def destroy_ddp(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    def train(self):
        self.set_model()
        
        if 'trainset_path' in self.params and 'validset_path' in self.params:
            train_dataset = load_dataset(
                path=self.train_path,
                mode='trainset',
                input_type=self.input_type,
                sh_order=self.sh_order,
                flip_streamline=self.flip_streamline,
                vox_step=self.vox_step
            )
            valid_dataset = load_dataset(
                path=self.valid_path,
                mode='validset',
                input_type=self.input_type,
                sh_order=self.sh_order,
                flip_streamline=self.flip_streamline,
                vox_step=self.vox_step
            )
        else:
            dataset = load_dataset(
                path=self.dataset_path,
                mode='dataset',
                input_type=self.input_type,
                sh_order=self.sh_order,
                flip_streamline=self.flip_streamline,
                vox_step=self.vox_step
            )
            
            train_dataset, valid_dataset = dataset.split_into_train_valid(split_frac=0.8)
            del dataset
            
        if self.flip_streamline:
            train_dataset.fun_flip_streamline()
            valid_dataset.fun_flip_streamline()
        
        # train_sampler = VolumeSampler(train_dataset, self.batch_size, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        train_sampler = VolumeBatchSampler(train_dataset, self.batch_size, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        valid_sampler = VolumeBatchSampler(valid_dataset, self.batch_size*2, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        if self.resume:
            _, _, best_val_loss = self.load_checkpoint()
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-6)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=self.decay_factor,
                                                         patience=self.decay_LR_patience, threshold=1e-6, min_lr=self.eta_min)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-6)
            self.current_epoch = 0
            best_val_loss = float('inf')

            # Optimization setup
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=self.decay_factor,
                                                         patience=self.decay_LR_patience, threshold=1e-6, min_lr=self.eta_min)
            
        gradscaler = torch.GradScaler("cuda")

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=6, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=6, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)

        bad_epochs = 0

        for epoch in range(self.current_epoch, self.epochs):
            train_loader.batch_sampler.set_epoch(epoch)
            valid_loader.batch_sampler.set_epoch(epoch)

            self.model.train()
            train_loss = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", unit="batch", disable=self.local_rank != 0) as tepoch:
                for batch_idx, (volume, inputs, targets, length) in enumerate(tepoch):       
                    volume = volume.float().to(self.device)
                    inputs = inputs.float().to(self.device)
                    targets = targets.float().to(self.device)

                    optimizer.zero_grad()
                    
                    with torch.autocast(device_type='cuda'):
                        loss = self.model(volume, inputs, targets, length)
                    
                    gradscaler.scale(loss).backward()
                    gradscaler.step(optimizer)
                    gradscaler.update()

                    train_loss += loss.item()
                    avg_loss = train_loss / (batch_idx + 1)
                    tepoch.set_postfix(train_loss=avg_loss)
                        
            if self.distributed:
                dist.barrier()
                
            if self.local_rank == 0:
                old_lr = optimizer.param_groups[0]['lr']

            val_loss = self.validate(valid_loader)
            scheduler.step(val_loss)
            
            if self.local_rank == 0:
                new_lr = optimizer.param_groups[0]['lr']

                if old_lr == new_lr:
                    self.logger.info(f'Epoch {epoch+1}: Learning rate remains the same at {old_lr}')
                else:
                    self.logger.info(f'Epoch {epoch+1}: Learning rate changed from {old_lr} to {new_lr}')

                self.logger.info(f'Epoch {epoch+1}: Training loss {avg_loss:.7f}, Validation loss {val_loss:.7f}')

                self.save_checkpoint(epoch, optimizer, scheduler, val_loss, is_best=False)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, optimizer, scheduler, best_val_loss, is_best=True)
                    self.logger.info(f'Epoch {epoch+1}: New best model saved with validation loss {best_val_loss}')
                    bad_epochs = 0
                else:
                    self.logger.info(f'Epoch {epoch+1}: No improvement in validation loss.')
                    bad_epochs += 1

                if bad_epochs > self.early_stopping_patience:
                    self.logger.info('Early stopping triggered.')
                    break

        self.destroy_ddp()
        self.logger.info("Training completed with performance: {}".format(best_val_loss))

        return best_val_loss

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            with tqdm(loader, desc="Validating", unit="batch", disable=self.local_rank != 0) as vepoch:
                for volume, inputs, targets, length in vepoch:
                    volume = volume.float().to(self.device)
                    inputs = inputs.float().to(self.device)
                    targets = targets.float().to(self.device)

                    with torch.autocast(device_type='cuda'):
                        loss = self.model(volume, inputs, targets, length)

                    batch_size = inputs.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size

                    avg_loss = total_loss / total_samples

                    if self.local_rank == 0:
                        vepoch.set_postfix(val_loss=avg_loss)

            total_loss_tensor = torch.tensor(total_loss).to(self.device)
            total_samples_tensor = torch.tensor(total_samples).to(self.device)
            if self.distributed:
                dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
                total_loss = total_loss_tensor.item()
                total_samples = total_samples_tensor.item()

            avg_loss = total_loss / total_samples
            return avg_loss

    def save_checkpoint(self, epoch, optimizer, scheduler, val_loss, is_best):
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'grad_directions': self.grad_directions,
            'num_layers': self.num_layers,
            'num_cells': self.num_cells,
            'cell_type': self.cell_type,
            'dropout_prob': self.dropout_prob,
            'val_loss': val_loss,
        }

        if is_best:
            best_path = self.checkpoint_path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model checkpoint saved at epoch {epoch+1} with validation loss {val_loss}")
        else:
            torch.save(checkpoint, self.checkpoint_path)
            self.logger.info(f"Checkpoint saved at epoch {epoch+1} with validation loss {val_loss}")

    def load_checkpoint(self):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        best_path = self.checkpoint_path.replace('.pt', '_best.pt')
        checkpoint = torch.load(best_path, map_location=map_location)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-6)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=self.decay_factor,
                                                     patience=self.decay_LR_patience, threshold=1e-6, min_lr=self.eta_min)
        self.current_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        val_loss = checkpoint['val_loss']
        self.logger.info(f"Resumed training from epoch {self.current_epoch} with validation loss {val_loss}")
        return optimizer, scheduler, val_loss
    