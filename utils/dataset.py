from concurrent.futures import ProcessPoolExecutor

import h5py
import copy
import os
import nibabel as nib
import nibabel.streamlines
import torch.utils.data as data
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.tracking.streamline import set_number_of_points, length

from utils.utils import *


def load_subject(input_path, input_type='sh', sh_order=6, vox_step=0.5):
    print("load subject's dwi from {}".format(input_path))
    dwi_path = locate_files(input_path, 'dwi')
    print("load subject's brain mask from {}".format(input_path))
    brain_mask_path = locate_files(input_path, 'brain-mask')
    print("load subject's wm mask from {}".format(input_path))
    try:
        wm_mask_path = locate_files(input_path, 'wm-mask')
    except Exception as e:
        print(f"Error loading WM mask: {e}")
        wm_mask_path = None
    print("load subject's tract from {}".format(input_path))
    tract_path = locate_files(input_path, 'tract')

    subject = TractographyData('train', dwi_path, brain_mask_path, wm_mask_path, tract_path, sh_order=sh_order, input_type=input_type, vox_step=vox_step)
    return subject


def load_dataset(path, mode, sh_order=6, input_type='sh', flip_streamline=False, vox_step=0.5):
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if os.path.isfile(os.path.join(path, fname)) and fname.endswith('.h5'):
                return TractographyDataset(h5_file=os.path.join(path, fname), flip_streamline=flip_streamline)

        max_workers = min(32, os.cpu_count() + 4)
        # max_workers = 10
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(load_subject, os.path.join(path, sub_dir), input_type, sh_order, vox_step) for sub_dir in os.listdir(path)]

            subjects = []
            for future in futures:
                subject = future.result()
                if subject is not None:
                    subjects.append(subject)

        dataset = TractographyDataset(subjects=subjects, flip_streamline=flip_streamline)
        dataset.save_to_h5(os.path.join(path, f'{mode}_dataset.h5'))
    else:
        dataset = TractographyDataset(h5_file=path, flip_streamline=flip_streamline)

    return dataset



class TractographyData(object):
    """
        Represents tractography data for a single subject.

        Attributes:
            mode (str): Mode of operation, either 'train' or 'track'.
            dwi_path (str): Path to the DWI file.
            brain_mask_path (str): Path to the brain mask file.
            wm_mask_path (str): Path to the white matter mask file.
            tractogram_path (str): Path to the tractogram file (required for training).
            use_sh (bool): Whether to use spherical harmonics.

        Methods:
            load_dwi(): Loads the DWI data and gradient table.
            load_mask(mask_path): Loads a binary mask from the given path.
            load_tractogram(step_size): Loads and optionally resamples the tractogram.
            preprocess_dwi(): Preprocesses the DWI data (masking, resampling).
    """

    def __init__(self, mode, DWI_path=None, brain_mask_path=None, wm_mask_path=None,
                 tractogram_path=None, sh_order=6, input_type='sh', vox_step=0.5):
        if mode not in ['train', 'track']:
            raise ValueError('Mode must be either "train" or "track".')

        self.mode = mode
        self.dwi_path = DWI_path
        self.brain_mask_path = brain_mask_path
        self.wm_mask_path = wm_mask_path
        self.tractogram_path = tractogram_path if self.mode == 'train' else None
        self.sh_order = sh_order
        self.input_type = input_type
        
        self.vox_step = vox_step
        
        if self.input_type not in ['resample', 'sh', 'fodf', 'rish']:
            raise ValueError('Input type must be either "resample", "sh", "fodf", or "rish".')

        if self.mode == 'train' and self.tractogram_path is None:
            raise ValueError("Tractogram path must be provided for training mode.")

        # Initialize attributes to None
        self.dwi = self.gradients = self.brain_mask = self.wm_mask = self.tractogram = self.ID = None
        self.volume = None

        self.ID = os.path.dirname(self.dwi_path).split('/')[-2]

        # Load data
        if self.dwi_path is not None:
            self.load_dwi()
        if self.brain_mask_path is not None:
            self.brain_mask = self.load_mask(self.brain_mask_path).astype(np.bool_)
        if self.wm_mask_path is not None:
            self.wm_mask = self.load_mask(self.wm_mask_path).astype(np.bool_)
        if self.tractogram_path is not None:
            step_size = self.dwi.header.get_zooms()[0] * self.vox_step if self.mode == 'train' else None
            self.load_tractogram(step_size)

        # Preprocess DWI volumes
        self.preprocess_dwi()

        # test mean normalize 03-19
        # self.volume = minmax_normalize(self.volume, self.brain_mask).astype(np.float32)
        # self.volume = zero_mean_unit_variance_normalize(self.volume, mask=self.brain_mask, per_channel=False).astype(np.float32)
        self.volume = mask_dwi(self.volume, self.brain_mask).astype(np.float32)

        if self.mode == 'train':
            self.masked_dwi = self.dwi = self.gradients = self.brain_mask = self.wm_mask = None
            
        if self.mode == 'track' and self.wm_mask_path is None:
            # calculate FA
            tensor_model = dti.TensorModel(self.gradients)
            tensor_fit = tensor_model.fit(self.masked_dwi)
            
            FA = dti.fractional_anisotropy(tensor_fit.evals)
            FA[np.isnan(FA)] = 0

            GA = dti.geodesic_anisotropy(tensor_fit.evals)
            GA[np.isnan(GA)] = 0
            
            self.FA = FA
            self.GA = GA

    def load_dwi(self):
        dwi_dir = os.path.dirname(self.dwi_path)

        dwi_file = glob.glob(os.path.join(dwi_dir, '*dwi*.nii*'))[0]
        self.dwi = nib.load(dwi_file)

        bval_file = glob.glob(os.path.join(dwi_dir, '*bval*'))[0]
        bvec_file = glob.glob(os.path.join(dwi_dir, '*bvec*'))[0]
        self.gradients = gradient_table(bval_file, bvec_file)

    @staticmethod
    def load_mask(mask_path):
        dwi_data = nib.load(mask_path)
        return (dwi_data.get_fdata() > 0.1) * 1.

    def load_tractogram(self, step_size=None):
        tract_file = nib.streamlines.load(self.tractogram_path)
        tractogram = tract_file.tractogram

        # Resample streamline to have a fixed step size, if needed.
        if step_size is not None:
            print("Resampling streamlines to have a step size of {}mm".format(step_size))
            streamlines = tractogram.streamlines
            streamlines._lengths = streamlines._lengths.astype(int)
            streamlines._offsets = streamlines._offsets.astype(int)
            lengths = length(streamlines)
            nb_points = np.ceil(lengths / step_size).astype(int)
            new_streamlines = (set_number_of_points(s, n) for s, n in zip(streamlines, nb_points))
            tractogram = nib.streamlines.Tractogram(new_streamlines, affine_to_rasmm=np.eye(4))

        # Compute matrix that brings streamlines back to diffusion voxel space.
        rasmm2vox_affine = np.linalg.inv(self.dwi.affine)
        tractogram.apply_affine(rasmm2vox_affine)

        self.tractogram = tractogram

    def preprocess_dwi(self):
        self.masked_dwi = mask_dwi(self.dwi.get_fdata().astype(np.float32), self.brain_mask)
        if self.input_type == 'fodf':
            print("compute fodf using CSD")
            csd_fit = compute_fodf_ssst(self.masked_dwi, self.gradients, self.brain_mask)
            volume = csd_fit.shm_coeff.astype(np.float32)
            self.volume = volume
        elif self.input_type == 'resample':
            self.volume = resample_dwi(self.masked_dwi, self.gradients, self.brain_mask, sh_order=self.sh_order).astype(np.float32)
        elif self.input_type == 'sh':
            self.volume = get_spherical_harmonics_coefficients(self.masked_dwi, self.gradients, self.brain_mask, sh_order=self.sh_order).astype(np.float32)
        elif self.input_type == 'rish':
            self.volume = compute_rish(self.masked_dwi, self.gradients, self.brain_mask, sh_order=self.sh_order).astype(np.float32)


def calculate_directions(streamline):
    directions = np.diff(streamline, axis=0)
    norms = np.linalg.norm(directions, axis=-1)[:, None]
    return directions / norms


class TractographyDataset(data.Dataset):
    def __init__(self, subjects=None, h5_file=None, flip_streamline=False, gaussian_noise=0.0):
        self.grad_directions = None
        self.volumes = []
        self.streamline_id_to_volume_id = []
        self.subject_streamlines = {}

        self.flip_streamline = flip_streamline

        if h5_file:
            self.load_from_h5(h5_file)
        else:
            if not subjects:
                raise ValueError("Subjects must be provided if not loading from an h5 file.")

            self.grad_directions = subjects[0].volume.shape[-1]
            self.streamlines = nibabel.streamlines.ArraySequence()
            self.streamline_id_to_volume_id = []

            for idx, subject in enumerate(subjects):
                sub_streamlines = subject.tractogram.streamlines
                
                self.streamlines.extend(sub_streamlines)
                self.streamline_id_to_volume_id.extend([idx] * len(sub_streamlines))
                
                self.volumes.append(subject.volume)

            subjects = None
            
    def load_from_h5(self, file_path):
        with h5py.File(file_path, 'r') as h5f:
            self.grad_directions = h5f['grad_directions'][:][0]
            self.volumes = [h5f['volumes'][key][()] for key in h5f['volumes'].keys()]
            self.streamline_id_to_volume_id = h5f['streamline_id_to_volume_id'][:]

        self.streamlines = nibabel.streamlines.ArraySequence().load(file_path.replace('h5', 'npz'))

    def save_to_h5(self, file_path):
        """
        Saves the dataset to an HDF5 file.
        """
        with h5py.File(file_path, 'w') as h5f:
            h5f.create_dataset('grad_directions', data=np.array([self.grad_directions]), dtype=np.int64)
            volumes_group = h5f.create_group('volumes')
            for idx, volume in enumerate(self.volumes):
                volumes_group.create_dataset(str(idx), data=volume, compression="gzip")
            h5f.create_dataset('streamline_id_to_volume_id', data=self.streamline_id_to_volume_id)

        self.streamlines.save(file_path.replace('h5', 'npz'))
        
    def split_into_train_valid(self, split_frac=0.9):
        
        np.random.seed(42)
        if len(self.volumes) > 1:
            volume_indices = np.random.permutation(len(self.volumes))

            split = np.ceil((1 - split_frac) * len(self.volumes)).astype(int)

            valid_volume_indices = volume_indices[:split]
            train_volume_indices = volume_indices[split:]

            volume_id_map = {old_id: new_id for new_id, old_id in enumerate(train_volume_indices)}
            volume_id_map.update({old_id: new_id for new_id, old_id in enumerate(valid_volume_indices)})

            train_volumes = [self.volumes[i] for i in train_volume_indices]
            valid_volumes = [self.volumes[i] for i in valid_volume_indices]

            del self.volumes

            train_streamline_indice = np.where(np.isin(self.streamline_id_to_volume_id, train_volume_indices))[0]
            valid_streamline_indice = np.where(np.isin(self.streamline_id_to_volume_id, valid_volume_indices))[0]

            train_streamlines = self.streamlines[train_streamline_indice]
            valid_streamlines = self.streamlines[valid_streamline_indice]

            del self.streamlines

            train_streamline_id_to_volume_id = np.array([volume_id_map[v_id] for v_id in np.array(self.streamline_id_to_volume_id)[train_streamline_indice]])     
            valid_streamline_id_to_volume_id = np.array([volume_id_map[v_id] for v_id in np.array(self.streamline_id_to_volume_id)[valid_streamline_indice]])

            del self.streamline_id_to_volume_id

            train_dataset = copy.deepcopy(self)
            valid_dataset = copy.deepcopy(self)

            train_dataset.streamlines = nibabel.streamlines.ArraySequence()
            train_dataset.streamlines.extend(train_streamlines)
            del train_streamlines
            
            train_dataset.streamline_id_to_volume_id = train_streamline_id_to_volume_id
            del train_streamline_id_to_volume_id
            train_dataset.volumes = train_volumes
            del train_volumes

            valid_dataset.streamlines = nibabel.streamlines.ArraySequence()
            valid_dataset.streamlines.extend(valid_streamlines)
            del valid_streamlines

            valid_dataset.streamline_id_to_volume_id = valid_streamline_id_to_volume_id
            del valid_streamline_id_to_volume_id
            valid_dataset.volumes = valid_volumes
            del valid_volumes
        else:
            indices = np.random.permutation(len(self.streamlines))
            
            split = int(split_frac * len(self.streamlines))
            
            train_indices = indices[:split]
            valid_indices = indices[split:]

            train_streamlines = self.streamlines[train_indices]
            valid_streamlines = self.streamlines[valid_indices]
            
            del self.streamlines
            
            train_streamline_id_to_volume_id = np.array(self.streamline_id_to_volume_id)[train_indices]
            valid_streamline_id_to_volume_id = np.array(self.streamline_id_to_volume_id)[valid_indices]

            del self.streamline_id_to_volume_id

            train_dataset = copy.deepcopy(self)
            valid_dataset = copy.deepcopy(self)

            del self.volumes
            
            train_dataset.streamlines = nibabel.streamlines.ArraySequence()
            train_dataset.streamlines.extend(train_streamlines)
            del train_streamlines

            train_dataset.streamline_id_to_volume_id = train_streamline_id_to_volume_id
            del train_streamline_id_to_volume_id

            valid_dataset.streamlines = nibabel.streamlines.ArraySequence()
            valid_dataset.streamlines.extend(valid_streamlines)
            del valid_streamlines

            valid_dataset.streamline_id_to_volume_id = valid_streamline_id_to_volume_id
            del valid_streamline_id_to_volume_id

        return train_dataset, valid_dataset

    def fun_flip_streamline(self):
        flip_streamline = [s[::-1] for s in self.streamlines]
        self.streamlines.extend(flip_streamline)
        self.streamline_id_to_volume_id = np.concatenate([self.streamline_id_to_volume_id, self.streamline_id_to_volume_id])
    
    def __getitem__(self, idx):
        volume = self.volumes[self.streamline_id_to_volume_id[idx]]
        streamline = self.streamlines[idx]
        target = streamline[1:] - streamline[:-1]
        target = target / (np.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
        
        return volume, streamline[:-1], target
