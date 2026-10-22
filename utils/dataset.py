from concurrent.futures import ProcessPoolExecutor

import h5py
import copy
import multiprocessing
import os
from pathlib import Path
import nibabel as nib
import nibabel.streamlines
import numpy as np
import torch.utils.data as data
from dipy.io import read_bvals_bvecs
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
    wm_mask_path = locate_files(input_path, 'wm-mask')
    print("load subject's tract from {}".format(input_path))
    tract_path = locate_files(input_path, 'tract')

    subject = TractographyData('train', dwi_path, brain_mask_path, wm_mask_path, tract_path, sh_order=sh_order, input_type=input_type, vox_step=vox_step)
    return subject


def load_dataset(path, mode, sh_order=6, input_type='sh', flip_streamline=False, vox_step=0.5):
    path = Path(path)
    if not path.is_dir():
        return TractographyDataset(h5_file=str(path), flip_streamline=flip_streamline)

    cached = path / f'{mode}_{input_type}_sh{sh_order}_step{vox_step:g}_dataset.h5'
    if cached.is_file():
        return TractographyDataset(h5_file=str(cached), flip_streamline=flip_streamline)

    subject_dirs = sorted(child for child in path.iterdir() if child.is_dir())
    if not subject_dirs:
        raise FileNotFoundError(
            f"No subject directories or matching cache {cached.name} found in {path}; "
            "pass an HDF5 file path explicitly to load an existing dataset.")

    workers = min(32, (os.cpu_count() or 1) + 4, len(subject_dirs))
    if workers == 1:
        subjects = [load_subject(str(subject), input_type, sh_order, vox_step)
                    for subject in subject_dirs]
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(load_subject, str(subject), input_type, sh_order, vox_step)
                       for subject in subject_dirs]
            subjects = [future.result() for future in futures]

    dataset = TractographyDataset(subjects=subjects, flip_streamline=flip_streamline)
    dataset.save_to_h5(str(cached))
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

        if self.dwi_path is None:
            raise ValueError("A DWI path is required.")
        if self.brain_mask_path is None:
            raise ValueError("A brain mask path is required.")
        if self.mode == 'train' and self.tractogram_path is None:
            raise ValueError("Tractogram path must be provided for training mode.")

        # Initialize attributes to None
        self.dwi = self.gradients = self.brain_mask = self.wm_mask = self.tractogram = self.ID = None
        self.volume = None

        dwi_parent = Path(self.dwi_path).parent
        self.ID = dwi_parent.parent.name or dwi_parent.name

        # Load data
        if self.dwi_path is not None:
            self.load_dwi()
        if self.brain_mask_path is not None:
            self.brain_mask = self.load_mask(self.brain_mask_path, self.dwi).astype(np.bool_)
        if self.wm_mask_path is not None:
            self.wm_mask = self.load_mask(self.wm_mask_path, self.dwi).astype(np.bool_)
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
        dwi_path = Path(self.dwi_path)
        self.dwi = nib.load(str(dwi_path))
        if len(self.dwi.shape) != 4:
            raise ValueError(f"DWI must be four-dimensional: {dwi_path}")

        name = dwi_path.name
        stem = name[:-7] if name.endswith('.nii.gz') else dwi_path.stem

        def sidecar(kind):
            candidates = [dwi_path.parent / f'{stem}.{kind}',
                          dwi_path.parent / f'{stem}.{kind}s']
            matches = [candidate for candidate in candidates if candidate.is_file()]
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise FileNotFoundError(f"Could not find the {kind} file paired with {dwi_path}.")
            raise ValueError(f"Multiple {kind} files match {dwi_path}: {matches}")

        bval_path = sidecar('bval')
        bvec_path = sidecar('bvec')
        bvals, bvecs = read_bvals_bvecs(str(bval_path), str(bvec_path))
        bvals = np.asarray(bvals).reshape(-1)
        bvecs = np.asarray(bvecs)
        if len(bvals) != self.dwi.shape[3]:
            raise ValueError("b-value count must match the DWI fourth dimension.")
        if bvecs.shape != (len(bvals), 3):
            raise ValueError("b-vector count must match the DWI fourth dimension.")
        if not np.isfinite(bvals).all() or (bvals < 0).any() or not np.isfinite(bvecs).all():
            raise ValueError("Gradient sidecars must contain finite, nonnegative b-values and finite b-vectors.")
        b0_mask = bvals <= 50
        if not b0_mask.any():
            raise ValueError("At least one b=0 volume is required for DWI normalization.")
        if (np.linalg.norm(bvecs[~b0_mask], axis=1) <= 0).any():
            raise ValueError("Every diffusion-weighted volume must have a nonzero b-vector.")
        self.gradients = gradient_table(bvals, bvecs, b0_threshold=50)

    @staticmethod
    def load_mask(mask_path, reference):
        image = nib.load(str(mask_path))
        if len(image.shape) != 3 or image.shape != reference.shape[:3] or not np.allclose(
                image.affine, reference.affine, rtol=0, atol=1e-4):
            raise ValueError(f"Mask must have the same three-dimensional grid as the DWI: {mask_path}")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"Mask contains non-finite values: {mask_path}")
        return data > 0.1

    def load_tractogram(self, step_size=None):
        tract_file = nib.streamlines.load(self.tractogram_path)
        tractogram = tract_file.tractogram

        # Resample streamline to have a fixed step size, if needed.
        if step_size is not None:
            if step_size <= 0:
                raise ValueError("Streamline resampling step size must be positive.")
            print("Resampling streamlines to have a step size of {}mm".format(step_size))
            streamlines = []
            for line in tractogram.streamlines:
                if not np.isfinite(line).all():
                    raise ValueError("Tractogram contains non-finite streamline coordinates.")
                if len(line):
                    keep = np.r_[True, np.linalg.norm(np.diff(line, axis=0), axis=1) > 0]
                    line = line[keep]
                if len(line) >= 2:
                    streamlines.append(line)
            lengths = length(streamlines)
            nb_points = np.ceil(lengths / step_size).astype(int) + 1
            new_streamlines = (set_number_of_points(line, int(count))
                               for line, count in zip(streamlines, nb_points))
            tractogram = nib.streamlines.Tractogram(new_streamlines, affine_to_rasmm=np.eye(4))

        # Compute matrix that brings streamlines back to diffusion voxel space.
        rasmm2vox_affine = np.linalg.inv(self.dwi.affine)
        tractogram.apply_affine(rasmm2vox_affine)

        self.tractogram = tractogram

    def preprocess_dwi(self):
        dwi_data = self.dwi.get_fdata().astype(np.float32)
        if not np.isfinite(dwi_data).all():
            raise ValueError(f"DWI contains non-finite values: {self.dwi_path}")
        self.masked_dwi = mask_dwi(dwi_data, self.brain_mask)
        if self.input_type == 'fodf':
            print("compute fodf using CSD")
            csd_fit = compute_fodf_ssst(self.masked_dwi, self.gradients, self.brain_mask,
                                        sh_order=self.sh_order)
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
    return np.divide(directions, norms, out=np.zeros_like(directions, dtype=np.float32),
                     where=norms > 0)


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
                if subject.volume.shape[-1] != self.grad_directions:
                    raise ValueError("All subjects must have the same number of input features.")
                sub_streamlines = subject.tractogram.streamlines
                
                self.streamlines.extend(sub_streamlines)
                self.streamline_id_to_volume_id.extend([idx] * len(sub_streamlines))
                
                self.volumes.append(subject.volume)

            subjects = None
            
    def load_from_h5(self, file_path):
        with h5py.File(file_path, 'r') as h5f:
            self.grad_directions = h5f['grad_directions'][:][0]
            volume_keys = sorted(h5f['volumes'].keys(), key=int)
            if volume_keys != [str(index) for index in range(len(volume_keys))]:
                raise ValueError("HDF5 volume keys must be contiguous numeric IDs starting at zero.")
            self.volumes = [h5f['volumes'][key][()] for key in volume_keys]
            self.streamline_id_to_volume_id = h5f['streamline_id_to_volume_id'][:].astype(np.int64)

        self.streamlines = nibabel.streamlines.ArraySequence().load(str(Path(file_path).with_suffix('.npz')))
        if len(self.streamline_id_to_volume_id) != len(self.streamlines):
            raise ValueError("HDF5 streamline-to-volume mapping does not match the saved streamlines.")
        if (self.streamline_id_to_volume_id < 0).any() or (self.streamline_id_to_volume_id >= len(self.volumes)).any():
            raise ValueError("HDF5 streamline-to-volume mapping references a missing volume.")

    def save_to_h5(self, file_path):
        """
        Saves the dataset to an HDF5 file.
        """
        file_path = Path(file_path)
        with h5py.File(file_path, 'w') as h5f:
            h5f.create_dataset('grad_directions', data=np.array([self.grad_directions]), dtype=np.int64)
            volumes_group = h5f.create_group('volumes')
            for idx, volume in enumerate(self.volumes):
                volumes_group.create_dataset(str(idx), data=volume, compression="gzip")
            h5f.create_dataset('streamline_id_to_volume_id', data=self.streamline_id_to_volume_id)

        self.streamlines.save(str(file_path.with_suffix('.npz')))
        
    def split_into_train_valid(self, split_frac=0.9):
        if not 0 < split_frac < 1:
            raise ValueError("split_frac must be between zero and one.")
        if len(self.streamlines) < 2:
            raise ValueError("At least two streamlines are required for a train/validation split.")

        rng = np.random.RandomState(42)
        volume_ids = np.asarray(self.streamline_id_to_volume_id, dtype=np.int64)
        if len(self.volumes) > 1:
            order = rng.permutation(len(self.volumes))
            valid_count = int(np.ceil((1 - split_frac) * len(self.volumes)))
            if not 0 < valid_count < len(self.volumes):
                raise ValueError("split_frac must leave at least one subject in each split.")
            groups = (order[valid_count:], order[:valid_count])
        else:
            order = rng.permutation(len(self.streamlines))
            train_count = int(split_frac * len(self.streamlines))
            if not 0 < train_count < len(self.streamlines):
                raise ValueError("split_frac must leave at least one streamline in each split.")
            groups = (order[:train_count], order[train_count:])

        datasets = []
        for group in groups:
            dataset = copy.copy(self)
            if len(self.volumes) > 1:
                dataset.volumes = [self.volumes[index] for index in group]
                volume_id_map = {int(old_id): new_id for new_id, old_id in enumerate(group)}
                indices = np.flatnonzero(np.isin(volume_ids, group))
                dataset.streamline_id_to_volume_id = np.array(
                    [volume_id_map[int(volume_ids[index])] for index in indices], dtype=np.int64)
            else:
                dataset.volumes = list(self.volumes)
                indices = group
                dataset.streamline_id_to_volume_id = volume_ids[indices].copy()
            dataset.streamlines = nibabel.streamlines.ArraySequence()
            dataset.streamlines.extend(self.streamlines[indices])
            datasets.append(dataset)
        return datasets[0], datasets[1]

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
