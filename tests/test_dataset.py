import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import nibabel as nib
import numpy as np

from utils.dataset import (
    TractographyData,
    TractographyDataset,
    calculate_directions,
    load_dataset,
)


def make_dataset(volume_ids):
    dataset = TractographyDataset.__new__(TractographyDataset)
    dataset.grad_directions = 1
    dataset.volumes = [np.full((1, 1, 1, 1), index, dtype=np.float32)
                       for index in range(max(volume_ids, default=-1) + 1)]
    dataset.streamlines = nib.streamlines.ArraySequence()
    dataset.streamline_id_to_volume_id = np.asarray(volume_ids, dtype=np.int64)
    for index, volume_id in enumerate(volume_ids):
        x = 100 * volume_id + index
        dataset.streamlines.append(np.array([[x, 0., 0.], [x + 1., 0., 0.]], dtype=np.float32))
    dataset.flip_streamline = False
    dataset.subject_streamlines = {}
    return dataset


class TractographyDataTests(unittest.TestCase):
    def _write_dwi(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.full((3, 4, 5, 7), value, dtype=np.float32)
        affine = np.diag([2., 2., 2., 1.])
        nib.save(nib.Nifti1Image(data, affine), path)
        bvals = np.array([0., 1000., 1000., 1000., 1000., 1000., 1000.])
        bvecs = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
        ])
        bvecs[4:] /= np.linalg.norm(bvecs[4:], axis=1, keepdims=True)
        stem = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
        np.savetxt(path.parent / f'{stem}.bval', bvals)
        np.savetxt(path.parent / f'{stem}.bvec', bvecs.T)
        return data, affine

    def test_load_dwi_uses_selected_nifti_and_matching_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'sub-01' / 'dwi'
            self._write_dwi(directory / 'a_dwi.nii.gz', 1.)
            expected, _ = self._write_dwi(directory / 'z_dwi.nii.gz', 9.)
            subject = TractographyData.__new__(TractographyData)
            subject.dwi_path = str(directory / 'z_dwi.nii.gz')

            subject.load_dwi()
            np.testing.assert_array_equal(subject.dwi.get_fdata(), expected)
            self.assertEqual(len(subject.gradients.bvals), 7)

    def test_load_dwi_rejects_ambiguous_matching_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'dwi'
            dwi = directory / 'sub_dwi.nii.gz'
            self._write_dwi(dwi, 1.)
            np.savetxt(directory / 'sub_dwi.bvals', np.zeros(7))
            subject = TractographyData.__new__(TractographyData)
            subject.dwi_path = str(dwi)
            with self.assertRaisesRegex(ValueError, 'Multiple bval files'):
                subject.load_dwi()

    def test_load_dwi_does_not_match_sidecars_with_unrelated_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'dwi'
            dwi = directory / 'a_dwi.nii.gz'
            self._write_dwi(dwi, 1.)
            (directory / 'a_dwi_other.bval').write_text('0 1000 1000 1000 1000 1000 1000')
            subject = TractographyData.__new__(TractographyData)
            subject.dwi_path = str(dwi)

            subject.load_dwi()
            self.assertEqual(len(subject.gradients.bvals), 7)

    def test_load_dwi_rejects_missing_b0_and_zero_diffusion_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'dwi'
            dwi = directory / 'sub_dwi.nii.gz'
            self._write_dwi(dwi, 1.)
            bval_path = directory / 'sub_dwi.bval'
            bvec_path = directory / 'sub_dwi.bvec'
            bvals = np.full(7, 1000.)
            bvecs = np.eye(3)[[0, 1, 2, 0, 1, 2, 0]]
            np.savetxt(bval_path, bvals)
            np.savetxt(bvec_path, bvecs.T)
            subject = TractographyData.__new__(TractographyData)
            subject.dwi_path = str(dwi)
            with self.assertRaisesRegex(ValueError, 'At least one b=0 volume'):
                subject.load_dwi()

            bvals[0] = 0.
            bvecs[1] = 0.
            np.savetxt(bval_path, bvals)
            np.savetxt(bvec_path, bvecs.T)
            with self.assertRaisesRegex(ValueError, 'nonzero b-vector'):
                subject.load_dwi()

    def test_preprocessing_rejects_nonfinite_dwi_values(self):
        subject = TractographyData.__new__(TractographyData)
        subject.dwi_path = 'synthetic.nii.gz'
        subject.dwi = nib.Nifti1Image(np.full((2, 2, 2, 2), np.nan, dtype=np.float32), np.eye(4))
        subject.brain_mask = np.ones((2, 2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, 'non-finite values'):
            subject.preprocess_dwi()

    def test_load_mask_requires_same_3d_grid_and_finite_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference = nib.Nifti1Image(np.zeros((3, 4, 5, 7), dtype=np.float32), np.eye(4))
            mask_path = directory / 'mask.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4)), mask_path)
            mask = TractographyData.load_mask(mask_path, reference)
            self.assertEqual(mask.shape, (3, 4, 5))
            self.assertTrue(mask.all())

            shifted = np.eye(4)
            shifted[0, 3] = 0.25
            nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), shifted), mask_path)
            with self.assertRaisesRegex(ValueError, 'same three-dimensional grid'):
                TractographyData.load_mask(mask_path, reference)

            nib.save(nib.Nifti1Image(np.ones((3, 4, 5, 1), dtype=np.float32), np.eye(4)), mask_path)
            with self.assertRaisesRegex(ValueError, 'same three-dimensional grid'):
                TractographyData.load_mask(mask_path, reference)

    def test_csd_preprocessing_forwards_requested_sh_order(self):
        subject = TractographyData.__new__(TractographyData)
        subject.dwi = nib.Nifti1Image(np.ones((2, 2, 2, 3), dtype=np.float32), np.eye(4))
        subject.brain_mask = np.ones((2, 2, 2), dtype=bool)
        subject.gradients = object()
        subject.sh_order = 8
        subject.input_type = 'fodf'
        coefficients = np.zeros((2, 2, 2, 45), dtype=np.float32)
        with patch('utils.dataset.compute_fodf_ssst', return_value=SimpleNamespace(shm_coeff=coefficients)) as fit:
            subject.preprocess_dwi()
        self.assertEqual(fit.call_args.kwargs['sh_order'], 8)
        self.assertEqual(subject.volume.shape[-1], 45)

    def test_tractogram_resampling_honors_requested_physical_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference_path = directory / 'reference.nii.gz'
            affine = np.eye(4)
            reference = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine)
            nib.save(reference, reference_path)
            header = nib.streamlines.TrkFile.create_empty_header()
            header['dimensions'] = reference.shape
            header['voxel_sizes'] = nib.affines.voxel_sizes(affine)
            header['voxel_to_rasmm'] = affine
            line = np.array([[1., 1., 1.], [4., 1., 1.]], dtype=np.float32)
            tractogram_path = directory / 'tract.trk'
            nib.streamlines.save(
                nib.streamlines.TrkFile(nib.streamlines.Tractogram([line], affine_to_rasmm=affine), header),
                tractogram_path)

            subject = TractographyData.__new__(TractographyData)
            subject.tractogram_path = str(tractogram_path)
            subject.dwi = nib.load(reference_path)
            subject.load_tractogram(step_size=1.)

        resampled = subject.tractogram.streamlines[0]
        self.assertEqual(len(resampled), 4)
        np.testing.assert_allclose(np.linalg.norm(np.diff(resampled, axis=0), axis=1), 1., atol=1e-5)

    def test_tractogram_removes_repeated_vertices_and_drops_collapsed_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference = nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4))
            header = nib.streamlines.TrkFile.create_empty_header()
            header['dimensions'] = reference.shape
            header['voxel_sizes'] = nib.affines.voxel_sizes(reference.affine)
            header['voxel_to_rasmm'] = reference.affine
            lines = [
                np.array([[1., 1., 1.], [1., 1., 1.], [3., 1., 1.]], dtype=np.float32),
                np.array([[2., 2., 2.], [2., 2., 2.]], dtype=np.float32),
            ]
            tractogram_path = directory / 'tract.trk'
            nib.streamlines.save(
                nib.streamlines.TrkFile(
                    nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4)), header),
                tractogram_path)

            subject = TractographyData.__new__(TractographyData)
            subject.tractogram_path = str(tractogram_path)
            subject.dwi = reference
            subject.load_tractogram(step_size=1.)

        self.assertEqual(len(subject.tractogram.streamlines), 1)
        resampled = subject.tractogram.streamlines[0]
        self.assertEqual(len(resampled), 3)
        np.testing.assert_allclose(np.linalg.norm(np.diff(resampled, axis=0), axis=1), 1., atol=1e-5)


class TractographyDatasetTests(unittest.TestCase):
    def test_split_preserves_source_and_keeps_volume_ids_aligned(self):
        source = make_dataset([0, 0, 1, 1, 2, 2])
        original_lines = [line.copy() for line in source.streamlines]
        original_mapping = source.streamline_id_to_volume_id.copy()
        original_volumes = [volume.copy() for volume in source.volumes]

        train, valid = source.split_into_train_valid(split_frac=0.67)

        np.testing.assert_array_equal(source.streamline_id_to_volume_id, original_mapping)
        self.assertEqual(len(source.volumes), 3)
        for actual, expected in zip(source.streamlines, original_lines):
            np.testing.assert_array_equal(actual, expected)
        for volume in source.volumes:
            self.assertTrue(any(np.array_equal(volume, expected) for expected in original_volumes))

        self.assertEqual(len(train.volumes) + len(valid.volumes), 3)
        combined_ids = []
        for subset in (train, valid):
            self.assertEqual(len(subset.streamlines), len(subset.streamline_id_to_volume_id))
            for streamline, volume_id in zip(subset.streamlines, subset.streamline_id_to_volume_id):
                source_id = int(streamline[0, 0]) // 100
                self.assertEqual(int(subset.volumes[volume_id][0, 0, 0, 0]), source_id)
                combined_ids.append(int(streamline[0, 0]))
        self.assertCountEqual(combined_ids, [int(line[0, 0]) for line in original_lines])

    def test_single_volume_split_keeps_volume_for_both_subsets(self):
        source = make_dataset([0, 0, 0, 0, 0])
        train, valid = source.split_into_train_valid(split_frac=0.6)
        self.assertEqual(len(train.volumes), 1)
        self.assertEqual(len(valid.volumes), 1)
        self.assertEqual(len(train.streamlines) + len(valid.streamlines), 5)
        self.assertEqual(len(source.streamlines), 5)

    def test_hdf5_volumes_load_in_numeric_id_order_and_sidecar_path_is_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'parent-h5-data'
            directory.mkdir()
            source = make_dataset(list(range(12)))
            path = directory / 'data.h5'
            source.save_to_h5(path)
            self.assertTrue((directory / 'data.npz').is_file())
            self.assertFalse((Path(temporary) / 'parent-npz-data').exists())

            loaded = TractographyDataset(h5_file=path)
            self.assertEqual([int(volume[0, 0, 0, 0]) for volume in loaded.volumes], list(range(12)))
            np.testing.assert_array_equal(loaded.streamline_id_to_volume_id, np.arange(12))

    def test_hdf5_rejects_noncontiguous_volume_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'broken.h5'
            with h5py.File(path, 'w') as h5f:
                h5f.create_dataset('grad_directions', data=[1])
                volumes = h5f.create_group('volumes')
                volumes.create_dataset('0', data=np.zeros((1, 1, 1, 1)))
                volumes.create_dataset('2', data=np.zeros((1, 1, 1, 1)))
                h5f.create_dataset('streamline_id_to_volume_id', data=[])
            with self.assertRaisesRegex(ValueError, 'contiguous numeric IDs'):
                TractographyDataset(h5_file=path)

    def test_directory_loader_uses_requested_mode_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            make_dataset([0, 0]).save_to_h5(directory / 'validset_sh_sh6_step0.5_dataset.h5')
            make_dataset([0, 0, 0]).save_to_h5(directory / 'validset_fodf_sh8_step0.5_dataset.h5')
            make_dataset([0]).save_to_h5(directory / 'unrelated.h5')

            loaded = load_dataset(directory, mode='validset')

        self.assertEqual(len(loaded.streamlines), 2)

    def test_directory_loader_requires_explicit_hdf5_path_for_existing_cache_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / 'unrelated.h5'
            make_dataset([0, 0]).save_to_h5(path)
            with self.assertRaisesRegex(FileNotFoundError, 'pass an HDF5 file path explicitly'):
                load_dataset(directory, mode='trainset')
            self.assertEqual(len(load_dataset(path, mode='trainset').streamlines), 2)

    def test_calculate_directions_returns_zero_for_duplicate_points(self):
        directions = calculate_directions(np.array([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]))
        np.testing.assert_array_equal(directions, [[0., 0., 0.], [1., 0., 0.]])
        self.assertTrue(np.isfinite(directions).all())


if __name__ == '__main__':
    unittest.main()
