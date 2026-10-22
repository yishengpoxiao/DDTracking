import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import vtk
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile

from slicer_backend.preprocessing import rotate_bvecs as rotate_bvecs_ras, transform_ras_matrix
from utilize.extract_dwi_bshell import extract_bshell, volume_iterator
from utilize.merge_tracts import merge_trks
from utilize.register_volume_to_MNI import align_to_MNI, read_bvecs, rotate_bvecs
from utilize.segment_tract_with_ROI import segment_tract_use_ROI


class UtilityTests(unittest.TestCase):
    def test_shell_extraction_handles_block_boundaries_and_preserves_order(self):
        data = np.arange(2 * 3 * 4 * 7, dtype=np.float32).reshape(2, 3, 4, 7)
        image = nib.Nifti1Image(data, np.eye(4))
        bvals = np.array([0, 1000, 2000, 1000, 990, 2000, 50])
        bvecs = np.arange(21, dtype=np.float32).reshape(7, 3)
        indices, extracted, out_bvals, out_bvecs = extract_bshell(
            image, bvals, bvecs, [1000], tol=20, block_size=3)
        np.testing.assert_array_equal(indices, [1, 3, 4])
        np.testing.assert_array_equal(extracted, data[..., indices])
        np.testing.assert_array_equal(out_bvals, [1000, 1000, 990])
        np.testing.assert_array_equal(out_bvecs, bvecs[indices])

    def test_shell_extraction_rejects_empty_or_misaligned_gradients(self):
        image = nib.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4))
        with self.assertRaisesRegex(ValueError, "At least one b-value shell"):
            extract_bshell(image, [0, 1000], np.zeros((2, 3)), [])
        with self.assertRaisesRegex(ValueError, "counts must match"):
            extract_bshell(image, [0], np.zeros((1, 3)), [1000])

    def test_registration_bvec_rotation_inverts_fixed_to_moving_and_handles_single_volume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            native_path = root / "native.nii.gz"
            template_path = root / "template.nii.gz"
            bvec_path = root / "input.bvec"
            transform_path = root / "fixed_to_moving.tfm"
            out_path = root / "rotated.bvec"
            native_affine = np.eye(4)
            template_affine = np.diag([-2., 2., 2., 1.])
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), native_affine), native_path)
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), template_affine), template_path)
            np.savetxt(bvec_path, np.array([[1.], [0.], [0.]]))
            fixed_to_moving = sitk.Euler3DTransform()
            fixed_to_moving.SetRotation(0., 0., np.pi / 2.)
            sitk.WriteTransform(fixed_to_moving, str(transform_path))

            self.assertEqual(read_bvecs(str(bvec_path)).shape, (1, 3))
            rotate_bvecs(str(bvec_path), str(transform_path), str(native_path),
                         str(template_path), str(out_path))
            actual = read_bvecs(str(out_path))
            expected = rotate_bvecs_ras(
                np.array([[1., 0., 0.]]), native_affine, template_affine,
                np.linalg.inv(transform_ras_matrix(fixed_to_moving)))
            np.testing.assert_allclose(actual, expected, atol=1e-6)
            self.assertFalse(np.allclose(actual, rotate_bvecs_ras(
                np.array([[1., 0., 0.]]), native_affine, template_affine,
                transform_ras_matrix(fixed_to_moving))))

    def test_registration_sidecars_do_not_match_unrelated_filename_prefixes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            dwi = directory / "sub-01_dwi.nii.gz"
            template = directory / "template.nii.gz"
            brain = directory / "brain.nii.gz"
            for path in (dwi, template, brain):
                path.touch()
            (directory / "sub-01_dwi_other.bval").touch()
            (directory / "sub-01_dwi_other.bvec").touch()
            with self.assertRaisesRegex(ValueError, "found 0; pass it explicitly"):
                align_to_MNI(str(dwi), str(template), str(brain), None)

    def test_merge_rejects_different_trk_reference_grids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trk_dir = root / "bundles"
            trk_dir.mkdir()
            streamline = np.array([[1., 2., 3.], [2., 3., 4.]])
            for name, xshift in (("a.trk", 0.), ("b.trk", 1.)):
                header = TrkFile.create_empty_header()
                header["dimensions"] = [10, 10, 10]
                affine = np.eye(4)
                affine[0, 3] = xshift
                header["voxel_to_rasmm"] = affine
                TrkFile(Tractogram([streamline], affine_to_rasmm=np.eye(4)), header).save(trk_dir / name)
            with self.assertRaisesRegex(ValueError, "reference geometry differs"):
                merge_trks(str(trk_dir))

    def test_merge_saves_world_streamlines_with_first_trk_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trk_dir = root / "bundles"
            trk_dir.mkdir()
            expected = np.array([[1., 2., 3.], [2., 3., 4.]])
            header = TrkFile.create_empty_header()
            header["dimensions"] = [10, 10, 10]
            header["voxel_to_rasmm"] = np.eye(4)
            for name in ("a.trk", "b.trk"):
                TrkFile(Tractogram([expected], affine_to_rasmm=np.eye(4)), header).save(trk_dir / name)
            output = Path(merge_trks(str(trk_dir)))
            loaded = nib.streamlines.load(str(output))
            self.assertEqual(len(loaded.tractogram.streamlines), 2)
            for streamline in loaded.tractogram.streamlines:
                np.testing.assert_allclose(streamline, expected, atol=1e-4)

    def test_merge_into_input_directory_does_not_reingest_previous_output(self):
        with tempfile.TemporaryDirectory() as temp:
            trk_dir = Path(temp) / "bundle"
            trk_dir.mkdir()
            header = TrkFile.create_empty_header()
            header["dimensions"] = [10, 10, 10]
            header["voxel_to_rasmm"] = np.eye(4)
            line = np.array([[1., 2., 3.], [2., 3., 4.]])
            TrkFile(Tractogram([line], affine_to_rasmm=np.eye(4)), header).save(trk_dir / "a.trk")
            TrkFile(Tractogram([line], affine_to_rasmm=np.eye(4)), header).save(trk_dir / "b.trk")

            output = merge_trks(str(trk_dir), out_dir=str(trk_dir))
            merge_trks(str(trk_dir), out_dir=str(trk_dir))
            self.assertEqual(len(nib.streamlines.load(output).tractogram.streamlines), 2)

    def test_merge_requires_integer_num_keep(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "num_keep must be a positive integer"):
                merge_trks(temp, num_keep=1.5)

    def test_roi_filter_rejects_negative_voxel_wraparound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shape = (40, 20, 20)
            reference_path = root / "reference.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), np.eye(4)), reference_path)
            tract_path = root / "tract.trk"
            header = TrkFile.create_empty_header()
            header["dimensions"] = shape
            header["voxel_to_rasmm"] = np.eye(4)
            good = np.array([[5.5, 5.5, 5.5], [20.5, 5.5, 5.5], [39.4, 5.5, 5.5]])
            wraps_negative = np.array([[5.5, 5.5, 5.5], [-1.5, 5.5, 5.5], [39.4, 5.5, 5.5]])
            TrkFile(Tractogram([good, wraps_negative], affine_to_rasmm=np.eye(4)), header).save(tract_path)
            tractseg = root / "tractseg"
            endpoints = tractseg / "endings_segmentations"
            bundles = tractseg / "bundle_segmentations"
            endpoints.mkdir(parents=True)
            bundles.mkdir()
            for suffix, x in (("b", 6), ("e", 39)):
                mask = np.zeros(shape, dtype=np.uint8)
                mask[x, 6, 6] = 1
                nib.save(nib.Nifti1Image(mask, np.eye(4)), endpoints / f"bundle_{suffix}.nii.gz")
            all_mask = np.zeros(shape, dtype=np.uint8)
            all_mask[6:40, 6, 6] = 1
            nib.save(nib.Nifti1Image(all_mask, np.eye(4)), bundles / "bundle.nii.gz")

            output = segment_tract_use_ROI(str(tract_path), str(reference_path), "bundle",
                                          str(root / "out"), str(tractseg), 0.5)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(output)
            reader.Update()
            polydata = reader.GetOutput()
            self.assertEqual(polydata.GetNumberOfLines(), 1)
            points = np.array([polydata.GetPoint(index) for index in range(polydata.GetNumberOfPoints())])
            np.testing.assert_allclose(points, good, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
