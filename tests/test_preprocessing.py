import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from dipy.data import get_sphere

from slicer_backend.preprocessing import (
    read_gradients,
    register_fa,
    resample,
    same_grid_data,
    to_sitk,
    transform_ras_matrix,
    rotate_bvecs,
    validate_sh_sampling,
)


class PreprocessingGeometryTests(unittest.TestCase):
    def test_fsl_bvec_axes_follow_90_degree_ras_rotation_for_image_handedness(self):
        template_affine = np.diag([-2., 2., 2., 1.])  # Published template frame is LAS.
        rotation = np.eye(4)
        rotation[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        cases = (
            ("positive determinant", np.eye(4), np.array([
                [0., 1., 0.], [-1., 0., 0.], [0., 0., 1.],
            ])),
            ("negative determinant", np.diag([-1., 1., 1., 1.]), np.array([
                [0., 1., 0.], [-1., 0., 0.], [0., 0., 1.],
            ])),
            ("oblique native", np.array([
                [np.cos(np.pi / 6), -np.sin(np.pi / 6), 0., 0.],
                [np.sin(np.pi / 6), np.cos(np.pi / 6), 0., 0.],
                [0., 0., 1., 0.], [0., 0., 0., 1.],
            ]), np.array([
                [-0.5, np.sqrt(3.) / 2., 0.],
                [-np.sqrt(3.) / 2., -0.5, 0.],
                [0., 0., 1.],
            ])),
        )
        bvecs = np.eye(3)
        for name, native_affine, expected_rotation in cases:
            with self.subTest(name=name):
                actual = rotate_bvecs(bvecs, native_affine, template_affine, rotation)
                expected = bvecs @ expected_rotation.T
                np.testing.assert_allclose(actual, expected, atol=1e-7)
                np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=1e-7)
                np.testing.assert_allclose(actual @ expected_rotation, bvecs, atol=1e-7)

    def test_centered_sitk_transform_converts_to_ras_and_roundtrips(self):
        transform = sitk.Euler3DTransform()
        transform.SetCenter((8., -4., 2.))
        transform.SetRotation(0., 0., np.pi / 2.)
        transform.SetTranslation((3., 5., -2.))
        matrix = transform_ras_matrix(transform)
        point_ras = np.array([1.5, -2., 7., 1.])
        point_lps = (-point_ras[0], -point_ras[1], point_ras[2])
        transformed_lps = np.asarray(transform.TransformPoint(point_lps))
        expected_ras = np.array([-transformed_lps[0], -transformed_lps[1], transformed_lps[2], 1.])
        np.testing.assert_allclose(matrix @ point_ras, expected_ras, atol=1e-8)
        self.assertGreater(np.linalg.norm(matrix[:3, 3]), 1.)
        self.assertGreater(np.linalg.norm(transform.GetCenter()), 1.)
        roundtrip = np.linalg.inv(matrix) @ (matrix @ point_ras)
        np.testing.assert_allclose(roundtrip, point_ras, atol=1e-8)

    def test_resample_moves_landmark_in_physical_space_and_keeps_mask_binary(self):
        affine = np.eye(4)
        source = np.zeros((5, 5, 5), dtype=np.float32)
        source[2, 2, 2] = 1.
        mask = np.zeros_like(source)
        mask[1:4, 1:4, 1:4] = 1
        reference = to_sitk(np.zeros_like(source), affine)
        fixed_to_moving = sitk.TranslationTransform(3, (1., 0., 0.))

        moved = resample(source, affine, reference, fixed_to_moving)
        moved_mask = resample(mask, affine, reference, fixed_to_moving, mask=True)
        self.assertEqual(np.unravel_index(np.argmax(moved), moved.shape), (3, 2, 2))
        self.assertEqual(float(moved[3, 2, 2]), 1.)
        self.assertTrue(set(np.unique(moved_mask)).issubset({0., 1.}))
        self.assertEqual(int(moved_mask.sum()), int(mask.sum()))

    def test_gradient_validation_rejects_count_missing_b0_and_degenerate_six_directions(self):
        directions = np.array([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        bvals = np.array([0., 1000., 1000., 1000., 1000., 1000., 1000.])
        with tempfile.TemporaryDirectory() as temporary:
            bval_path = Path(temporary) / "sub.bval"
            bvec_path = Path(temporary) / "sub.bvec"

            np.savetxt(bval_path, bvals)
            np.savetxt(bvec_path, np.vstack(([0., 0., 0.], directions)))
            with self.assertRaisesRegex(ValueError, "counts must match"):
                read_gradients(bval_path, bvec_path, 8, 1000., 100.)

            np.savetxt(bval_path, np.full(7, 1000.))
            with self.assertRaisesRegex(ValueError, "at least one b0"):
                read_gradients(bval_path, bvec_path, 7, 1000., 100.)

            np.savetxt(bval_path, bvals)
            np.savetxt(bvec_path, np.vstack(([0., 0., 0.], np.tile([1., 0., 0.], (6, 1)))))
            with self.assertRaisesRegex(ValueError, "cannot determine a diffusion tensor"):
                read_gradients(bval_path, bvec_path, 7, 1000., 100.)

    def test_gradient_selection_keeps_b0_and_one_shell_only(self):
        directions = np.array([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
        ])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        bvals = np.array([0., 20., 1000., 1000., 1000., 1000., 1000., 1000., 2000.])
        bvecs = np.vstack(([0., 0., 0.], [0., 0., 0.], directions, [1., 0., 0.]))
        with tempfile.TemporaryDirectory() as temporary:
            bval_path = Path(temporary) / "sub.bval"
            bvec_path = Path(temporary) / "sub.bvec"
            np.savetxt(bval_path, bvals)
            np.savetxt(bvec_path, bvecs)
            selected_bvals, selected_bvecs, keep = read_gradients(
                bval_path, bvec_path, len(bvals), 1000., 100.)

        np.testing.assert_array_equal(keep, [True, True, True, True, True, True, True, True, False])
        np.testing.assert_array_equal(selected_bvals, bvals[:-1])
        np.testing.assert_allclose(selected_bvecs[:2], 0.)
        np.testing.assert_allclose(selected_bvecs[2:], directions, atol=1e-7)

    def test_sh6_requires_enough_independent_shell_directions(self):
        six_dti_directions = np.array([
            [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
        ])
        six_dti_directions /= np.linalg.norm(six_dti_directions, axis=1, keepdims=True)
        with self.assertRaisesRegex(ValueError, "at least 28 independent"):
            validate_sh_sampling(np.full(6, 1000.), six_dti_directions, 6)

        repeated = np.tile(six_dti_directions[0], (28, 1))
        with self.assertRaisesRegex(ValueError, "do not span the SH6 feature basis"):
            validate_sh_sampling(np.full(28, 1000.), repeated, 6)

        directions = get_sphere("repulsion724").vertices[:100]
        validate_sh_sampling(np.full(len(directions), 1000.), directions, 6)

    def test_same_grid_data_rejects_affine_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.nii.gz"
            reference = nib.Nifti1Image(np.zeros((4, 5, 6, 7), dtype=np.float32), np.eye(4))
            shifted = np.eye(4)
            shifted[0, 3] = 0.25
            nib.save(nib.Nifti1Image(np.ones((4, 5, 6), dtype=np.float32), shifted), path)
            with self.assertRaisesRegex(ValueError, "same shape and affine"):
                same_grid_data(path, reference, "Brain mask")

    def test_rigid_registration_recovers_translation_and_improves_alignment(self):
        x, y, z = np.mgrid[:32, :32, :32]
        phantom = (
            np.exp(-((x - 9) ** 2 / 30 + (y - 12) ** 2 / 20 + (z - 16) ** 2 / 15))
            + 0.8 * np.exp(-((x - 23) ** 2 / 13 + (y - 20) ** 2 / 28 + (z - 8) ** 2 / 11))
            + 0.5 * np.exp(-((x - 18) ** 2 / 36 + (y - 7) ** 2 / 12 + (z - 25) ** 2 / 8))
        ).astype(np.float32)
        fixed = sitk.GetImageFromArray(phantom.transpose(2, 1, 0))
        fixed_to_moving = sitk.TranslationTransform(3, (2., -3., 1.))
        moving = sitk.Resample(fixed, fixed, fixed_to_moving.GetInverse(),
                               sitk.sitkLinear, 0., sitk.sitkFloat32)

        recovered, _ = register_fa(fixed, moving)
        registered = sitk.Resample(moving, fixed, recovered, sitk.sitkLinear, 0., sitk.sitkFloat32)
        fixed_array = sitk.GetArrayFromImage(fixed)
        before = np.mean((sitk.GetArrayFromImage(moving) - fixed_array) ** 2)
        after = np.mean((sitk.GetArrayFromImage(registered) - fixed_array) ** 2)
        recovered_translation = np.asarray(recovered.GetTranslation())
        np.testing.assert_allclose(recovered_translation, (2., -3., 1.), atol=0.25)
        self.assertLess(after, before * 0.01)


if __name__ == "__main__":
    unittest.main()
