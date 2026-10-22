import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import vtk

from slicer_backend.tractograms import export, write_trk, write_vtp


class TractogramExportTests(unittest.TestCase):
    def test_native_and_mni_exports_preserve_ras_points_and_vtp_space_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference = directory / "native.nii.gz"
            affine = np.array([
                [-2., 0., 0., 10.], [0., 2., 0., -8.],
                [0., 0., 2., 4.], [0., 0., 0., 1.],
            ])
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), affine), reference)
            mni_lines = [np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)]
            mni_trk = directory / "MNI_tracks.trk"
            write_trk(mni_lines, reference, mni_trk)
            native_to_mni = np.eye(4)
            native_to_mni[:3, 3] = (10., -4., 2.)
            mni_to_native = np.linalg.inv(native_to_mni)

            paths = export(mni_trk, reference, mni_to_native, directory)
            native_expected = mni_lines[0] + mni_to_native[:3, 3]
            native_trk = nib.streamlines.load(paths["native_trk"]).tractogram
            np.testing.assert_allclose(native_trk.streamlines[0], native_expected, atol=1e-5)
            mni_loaded = nib.streamlines.load(paths["mni_trk"]).tractogram
            np.testing.assert_allclose(mni_loaded.streamlines[0], mni_lines[0], atol=1e-5)

            for key, expected in (("native_vtp", native_expected), ("mni_vtp", mni_lines[0])):
                reader = vtk.vtkXMLPolyDataReader()
                reader.SetFileName(paths[key])
                reader.Update()
                polydata = reader.GetOutput()
                points = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
                np.testing.assert_allclose(points, expected, atol=1e-6)
                self.assertEqual(polydata.GetNumberOfLines(), 1)
                space = polydata.GetFieldData().GetAbstractArray("SPACE")
                self.assertIsNotNone(space)
                self.assertEqual(space.GetValue(0), "RAS")

    def test_empty_tractograms_write_valid_empty_vtp_and_trk(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reference = directory / "reference.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)), reference)
            vtp = directory / "empty.vtp"
            trk = directory / "empty.trk"
            write_vtp([], vtp)
            write_trk([], reference, trk)

            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(str(vtp))
            reader.Update()
            self.assertEqual(reader.GetOutput().GetNumberOfPoints(), 0)
            self.assertEqual(reader.GetOutput().GetNumberOfLines(), 0)
            self.assertEqual(len(nib.streamlines.load(str(trk)).tractogram.streamlines), 0)


if __name__ == "__main__":
    unittest.main()
