"""Write physical RAS tractograms for Slicer and other tractography software."""

from pathlib import Path

import nibabel as nib
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def write_vtp(streamlines, path):
    lengths = np.array([len(line) for line in streamlines], dtype=np.int64)
    points = np.concatenate(streamlines).astype(np.float32) if len(lengths) else np.empty((0, 3), np.float32)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    lines = vtk.vtkCellArray()
    lines.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(points), dtype=np.int64), deep=True))
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)
    coordinate_system = vtk.vtkStringArray()
    coordinate_system.SetName("SPACE")
    coordinate_system.InsertNextValue("RAS")
    polydata.GetFieldData().AddArray(coordinate_system)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.SetDataModeToBinary()
    if not writer.Write():
        raise OSError(f"Cannot write tractogram: {path}")


def write_trk(lines, reference_path, path):
    reference = nib.load(str(reference_path))
    header = nib.streamlines.TrkFile.create_empty_header()
    header["dimensions"] = reference.shape[:3]
    header["voxel_sizes"] = nib.affines.voxel_sizes(reference.affine)
    header["voxel_to_rasmm"] = reference.affine
    header["voxel_order"] = "".join(nib.aff2axcodes(reference.affine)).encode("ascii")
    tractogram = nib.streamlines.Tractogram(lines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(nib.streamlines.TrkFile(tractogram, header), str(path))


def export(mni_trk, native_reference, mni_to_native_ras, directory):
    directory = Path(directory)
    tractogram = nib.streamlines.load(str(mni_trk)).tractogram
    tractogram.to_world()
    mni_lines = tractogram.streamlines
    matrix = np.asarray(mni_to_native_ras)
    native_lines = nib.streamlines.ArraySequence(
        nib.affines.apply_affine(matrix, line) for line in mni_lines)
    native_trk = directory / "native_tracks.trk"
    native_vtp = directory / "native_tracks.vtp"
    mni_vtp = directory / "MNI_tracks.vtp"
    write_trk(native_lines, native_reference, native_trk)
    write_vtp(native_lines, native_vtp)
    write_vtp(mni_lines, mni_vtp)
    return {"native_trk": str(native_trk), "native_vtp": str(native_vtp),
            "mni_trk": str(mni_trk), "mni_vtp": str(mni_vtp),
            "streamline_count": len(mni_lines)}
