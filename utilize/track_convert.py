import argparse
import gzip
import io
import os

import nibabel as nib
import numpy as np
import scipy.io
from dipy.tracking.streamline import transform_streamlines
import dipy.io.vtk
from dipy.io.dpy import Dpy
import vtk


def build_argparser():
    DESCRIPTION = "Convert tractograms."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('in_tractograms', metavar='bundle', help='input tractograms.')
    p.add_argument('out_tractograms', metavar='bundle', help='output tractograms.')
    p.add_argument('--lps', action='store_true', default=False, help='whether the output tractogram is in LPS coordinate system.')
    return p


def parse_tt(tinytrack):
    """Read every DSI Studio TinyTrack block as MNI RAS+ streamlines."""
    tinytrack = os.fspath(tinytrack)
    if tinytrack.endswith('.tt.gz'):
        with gzip.open(tinytrack, 'rb') as handle:
            mat = scipy.io.loadmat(io.BytesIO(handle.read()))
    else:
        mat = scipy.io.loadmat(tinytrack)
    if 'trans_to_mni' not in mat:
        raise ValueError(f"{tinytrack} is not a DSI Studio TinyTrack file")
    tt_affine = np.asarray(mat['trans_to_mni'], dtype=np.float64).reshape(4, 4)
    block_names = sorted(
        (name for name in mat if name == 'track' or (name.startswith('track') and name[5:].isdigit())),
        key=lambda name: 0 if name == 'track' else int(name[5:]) + 1,
    )
    if not block_names:
        raise ValueError(f"{tinytrack} contains no TinyTrack blocks")

    streamlines_lps_center = []
    for name in block_names:
        buffer = np.ascontiguousarray(np.asarray(mat[name]).ravel())
        if buffer.dtype != np.uint8:
            raise ValueError(f"{tinytrack}: {name} must be uint8, got {buffer.dtype}")
        signed_buffer = buffer.view(np.int8)
        position = 0
        while position < len(buffer):
            if position + 16 > len(buffer):
                raise ValueError(f"{tinytrack}: truncated {name} record")
            track_size = int(buffer[position:position + 4].view('<u4')[0])
            if track_size <= 0 or track_size % 3:
                raise ValueError(f"{tinytrack}: invalid {name} record length")
            next_position = position + track_size + 13
            if next_position <= position or next_position > len(buffer):
                raise ValueError(f"{tinytrack}: {name} record exceeds its block")

            point_count = track_size // 3
            start = buffer[position + 4:position + 16].view('<i4')[:3]
            points = np.empty((point_count, 3), dtype=np.float32)
            points[0] = start
            if point_count > 1:
                delta = signed_buffer[position + 16:position + 16 + 3 * (point_count - 1)]
                points[1:] = start + delta.astype(np.int32).reshape(-1, 3).cumsum(axis=0)
            streamlines_lps_center.append(points / 32.0 - 0.5)
            position = next_position

    return transform_streamlines(streamlines_lps_center, tt_affine)


def load_tract(path, lps=False):
    """
    Load a tractography file and return the streamlines in RASMM space.
    """
    path = os.fspath(path)
    if path.endswith(('.tck', '.trk')):
        tractogram = nib.streamlines.load(path)
        streamlines = tractogram.streamlines

    elif path.endswith(('.vtk', '.vtp', '.fib')):
        streamlines = dipy.io.vtk.load_vtk_streamlines(path, to_lps=lps)
        
    elif path.endswith('.dpy'):
        dpy_obj = Dpy(path, mode='r')
        streamlines = list(dpy_obj.read_tracks())
        dpy_obj.close()

    elif '.tt' in os.path.basename(path):
        streamlines = parse_tt(path)
    
    else:
        raise ValueError(f"{path} is an unsupported file format")
    
    return streamlines


def convert_fiber_to_polydata(streamlines):
    """Build PolyData directly from ordered RAS+ streamlines."""
    polydata = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for streamline in streamlines:
        streamline = np.asarray(streamline, dtype=np.float32)
        if streamline.ndim != 2 or streamline.shape[1] != 3 or len(streamline) == 0:
            raise ValueError("Each streamline must have shape (N, 3) with N > 0")
        point_ids = vtk.vtkIdList()
        for point in streamline:
            point_ids.InsertNextId(points.InsertNextPoint(point))
        lines.InsertNextCell(point_ids)
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.Modified()
    return polydata


def write_polydata(polydata, path):
    """Write binary legacy VTK or binary XML VTP PolyData."""
    extension = os.path.splitext(os.fspath(path))[1].lower()
    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetDataModeToBinary()
    else:
        raise ValueError(f"Unsupported PolyData extension: {extension}")
    writer.SetFileName(os.fspath(path))
    writer.SetInputData(polydata)
    if not writer.Write():
        raise OSError(f"Failed to write PolyData: {path}")


def save_tract(streamlines, path, lps=False):
    """
    Save streamlines to a tractography file.
    """
    path = os.fspath(path)
    if path.endswith(('.tck', '.trk')):
        new_tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        nib.streamlines.save(new_tractogram, path)

    elif path.endswith(('.vtk', '.vtp')):
        # ``parse_tt`` returns RAS+ MNI streamlines.  Keep that convention by
        # default; retain the existing --lps option by flipping only on request.
        if lps:
            streamlines = [np.asarray(s) * np.array([-1.0, -1.0, 1.0]) for s in streamlines]
        write_polydata(convert_fiber_to_polydata(streamlines), path)

    elif path.endswith('.fib'):
        dipy.io.vtk.save_vtk_streamlines(streamlines, path, to_lps=lps, binary=True)

    elif path.endswith('.dpy'):
        dpy_obj = Dpy(path, mode='w')
        dpy_obj.write_tracks(streamlines)
        dpy_obj.close()
        
    else:
        raise ValueError(f"{path} is an unsupported file format")

if __name__ == '__main__':
    parser = build_argparser()

    args = parser.parse_args()

    lines = load_tract(args.in_tractograms, lps=args.lps)

    # Save the tractogram to the output file
    save_tract(lines, args.out_tractograms, lps=args.lps)
