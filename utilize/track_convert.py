import argparse
import os
import nibabel as nib
import numpy as np
import scipy.io
import gzip, os
from nibabel.streamlines.tractogram import Tractogram
from dipy.tracking.streamline import transform_streamlines

import dipy.io.vtk
from dipy.io.dpy import Dpy


def build_argparser():
    DESCRIPTION = "Convert tractograms."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('in_tractograms', metavar='bundle', help='input tractograms.')
    p.add_argument('out_tractograms', metavar='bundle', help='output tractograms.')
    p.add_argument('--lps', action='store_true', default=False, help='whether the output tractogram is in LPS coordinate system.')
    return p


def parse_tt(tinytrack):
    """
    Parse DSI-Studio TinyTrack format (.tt.gz) to extract streamlines and metadata.
    # matrix | destination
    # dimension | A 1-by-3 vector storing the dimension of the containing image.
    # voxel_size | A 1-by-3 vector storing the voxel size in mm.
    # track
    """
    if tinytrack.endswith('.tt.gz'):
        mat_file = tinytrack.replace('.gz', '')
        with gzip.open(tinytrack, 'rb') as f_in, open(mat_file, 'wb') as f_out:
            f_out.write(f_in.read())
            
        mat = scipy.io.loadmat(mat_file)
        os.remove(mat_file)
    else:
        mat = scipy.io.loadmat(tinytrack)
    
    tt_affine = mat['trans_to_mni'].reshape(4,4)
    
    buf1 = mat.get('track').ravel()
    buf2 = buf1.view(np.int8)

    length = len(buf1)
    pos = []
    i = 0
    while i < length:
        pos.append(i)
        track_length = buf1[i:i+4].view(np.uint32)[0]
        i += int(track_length) + 13

    def process_single_track(p):
        size_val = buf1[p:p+4].view(np.uint32)[0]
        
        num_points = int(size_val // 3)
        
        start = buf1[p+4:p+16].view(np.int32)[:3]
        
        track_pts = np.empty((num_points, 3), dtype=np.float32)
        track_pts[0, :] = start.astype(np.float32, copy=False)
        
        if num_points > 1:
            delta = buf2[p+16:p+16 + 3 * (num_points - 1)].astype(np.int32, copy=False).reshape(-1, 3)
            track_pts[1:, :] = (start + delta.cumsum(axis=0)).astype(np.float32, copy=False)

        track_pts /= 32.0
        return track_pts

    streamlines = [process_single_track(p) for p in pos]
    
    streamlines_lps_center = [s - 0.5 for s in streamlines]
    streamlines = transform_streamlines(streamlines_lps_center, tt_affine)

    return streamlines


def load_tract(path, lps=False):
    """
    Load a tractography file and return the streamlines in RASMM space.
    """
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


def save_tract(streamlines, path, lps=False):
    """
    Save streamlines to a tractography file.
    """
    if path.endswith(('.tck', '.trk')):
        new_tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
        nib.streamlines.save(new_tractogram, path)

    elif path.endswith(('.vtk', '.vtp', '.fib')):
        binary = path.endswith(('.vtk', '.fib'))
        dipy.io.vtk.save_vtk_streamlines(streamlines, path, to_lps=lps, binary=binary)

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
    
