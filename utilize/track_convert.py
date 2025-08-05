import argparse
import os
import nibabel as nib
import numpy as np
from nibabel.streamlines.tractogram import Tractogram

import dipy.io.vtk
from dipy.io.dpy import Dpy


def build_argparser():
    DESCRIPTION = "Convert tractograms."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('in_tractograms', metavar='bundle', help='input tractograms.')
    p.add_argument('out_tractograms', metavar='bundle', help='output tractograms.')
    p.add_argument('--lps', action='store_true', default=False, help='whether the output tractogram is in LPS coordinate system.')
    return p

def load_streamlines_with_reference(filepath, lps=False):
    _, ext = os.path.splitext(filepath)
    if ext in ['.tck', '.trk']:
        tractogram = nib.streamlines.load(filepath)

        lines = tractogram.streamlines

    elif ext in ['.vtk', '.vtp', '.fib']:
        lines = dipy.io.vtk.load_vtk_streamlines(filepath, to_lps=lps)

    elif ext in ['.dpy']:
        dpy_obj = Dpy(filepath, mode='r')
        lines = list(dpy_obj.read_tracks())
        dpy_obj.close()

    else:
        raise ValueError('{} is an unsupported file format'.format(filepath))

    return lines


def save_tractogram(lines, filename, lps=False):
    _, ext = os.path.splitext(filename)
    if ext in ['.tck', '.trk']:
        new_tractogram = Tractogram(lines, affine_to_rasmm=np.eye(4))
        nib.streamlines.save(new_tractogram, filename)

    elif ext in ['.vtk', '.vtp', '.fib']:
        binary = ext in ['.vtk', '.fib']
        dipy.io.vtk.save_vtk_streamlines(lines, filename, to_lps=lps, binary=binary)
    elif ext in ['.dpy']:
        dpy_obj = Dpy(filename, mode='w')
        dpy_obj.write_tracks(lines)
        dpy_obj.close()

    else:
        raise ValueError('{} is an unsupported file format'.format(filename))

if __name__ == '__main__':
    parser = build_argparser()

    args = parser.parse_args()

    lines = load_streamlines_with_reference(args.in_tractograms, lps=args.lps)

    # Save the tractogram to the output file
    save_tractogram(lines, args.out_tractograms, lps=args.lps)

    
