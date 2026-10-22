import argparse
import glob
import os

import numpy as np
import nibabel as nib

from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines.trk import TrkFile

from collections import defaultdict


def merge_trks(trk_dir, keep=1.0, num_keep=None, weighted=False, out_dir=None):
    """
    Merges .trk files found in a directory and optionally subsamples a fraction of the fibers.

    Parameters:
        trk_dir (str): Directory containing .trk files to merge.
        keep (float): Fraction of fibers to keep; if less than 1, subsamples the fibers.
        weighted (bool): If True, perform subsampling weighted by bundle size.
        out_dir (str): Output directory for the merged .trk file. Uses trk_dir if None.

    Notes:
        .trk reference grids must match. .tck streamlines are interpreted in
        world coordinates and need at least one .trk to define output geometry.
    """
    if not os.path.isdir(trk_dir):
        raise FileNotFoundError(f"Tractogram directory does not exist: {trk_dir}")
    if not 0 < keep <= 1:
        raise ValueError("keep must be greater than zero and at most one.")
    if num_keep is not None and (isinstance(num_keep, (bool, np.bool_))
                                 or not isinstance(num_keep, (int, np.integer)) or num_keep <= 0):
        raise ValueError("num_keep must be a positive integer.")
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(trk_dir), "merged_tracts")
    prefix = os.path.basename(os.path.normpath(trk_dir)).split('_')[0]
    save_path = os.path.abspath(os.path.join(
        out_dir, f"{prefix}_whole_tract{'_weighted' if weighted else ''}.trk"))

    bundles = []
    header = None
    
    all_tract = glob.glob(os.path.join(trk_dir, "*.trk")) + glob.glob(os.path.join(trk_dir, "*.tck"))
    all_tract = sorted(path for path in all_tract if os.path.abspath(path) != save_path)

    for bundle_idx, trk_path in enumerate(all_tract):
        print(f"Loading {os.path.basename(trk_path):.<20}", end="\r")
        trk_file = nib.streamlines.load(trk_path)
        if len(trk_file.tractogram.streamlines) == 0:
            continue

        if trk_path.lower().endswith('.trk'):
            if header is None:
                header = trk_file.header
            elif (not np.array_equal(header['dimensions'], trk_file.header['dimensions'])
                  or not np.allclose(header['voxel_to_rasmm'], trk_file.header['voxel_to_rasmm'], atol=1e-4)):
                raise ValueError(f"Tractogram reference geometry differs: {trk_path}")
        bundles.append(trk_file.tractogram)

    if not bundles:
        raise ValueError("No nonempty .trk or .tck files found in the directory.")
    if header is None:
        raise ValueError("At least one nonempty .trk file is required to define output reference geometry.")

    n_fibers = sum([len(b.streamlines) for b in bundles])
    n_bundles = len(bundles)
    print(f"Loaded {n_fibers} fibers from {n_bundles} bundles.")

    # Merging tractograms
    merged_bundles = bundles[0].copy()
    for b in bundles[1:]:
        merged_bundles.extend(b)

    # Optional subsampling
    if keep < 1 or num_keep is not None:
        if weighted:
            p = np.zeros(n_fibers)
            offset=0
            for b in bundles:
                l = len(b.streamlines)
                p[offset:offset+l] = 1 / (l * n_bundles)
                offset += l
        else:
            p = np.ones(n_fibers) / n_fibers
        
        if num_keep is not None:
            keep_n = min(num_keep, n_fibers)
        else:
            keep_n = int(keep * n_fibers)
        print(f"Subsampling {keep_n} fibers.")

        rng = np.random.default_rng(42)
        subsample_indice = rng.choice(
            n_fibers,
            size=keep_n,
            replace=False,
            p=p)

        subsample = merged_bundles.streamlines[subsample_indice]
        tractogram = Tractogram(
                streamlines=subsample,
                affine_to_rasmm=np.eye(4)
            )

    else:
        tractogram = merged_bundles

    os.makedirs(out_dir, exist_ok=True)

    # Saving the merged tractogram
    print(f"Saving {save_path}")

    TrkFile(tractogram, header).save(save_path)
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge trk files.\n"
                                                 "Merge several bundle trks with optional weighted subsampling.\n\n"
                                                 "WARNING: Assumes that each trk file has the same affine.\n\n"
                                                 "HCP whole brain ~ 1.700.000 fibers from 72 bundles\n"
                                                 "keep=0.05 ~ 90k fibers ~ 3.5 Mio segments"
                                                 "n_segments ~ 40 x n_fibers\n")

    parser.add_argument("trk_dir", help="Directory containing trk files.")
    parser.add_argument("--keep", default=1.0, type=float, help="Fraction of fibers to keep during subsampling.")
    parser.add_argument("--num_keep", default=None, type=int, help="Number of fibers to keep during subsampling.")
    parser.add_argument("--weighted", action="store_true", help="Perform subsampling weighted by bundle size.")
    parser.add_argument("--out_dir", help="Directory for saving the merged .trk file.", type=str, default=None)

    args = parser.parse_args()
    merge_trks(trk_dir=args.trk_dir, keep=args.keep, num_keep=args.num_keep, weighted=args.weighted, out_dir=args.out_dir)
