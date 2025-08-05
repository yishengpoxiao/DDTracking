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
        Alignment between trk files is not checked but assumed to be the same.
    """
    bundles = []
    first_file = True
    
    all_tract = glob.glob(os.path.join(trk_dir, "*.trk")) + glob.glob(os.path.join(trk_dir, "*.tck"))
    all_tract = sorted(all_tract)

    for bundle_idx, trk_path in enumerate(all_tract):
        print(f"Loading {os.path.basename(trk_path):.<20}", end="\r")
        trk_file = nib.streamlines.load(trk_path)
        if len(trk_file.tractogram.streamlines) == 0:
            continue

        bundles.append(trk_file.tractogram)

        if first_file:
            prefix = os.path.basename(trk_dir).split('_')[0]
            header = trk_file.header
            first_file = False

    if not bundles:
        print("No .trk files found in the directory.")
        return

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

        np.random.seed(42)
        subsample_indice = np.random.choice(
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

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(trk_dir), "merged_tracts")
    os.makedirs(out_dir, exist_ok=True)

    # Saving the merged tractogram
    save_path = os.path.join(out_dir, f"{prefix}_whole_tract{'_weighted' if weighted else ''}.trk")
    print(f"Saving {save_path}")

    TrkFile(tractogram, header).save(save_path)


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
    assert args.keep >= 0.001, "The keep parameter should be >= 0.001."

    merge_trks(trk_dir=args.trk_dir, keep=args.keep, num_keep=args.num_keep, weighted=args.weighted, out_dir=args.out_dir)
