"""Exercise the whole CPU pipeline with published weights and synthetic DWI.

Run: python tests/synthetic_pipeline.py --model /path/to/weights/DDTracking.pt
This checks software integration and coordinates, not anatomical validity.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nibabel as nib
import numpy as np
import torch
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.reconst.dti import TensorModel

from slicer_backend.run import parser, run


def create_inputs(directory):
    shape = (24, 24, 24)
    x, y, z = np.indices(shape).astype(float)
    brain = ((x - 11.5) / 10)**2 + ((y - 11.5) / 9)**2 + ((z - 11.5) / 8)**2 < 1
    anisotropy = (.2 + .7 * np.exp(-((x - 9)**2 / 35 + (y - 12)**2 / 22 + (z - 10)**2 / 28))
                  + .1 * np.exp(-((x - 16)**2 + (y - 9)**2 + (z - 13)**2) / 12))
    eigenvalues = np.zeros(shape + (3,))
    eigenvalues[..., 0] = .0015
    eigenvalues[..., 1] = eigenvalues[..., 2] = .0015 * (1 - anisotropy)
    bvecs = np.vstack([np.zeros((2, 3)), get_sphere("repulsion100").vertices[:40]])
    bvals = np.r_[0., 0., np.ones(40) * 1000]
    attenuation = np.einsum("...i,ni->...n", eigenvalues, bvecs * bvecs)
    data = (1000 * np.exp(-attenuation * bvals) * brain[..., None]).astype("float32")
    fa = TensorModel(gradient_table(bvals, bvecs=bvecs)).fit(data, mask=brain).fa.astype("float32")
    fixed_affine = np.diag([-1.25, 1.25, 1.25, 1.])
    fixed_affine[:3, 3] = [14.375, -14.375, -14.375]
    native_affine = fixed_affine.copy()
    native_affine[:3, 3] += [2.5, -1.25, 1.25]
    wm = np.zeros(shape, dtype="uint8")
    wm[11:13, 11:13, 11:13] = 1
    for name, array, affine in (("dwi", data, native_affine), ("brain", brain.astype("uint8"), native_affine),
                                ("wm", wm, native_affine), ("template", fa, fixed_affine)):
        nib.save(nib.Nifti1Image(array, affine), directory / (name + ".nii.gz"))
    np.savetxt(directory / "dwi.bval", bvals[None, :])
    np.savetxt(directory / "dwi.bvec", bvecs.T)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True)
    options = cli.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="ddtracking-synthetic-"))
    create_inputs(directory)
    np.random.seed(42)
    torch.manual_seed(42)
    args = parser().parse_args([
        "--dwi", str(directory / "dwi.nii.gz"), "--bvals", str(directory / "dwi.bval"),
        "--bvecs", str(directory / "dwi.bvec"), "--brain-mask", str(directory / "brain.nii.gz"),
        "--wm-mask", str(directory / "wm.nii.gz"), "--template", str(directory / "template.nii.gz"),
        "--model", options.model, "--output-dir", str(directory / "output"),
        "--min-length", ".1", "--max-length", "3", "--step-size", ".5",
        "--max-angle", "180", "--batch-size", "32", "--device", "cpu",
    ])
    result_path = run(args)
    result = json.loads(result_path.read_text())
    assert result["streamline_count"] > 0
    native = nib.streamlines.load(result["native_trk"]).tractogram.streamlines
    mni = nib.streamlines.load(result["mni_trk"]).tractogram.streamlines
    matrix = np.asarray(result["mni_to_native_ras"])
    assert len(native) == len(mni) == result["streamline_count"]
    for source, target in zip(mni, native):
        np.testing.assert_allclose(nib.affines.apply_affine(matrix, source), target, atol=1e-4)
        length = np.linalg.norm(np.diff(target, axis=0), axis=1).sum()
        assert .1 - 1e-4 <= length <= 3 + 1e-4
    print(f"Synthetic pipeline passed: {result['streamline_count']} streamlines; outputs in {directory}")


if __name__ == "__main__":
    main()
