import numpy as np


def volume_iterator(img, blocksize=1, start=0, end=0):
    if len(img.shape) != 4:
        raise ValueError("DWI image must be four-dimensional.")
    nb_volumes = img.shape[-1]
    end = end if end else img.shape[-1]
    if not (0 <= start <= end <= nb_volumes):
        raise ValueError("Volume range must be within the DWI fourth dimension.")
    if not isinstance(blocksize, int) or blocksize <= 0:
        raise ValueError("blocksize must be a positive integer.")
    for first in range(start, end, blocksize):
        stop = min(first + blocksize, end)
        yield list(range(first, stop)), np.asarray(img.dataobj[..., first:stop], dtype=np.float32)


def extract_bshell(dwi, bvals, bvecs, bvals_to_extract, tol=50, block_size=None):
    if len(dwi.shape) != 4:
        raise ValueError("DWI image must be four-dimensional.")
    bvals = np.asarray(bvals).reshape(-1)
    bvecs = np.asarray(bvecs)
    if bvecs.ndim != 2:
        raise ValueError("b-vectors must be a 2D array.")
    if bvecs.shape == (3, dwi.shape[-1]):
        bvecs = bvecs.T
    if bvals.shape != (dwi.shape[-1],) or bvecs.shape != (dwi.shape[-1], 3):
        raise ValueError("b-value and b-vector counts must match the DWI fourth dimension.")
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
        raise ValueError("b-values and b-vectors must be finite.")
    shells = list(bvals_to_extract)
    if not shells:
        raise ValueError("At least one b-value shell must be requested.")
    if tol < 0:
        raise ValueError("Shell tolerance must be nonnegative.")
    if block_size is None:
        block_size = dwi.shape[-1]
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    indices = np.flatnonzero(np.logical_or.reduce([
        np.abs(bvals - shell) <= tol for shell in shells]))
    if len(indices) == 0:
        raise ValueError("There are no volumes that have the supplied b-values"
                         ": {}".format(bvals_to_extract))
    
    shell_data = np.empty(dwi.shape[:-1] + (len(indices),), dtype=np.float32)
    for vi, data in volume_iterator(dwi, block_size):
        selected_output = np.flatnonzero((indices >= vi[0]) & (indices <= vi[-1])) if vi else []
        if len(selected_output):
            source = indices[selected_output] - vi[0]
            shell_data[..., selected_output] = data[..., source]

    output_bvals = bvals[indices].astype(int)
    output_bvecs = bvecs[indices, :]

    return indices, shell_data, output_bvals, output_bvecs
