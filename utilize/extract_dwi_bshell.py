from dipy.core.gradients import get_bval_indices
import numpy as np


def volume_iterator(img, blocksize=1, start=0, end=0):
    assert end <= img.shape[-1], "End limit provided is greater than the " \
                                 "total number of volumes in image"

    nb_volumes = img.shape[-1]
    end = end if end else img.shape[-1]

    if blocksize == nb_volumes:
        yield list(range(start, end)), \
              img.get_fdata(dtype=np.float32)[..., start:end]
    else:
        stop = start
        for i in range(start, end - blocksize, blocksize):
            start, stop = i, i + blocksize
            yield list(range(start, stop)), img.dataobj[..., start:stop]

        if stop < end:
            yield list(range(stop, end)), img.dataobj[..., stop:end]


def extract_bshell(dwi, bvals, bvecs, bvals_to_extract, tol=50, block_size=None):
    indices = [get_bval_indices(bvals, shell, tol=tol)
               for shell in bvals_to_extract]
    indices = np.unique(np.sort(np.hstack(indices)))
    
    if len(indices) == 0:
        raise ValueError("There are no volumes that have the supplied b-values"
                         ": {}".format(bvals_to_extract))
    
    if block_size is None:
        block_size = dwi.shape[-1]
    
    shell_data = np.zeros((dwi.shape[:-1] + (len(indices),)))
    for vi, data in volume_iterator(dwi, block_size):
        in_volume = np.array([i in vi for i in indices])
        in_data = np.array([i in indices for i in vi])
        shell_data[..., in_volume] = data[..., in_data]

    output_bvals = bvals[indices].astype(int)
    output_bvecs = bvecs[indices, :]

    return indices, shell_data, output_bvals, output_bvecs
