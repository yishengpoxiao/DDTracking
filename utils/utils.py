import glob
import os
import random
import multiprocessing
import itertools

import numpy as np
import scipy.ndimage as ndimage
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh, sh_to_sf, sh_to_sf_matrix, order_from_ncoef, sph_harm_ind_list, normalize_data
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, mask_for_response_ssst, response_from_mask_ssst
from dipy.reconst.mcsd import MSDeconvFit
from dipy.reconst.multi_voxel import MultiVoxelFit
from dipy.utils.optpkg import optional_package

import torch
import torch.nn.functional as F

cvx, have_cvxpy, _ = optional_package("cvxpy")


def add_gaussian_noise(data, noise_level=0.05, prob_per_channel=1.0):
    variance = random.uniform(0, noise_level)
    
    for c in range(data.shape[-1]):
        if np.random.uniform() < prob_per_channel:
            data[..., c] += np.random.normal(0.0, variance, size=data[..., c].shape)
    return data


def zero_mean_unit_variance_normalize(data, mask=None, per_channel=True, epsilon=1e-8):
    if mask is None:
        # If no mask is given, use non-zero data voxels
        mask = np.all(data != 0, axis=-1)
    else:
        # Mask resolution must fit DWI resolution
        assert mask.shape == data.shape[:3], "Normalization mask resolution " \
                                             "does not fit data..."

    # Computing mean and std.
    # Also dealing with extreme cases where std=0. Shouldn't happen. It means
    # that this data is meaningless for your model. Here, we won't divide the
    # data, just move its mean = value in all voxels will now be 0.
    if per_channel:
        # data[mask] becomes a 2D array. Taking axis 0 = the voxels.
        mean = np.mean(data[mask], axis=0)
        std = np.std(data[mask], axis=0)
    else:
        mean = np.mean(data[mask])
        std = np.std(data[mask])

    # If std ~ 0, replace by eps.
    std = np.maximum(std, epsilon)

    standardized_data = (data - mean) / std
    standardized_data[~mask] = 0.0

    return standardized_data
    # for b in range(data.shape[0]):
    #     if per_channel:
    #         for c in range(data.shape[1]):
    #             mean = data[b, c].mean()
    #             std = data[b, c].std() + epsilon
    #             data_normalized[b, c] = (data[b, c] - mean) / std
    #     else:
    #         mean = data[b].mean()
    #         std = data[b].std() + epsilon
    #         data_normalized[b] = (data[b] - mean) / std
    # return data_normalized


def minmax_normalize(samples, mask):
    if mask.ndim == 3:
        mask = np.broadcast_to(mask[..., None], samples.shape)

    out = samples

    sample_mins = np.broadcast_to(np.min(samples, -1)[..., None], samples.shape)
    sample_maxes = np.broadcast_to(np.max(samples, -1)[..., None], samples.shape)
    ranges = sample_maxes - sample_mins

    ranges = np.where(ranges < 1e-6, 1.0, ranges)

    out[mask] = (out[mask] - sample_mins[mask]) / (ranges[mask])
    return out


def calc_mean_dwi(dwi, mask):
    DW_means = np.zeros(dwi.shape[3])
    for i in range(len(DW_means)):
        curr_volume = dwi[:, :, :, i]
        if len(mask) > 0:
            curr_volume = curr_volume[mask > 0]
        else:
            curr_volume = curr_volume[curr_volume > 0]
        DW_means[i] = np.mean(curr_volume)

    return DW_means


def locate_files(path, extension):
    if extension == 'dwi':
        return glob.glob(os.path.join(path, 'dwi', '*.nii*'))[0]
    elif extension == 'tract':
        if glob.glob(os.path.join(path, 'merged_tract', '*.trk')):
            return glob.glob(os.path.join(path, 'merged_tract', '*.trk'))[0]
        if glob.glob(os.path.join(path, 'merged_tract', '*.tck')):
            return glob.glob(os.path.join(path, 'merged_tract', '*.tck'))[0]
    elif extension == 'brain-mask':
        return glob.glob(os.path.join(path, 'mask', '*brain*.nii*'))[0]
    elif extension == 'wm-mask':
        return glob.glob(os.path.join(path, 'mask', '*wm*.nii*'))[0]


def mask_dwi(weights, mask):
    """
    Mask dwi.

    Parameters:
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    mask : ndarray of shape (X, Y, Z) or (X, Y, Z, #gradients)
        Brain mask image.

    Returns
    -------
    ndarray : Masked diffusion weighted images.
    """
    if mask.ndim == 3:
        masked_dwi = weights * np.tile(mask[..., None], (1, 1, 1, weights.shape[-1]))
    else:
        masked_dwi = weights * mask

    return masked_dwi


def normalize_dwi(weights, b0):
    """
    Normalize dwi by the first b0.

    Parameters:
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : ndarray of shape (X, Y, Z)
        B0 image.

    Returns
    -------
    ndarray : Diffusion weights normalized by the B0.
    """
    zeros_mask = b0 == 0

    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    erroneous_voxels = np.any(weights > b0, axis=-1)
    nb_erroneous_voxels = np.sum(erroneous_voxels)
    if nb_erroneous_voxels != 0:
        print("Nb. erroneous voxels: {}".format(nb_erroneous_voxels))
        weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    b0[zeros_mask] = 1e-10
    weights_normed = weights / b0
    weights_normed *= ~zeros_mask[..., None]

    return weights_normed


def get_spherical_harmonics_coefficients(weights, gradient, mask=None, sh_basis='descoteaux07', sh_order=6, smooth=0.006, normalize=True):
    print('dwi', weights.shape)

    bvals = np.asarray(gradient.bvals)
    bvecs = np.asarray(gradient.bvecs)
    b0_mask = gradient.b0s_mask

    b0 = weights[..., b0_mask].mean(axis=-1)

    bvecs = bvecs[np.logical_not(b0_mask)]
    weights = weights[..., np.logical_not(b0_mask)]

    if normalize:
        weights = normalize_dwi(weights, b0)

    sphere = Sphere(xyz=bvecs)

    sh = sf_to_sh(weights, sphere, sh_order, sh_basis, smooth=smooth,
                  legacy=True)
    sh = np.nan_to_num(sh, nan=0.0, posinf=0.0, neginf=0.0)

    print('sh', sh.shape)

    if mask is not None:
        sh = sh * mask[..., None]

    return sh


def resample_dwi(weights, gradient, mask=None, sh_basis='descoteaux07', directions=None, sh_order=6, smooth=0.006, normalize=True):
    data_sh = get_spherical_harmonics_coefficients(weights, gradient, mask=mask, sh_basis=sh_basis, sh_order=sh_order, smooth=smooth, normalize=normalize)

    sphere = get_sphere('repulsion100')
    if directions is not None:
        sphere = Sphere(xyz=directions)
    
    data_resampled = sh_to_sf(data_sh, sphere, sh_order, sh_basis, legacy=True)

    print('resample', data_resampled.shape)

    if mask is not None:
        data_resampled = data_resampled * mask[..., None]

    return data_resampled


def compute_rish(weights, gradient, mask=None, sh_order=6, sh_basis='descoteaux07', smooth=0.006, normalize=True):
    data_sh = get_spherical_harmonics_coefficients(weights, gradient, mask=mask, sh_basis=sh_basis, sh_order=sh_order, smooth=smooth, normalize=normalize)

    degree_ids, order_ids = sph_harm_ind_list(sh_order)

    if mask is not None:
        data_sh = data_sh * mask[..., None]

    n_indices_per_order = np.bincount(order_ids)

    # Get start index of each order (e.g. for order 6 : [0,1,6,15])
    order_positions = np.concatenate([[0],
                                      np.cumsum(n_indices_per_order)])[:-1]

    # Get paired indices for np.add.reduceat, specifying where to reduce.
    # The last index is omitted, it is automatically replaced by len(array)-1
    # (e.g. for order 6 : [0,1, 1,6, 6,15, 15,])
    reduce_indices = np.repeat(order_positions, 2)[1:]

    # Compute the sum of squared coefficients using numpy's `reduceat`
    squared_sh = np.square(data_sh)
    rish = np.add.reduceat(squared_sh, reduce_indices, axis=-1)[..., ::2]

    # Apply mask
    if mask is not None:
        rish *= mask[..., None]

    print('rish', rish.shape)

    return rish



def eval_volume_at_3d_coordinates(volume, coords):
    """
    Evaluates the volume data at the given coordinates using trilinear interpolation.
    Parameters
    ----------
    volume : 3D or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    Returns
    -------
    output : ndarray
        Values from volume.
    """
    if volume.ndim not in {3, 4}:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return ndimage.map_coordinates(volume, coords.T, order=1, mode='nearest')
    else:  # volume.ndim == 4
        values_4d = [ndimage.map_coordinates(volume[..., i], coords.T, order=1, mode='nearest') for i in
                     range(volume.shape[-1])]
        return np.ascontiguousarray(np.array(values_4d).T)


def init_seeds_in_voxel(WM_mask, n_seeds_per_voxel):
    """
INPUT: a binary WM mask - np array of size XxYxZ with "1" in WM voxels, "0" othwerwise
       n_seeds_per_voxel: how many seed points to generate within each voxel

OUTPUT: seed_points - zero-padded np array of size n x 3, holding n random points within the WM_mask
"""
    seed_points = []

    indices = np.array(np.nonzero(WM_mask)).T
    for idx in indices:
        seeds_in_voxel = idx + np.random.uniform(-0.5, 0.5, size=(n_seeds_per_voxel, 3))
        seed_points.extend(seeds_in_voxel)

    seed_points = np.array(seed_points, dtype=np.float32)
    return seed_points


def init_seeds(WM_mask, n_seeds):
    """
INPUT: a binary WM mask - np array of size XxYxZ with "1" in WM voxels, "0" othwerwise
       n_seeds - how many seed points to draw

OUTPUT: seed_points - zero-padded np array of size n x 3, holding n random points within the WM_mask
"""

    # mask_idxs = 0.5 * np.array(np.nonzero(WM_mask)).T
    mask_idxs = np.array(np.nonzero(WM_mask)).T
    seed_points = mask_idxs[random.sample(range(len(mask_idxs)), n_seeds)]
    return seed_points


def mask_dilate(mask, iterations=1):
    """
INPUT: mask - a 3D binary image (np array)
       iterations - number of iterations to dilate the mask

OUTPUT: out_mask - dilated mask using the specified SE
"""
    out_mask = ndimage.binary_dilation(mask, iterations=iterations).astype(mask.dtype)

    return out_mask


def is_within_mask(positions, mask):
    """
INPUT: positions - a Nx3 vector of brain positions
       mask - a 3D binary mask

OUTPUT: is_inside - a Nx1 boolean vector specifying if p=positions[i,:] is inside the mask
"""
    is_inside = np.zeros(positions.shape[0], dtype=bool)
    
    within_bounds = (positions >= 0).all(axis=-1) & (positions < np.array(mask.shape)).all(axis=-1)
    valid_positions = positions[within_bounds]

    is_inside[within_bounds] = mask[valid_positions[:, 0].astype(int), valid_positions[:, 1].astype(int), valid_positions[:, 2].astype(int)]

    return is_inside


def interpolate_volume_in_neighborhood_torch(
        volume_as_tensor, coords_vox_corner, max_edge, neighborhood_vectors_vox=None):
    """
    Params
    ------
    volume_as_tensor: torch.Tensor
        The data: a 4D tensor with shape (D, H, W, F), where F is the number of features.
    coords_vox_corner: torch.Tensor of shape (M, 3)
        A list of points (3D coordinates) in voxel space with origin at the corner.
    max_edge: float
        The maximum edge length for normalizing coordinates.
    neighborhood_vectors_vox: np.ndarray or torch.Tensor of shape (N, 3), optional
        The neighbor offsets to add to each coordinate. Should be in voxel space.

    Returns
    -------
    interpolated_data: torch.Tensor of shape (M, F * N)
        The interpolated data: M points with concatenated neighbor features.
    """

    m_input_points = coords_vox_corner.shape[0]

    if (neighborhood_vectors_vox is not None and
            len(neighborhood_vectors_vox) > 0):

        n_neighb = neighborhood_vectors_vox.shape[0]

        # Expand coordinates and neighborhood vectors
        coords_expanded = coords_vox_corner.unsqueeze(1).repeat(1, n_neighb, 1).reshape(-1, 3)
        vectors_expanded = neighborhood_vectors_vox.repeat(m_input_points, 1)

        # Compute new coordinates with neighborhood offsets
        coords_vox_corner = coords_expanded + vectors_expanded
    else:  # No neighborhood:
        coords_vox_corner = coords_vox_corner

    position_normalized = ((coords_vox_corner / max_edge) * 2.0 - 1.0).flip(-1)
    position_normalized = position_normalized.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    interpolated_data = F.grid_sample(
        volume_as_tensor, position_normalized, align_corners=True, mode='bilinear'
    )
    # Reshape interpolated data to (M, F * N)
    interpolated_data = interpolated_data.squeeze().permute(1, 0)

    return interpolated_data


def _fit_from_model_parallel(args):
    """
    Refering to https://github.com/scilus/scilpy/blob/master/scilpy/reconst/fodf.py
    """
    model = args[0]
    data = args[1]
    chunk_id = args[2]

    sub_fit_array = np.zeros((data.shape[0],), dtype='object')
    for i in range(data.shape[0]):
        if data[i].any():
            try:
                sub_fit_array[i] = model.fit(data[i])
            except cvx.error.SolverError:
                coeff = np.full((len(model.n)), np.NaN)
                sub_fit_array[i] = MSDeconvFit(model, coeff, None)

    return chunk_id, sub_fit_array


def fit_from_model(model, data, mask=None, nbr_processes=None):
    """
    Fit the model to data. Can use parallel processing.
    
    Refering to https://github.com/scilus/scilpy/blob/master/scilpy/reconst/fodf.py

    Parameters
    ----------
    model : a model instance
        It will be used to fit the data.
        e.g: An instance of dipy.reconst.shm.SphHarmFit.
    data : np.ndarray (4d)
        Diffusion data.
    mask : np.ndarray, optional
        If `mask` is provided, only the data inside the mask will be
        used for computations.
    nbr_processes : int, optional
        The number of subprocesses to use.
        Default: multiprocessing.cpu_count()

    Returns
    -------
    fit_array : MultiVoxelFit
        Dipy's MultiVoxelFit, containing the fit.
        It contains an array of fits. Any attributes of its individuals fits
        (of class given by 'model.fit') can be accessed through the
        MultiVoxelFit to get all fits at once.
    """
    data_shape = data.shape
    if mask is None:
        mask = np.sum(data, axis=3).astype(bool)
    else:
        mask_any = np.sum(data, axis=3).astype(bool)
        mask *= mask_any

    nbr_processes = multiprocessing.cpu_count() \
        if nbr_processes is None or nbr_processes <= 0 \
        else nbr_processes

    # Ravel the first 3 dimensions while keeping the 4th intact, like a list of
    # 1D time series voxels. Then separate it in chunks of len(nbr_processes).
    data = data[mask].reshape((np.count_nonzero(mask), data_shape[3]))
    chunks = np.array_split(data, nbr_processes)

    chunk_len = np.cumsum([0] + [len(c) for c in chunks])
    pool = multiprocessing.Pool(nbr_processes)
    results = pool.map(_fit_from_model_parallel,
                       zip(itertools.repeat(model),
                           chunks,
                           np.arange(len(chunks))))
    pool.close()
    pool.join()

    # Re-assemble the chunk together in the original shape.
    fit_array = np.zeros(data_shape[0:3], dtype='object')
    tmp_fit_array = np.zeros((np.count_nonzero(mask)), dtype='object')
    for i, fit in results:
        tmp_fit_array[chunk_len[i]:chunk_len[i+1]] = fit

    fit_array[mask] = tmp_fit_array
    fit_array = MultiVoxelFit(model, fit_array, mask)

    return fit_array


def compute_fodf_ssst(weights, gradient, brain_mask, sh_order=6, n_process=None, fa_thresh=0.7, min_fa_thresh=0.5,
                     min_nvox=300, roi_radii=20, roi_center=None):
    nvox = 0
    while nvox < min_nvox and fa_thresh >= min_fa_thresh - 0.00001:
        mask = mask_for_response_ssst(gradient, weights,
                                      roi_center=roi_center,
                                      roi_radii=roi_radii,
                                      fa_thr=fa_thresh)
        nvox = np.sum(mask)
        response, ratio = response_from_mask_ssst(gradient, weights, mask)
        print("Number of indices is {:d} with threshold of {:.2f}".format(nvox, fa_thresh))

        fa_thresh -= 0.05

    if nvox < min_nvox:
        print(
            "!!!Waring:!!!Could not find at least {:d} voxels with sufficient FA "
            "to estimate the FRF!!".format(min_nvox))

    full_response = np.array([response[0][0], response[0][1],
                              response[0][2], response[1]])
    
    full_response = (full_response[0:3], full_response[3])

    csd_model = ConstrainedSphericalDeconvModel(gradient, full_response, sh_order=sh_order)

    if n_process is None:
        n_process = multiprocessing.cpu_count()

    csd_fit = fit_from_model(csd_model, weights, mask=brain_mask, nbr_processes=n_process)
    return csd_fit
