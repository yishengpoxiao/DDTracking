import nibabel as nib
import numpy as np
import os
import glob
import dipy
import dipy.io.streamline

from scipy.ndimage import binary_dilation
from dipy.io.vtk import save_vtk_streamlines
import vtk


def segment_tract_use_ROI(input_trk, refer_vol, tract_name, output_dir, tractseg_output_dir, in_mask_ratio_threshold=0.8):
    if not 0 <= in_mask_ratio_threshold <= 1:
        raise ValueError("in_mask_ratio_threshold must be between zero and one.")
    reference = nib.load(refer_vol) if isinstance(refer_vol, (str, os.PathLike)) else refer_vol
    # Invalid bounding boxes are filtered below; rejecting the full file here
    # would prevent excluding just the offending streamline.
    sft = dipy.io.streamline.load_tractogram(input_trk, refer_vol, bbox_valid_check=False)

    sft.to_vox()
    sft.to_center()

    streamline_vox = sft.streamlines
    # step 1: Endpoint ROI
    def load_mask(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required ROI mask does not exist: {path}")
        image = nib.load(path)
        if image.shape != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-4):
            raise ValueError(f"ROI mask must match the reference volume grid: {path}")
        return image.get_fdata() > 0

    endpoint_dir = os.path.join(tractseg_output_dir, 'endings_segmentations')
    endpoint_mask1 = load_mask(os.path.join(endpoint_dir, f'{tract_name}_b.nii.gz'))
    endpoint_mask2 = load_mask(os.path.join(endpoint_dir, f'{tract_name}_e.nii.gz'))

    endpoint_mask1 = binary_dilation(endpoint_mask1, iterations=6)
    endpoint_mask2 = binary_dilation(endpoint_mask2, iterations=6)

    # Extract VB and WPC.
    valid_streamlines = np.asarray([
        len(streamline) > 0 and np.isfinite(streamline).all()
        for streamline in streamline_vox
    ], dtype=bool)
    if not valid_streamlines.any():
        raise ValueError("Input tractogram contains no nonempty streamlines.")
    voxel_beg = np.asarray([s[0] if len(s) else [0, 0, 0] for s in streamline_vox], dtype=np.float64).T
    voxel_end = np.asarray([s[-1] if len(s) else [0, 0, 0] for s in streamline_vox], dtype=np.float64).T

    def nearest_mask_values(mask, points):
        points = np.asarray(points).T
        finite = np.isfinite(points).all(axis=1)
        indices = np.zeros(points.shape, dtype=np.int64)
        indices[finite] = np.floor(points[finite] + 0.5).astype(np.int64)
        inside = finite & (indices >= 0).all(axis=1) & (indices < np.asarray(mask.shape)).all(axis=1)
        values = np.zeros(len(points), dtype=bool)
        valid = indices[inside]
        values[inside] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]
        return values

    map1_beg = nearest_mask_values(endpoint_mask1, voxel_beg)
    map2_beg = nearest_mask_values(endpoint_mask2, voxel_beg)
    map1_end = nearest_mask_values(endpoint_mask1, voxel_end)
    map2_end = nearest_mask_values(endpoint_mask2, voxel_end)

    vs_ids = np.logical_or(
        np.logical_and(map1_beg, map2_end), np.logical_and(map1_end, map2_beg))

    vs_ids = np.arange(len(vs_ids))[vs_ids & valid_streamlines].astype(np.int32)

    # step 2: All mask
    all_mask_path = os.path.join(tractseg_output_dir, 'bundle_segmentations', f'{tract_name}.nii.gz')
    all_mask = load_mask(all_mask_path)

    # dimensions = all_mask.shape
    # all_mask = binary_dilation(all_mask, iterations=1)

    # inv_all_mask = np.zeros(dimensions, dtype=np.uint8)
    # inv_all_mask[all_mask == 0] = 1

    tmp_steamline = streamline_vox[vs_ids]

    # streamlines_case = _streamlines_in_mask(list(tmp_steamline), inv_all_mask, np.eye(3), [0, 0, 0])
    # out_of_mask_ids_in_vs = np.where(streamlines_case == [0, 1][True])[0].tolist()
    # out_of_mask_ids_in_vs = np.asarray(out_of_mask_ids_in_vs, dtype=np.int32)

    # out_of_mask_ids = vs_ids[out_of_mask_ids_in_vs]
    # wpc_ids = out_of_mask_ids

    # vs_ids = np.setdiff1d(vs_ids, wpc_ids)

    keep_ids = []
    for idx, vox_coords in zip(vs_ids, tmp_steamline):
        if not np.isfinite(vox_coords).all():
            continue
        coords_int = np.floor(vox_coords + 0.5).astype(np.int64)
        if np.any(coords_int < 0) or np.any(coords_int >= np.asarray(all_mask.shape)):
            continue
        idxs = (coords_int[:, 0], coords_int[:, 1], coords_int[:, 2])
        inside = all_mask[idxs]

        ratio = inside.sum() / float(len(inside))
        if ratio >= in_mask_ratio_threshold:
            keep_ids.append(idx)

    vs_ids = np.asarray(keep_ids, dtype=np.int32)
    
    vc_sft = sft[vs_ids]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{tract_name}_vc.vtk')
    print(f'saving {tract_name} tract')
    vc_sft.to_rasmm()
    vc_sft.to_center()
    if len(vc_sft.streamlines):
        save_vtk_streamlines(vc_sft.streamlines, output_path, to_lps=False)
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(vtk.vtkPolyData())
        writer.SetFileTypeToBinary()
        if not writer.Write():
            raise IOError(f"Could not write empty tractogram: {output_path}")
    return output_path
