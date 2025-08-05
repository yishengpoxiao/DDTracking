import nibabel as nib
import numpy as np
import os
import glob
import dipy
import dipy.io.streamline
from dipy.tracking.vox2track import _streamlines_in_mask

from scipy.ndimage import map_coordinates, generate_binary_structure, binary_dilation
from dipy.io.vtk import save_vtk_streamlines


def segment_tract_use_ROI(input_trk, refer_vol, tract_name, output_dir, tractseg_output_dir, in_mask_ratio_threshold=0.8):
    sft = dipy.io.streamline.load_tractogram(input_trk, refer_vol)

    sft.to_vox()
    sft.to_corner()

    streamline_vox = sft.streamlines
    # step 1: Endpoint ROI
    mask1_path = glob.glob(os.path.join(tractseg_output_dir, 'endings_segmentations', f'{tract_name}_b.nii.gz'))[0]
    endpoint_mask1 = nib.load(mask1_path).get_fdata().astype(bool)
    
    mask2_path = glob.glob(os.path.join(tractseg_output_dir, 'endings_segmentations', f'{tract_name}_e.nii.gz'))[0]
    endpoint_mask2 = nib.load(mask2_path).get_fdata().astype(bool)

    endpoint_mask1 = binary_dilation(endpoint_mask1, iterations=6)
    endpoint_mask2 = binary_dilation(endpoint_mask2, iterations=6)

    # Extract VB and WPC.
    voxel_beg = np.asarray([s[0] for s in streamline_vox],
                            dtype=np.int16).transpose(1, 0)
    voxel_end = np.asarray([s[-1] for s in streamline_vox],
                            dtype=np.int16).transpose(1, 0)

    map1_beg = map_coordinates(endpoint_mask1, voxel_beg, order=0, mode='nearest')
    map2_beg = map_coordinates(endpoint_mask2, voxel_beg, order=0, mode='nearest')

    map1_end = map_coordinates(endpoint_mask1, voxel_end, order=0, mode='nearest')
    map2_end = map_coordinates(endpoint_mask2, voxel_end, order=0, mode='nearest')

    vs_ids = np.logical_or(
        np.logical_and(map1_beg, map2_end), np.logical_and(map1_end, map2_beg))

    vs_ids = np.arange(len(vs_ids))[vs_ids].astype(np.int32)

    # step 2: All mask
    all_mask_path = os.path.join(tractseg_output_dir, 'bundle_segmentations', f'{tract_name}.nii.gz')
    all_mask = nib.load(all_mask_path).get_fdata().astype(bool)

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
        coords_int = np.round(vox_coords).astype(np.int32)
        idxs = (coords_int[:, 0], coords_int[:, 1], coords_int[:, 2])
        inside = all_mask[idxs]

        ratio = inside.sum() / float(len(inside))
        if ratio >= in_mask_ratio_threshold:
            keep_ids.append(idx)

    vs_ids = np.asarray(keep_ids, dtype=np.int32)
    
    vc_sft = sft[vs_ids]
    try:
        print(f'saving {tract_name} tract')
        vc_sft.to_rasmm()
        save_vtk_streamlines(vc_sft.streamlines, os.path.join(output_dir, f'{tract_name}_vc.vtk'), to_lps=False)
    except Exception as e:
        print(f'Error saving {tract_name} tract: {e}')
    
    return
