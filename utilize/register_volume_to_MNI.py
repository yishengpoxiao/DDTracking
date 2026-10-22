import os, shutil
import sys
import subprocess
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from slicer_backend.preprocessing import rotate_bvecs as rotate_bvecs_ras, transform_ras_matrix


"""
ismrm dataset preprocessing, freesurfer is needed
"""

def read_bvecs(this_fname):
    """Read FSL 3xN or row-wise Nx3 b-vectors as an Nx3 array."""
    values = np.loadtxt(this_fname, delimiter=None)
    values = np.atleast_2d(values)
    if values.shape[0] == 3:
        values = values.T
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"b-vectors must be 3xN or Nx3: {this_fname}")
    if not np.isfinite(values).all():
        raise ValueError(f"b-vectors contain non-finite values: {this_fname}")
    return values


def rotate_bvecs(bvecs_in, transform_path, native_image, template_image, bvecs_out):
    """Reorient FSL b-vectors using the inverse of ANTs' fixed-to-moving transform."""
    bvecs = read_bvecs(bvecs_in)
    native = nib.load(native_image)
    template = nib.load(template_image)
    fixed_to_moving_lps = sitk.ReadTransform(transform_path)
    native_to_mni_ras = np.linalg.inv(transform_ras_matrix(fixed_to_moving_lps))
    rotated = rotate_bvecs_ras(bvecs, native.affine, template.affine, native_to_mni_ras)
    norms = np.linalg.norm(rotated, axis=1)
    nonzero = norms > 0
    rotated[nonzero] /= norms[nonzero, None]
    np.savetxt(bvecs_out, rotated.T, fmt='%1.6f')


def align_to_MNI(input_image, refer_MNI_FA, brain_mask_path, wm_mask_path,
                 bval_path=None, bvec_path=None):
    # calculate FA
    basic_dir = os.path.dirname(input_image)
    
    basic_name = os.path.basename(input_image).replace('.nii.gz', '').replace('.nii', '')

    def gradient_path(value, extension, label):
        if value:
            path = os.path.abspath(value)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} file does not exist: {path}")
            return path
        matches = [os.path.join(basic_dir, basic_name + suffix)
                   for suffix in (f'.{extension}', f'.{extension}s')
                   if os.path.isfile(os.path.join(basic_dir, basic_name + suffix))]
        if len(matches) != 1:
            raise ValueError(f"Expected one matching {label} file for {input_image}, found {len(matches)}; pass it explicitly.")
        return matches[0]

    bval_path = gradient_path(bval_path, 'bval', 'b-values')
    bvec_path = gradient_path(bvec_path, 'bvec', 'b-vectors')
    if not os.path.isfile(input_image) or not os.path.isfile(refer_MNI_FA):
        raise FileNotFoundError("Input DWI and MNI FA template must both exist.")
    if not os.path.isfile(brain_mask_path):
        raise FileNotFoundError(f"Brain mask does not exist: {brain_mask_path}")
    if wm_mask_path and not os.path.isfile(wm_mask_path):
        raise FileNotFoundError(f"WM mask does not exist: {wm_mask_path}")
    
    dti_dir = os.path.join(basic_dir, 'dti')
    os.makedirs(dti_dir, exist_ok=True)
    
    command = ['dtifit', '-k', input_image, '-o', dti_dir + '/dti', '-m', brain_mask_path, '-r', bvec_path, '-b', bval_path]
    subprocess.check_call(command)
    
    FA_path = os.path.join(dti_dir, 'dti_FA.nii.gz')
    
    # align FA to MNI
    transform_dir = os.path.join(basic_dir, 'transform')
    os.makedirs(transform_dir, exist_ok=True)
    
    command = ['antsRegistrationSyNQuick.sh', '-d', '3', '-f', refer_MNI_FA, '-m', FA_path, '-o', os.path.join(transform_dir, 'FA_2_MNI'), '-t', 'r', '-n', '-1']
    subprocess.check_call(command)
    
    command = ['ConvertTransformFile', '3', os.path.join(transform_dir, 'FA_2_MNI0GenericAffine.mat'), os.path.join(transform_dir, 'FA_2_MNI.mat'), '--hm']
    subprocess.check_call(command)
    
    command = ['ConvertTransformFile', '3', os.path.join(transform_dir, 'FA_2_MNI0GenericAffine.mat'), os.path.join(transform_dir, 'FA_2_MNI.tfm')]
    subprocess.check_call(command)
    
    # align Diffusion to MNI
    command = ['antsApplyTransforms', '-d', '3', '-e', '3', '-i', input_image, '-r', refer_MNI_FA, '-o', os.path.join(transform_dir, 'Diffusion_MNI_dwi.nii.gz'), '-t', os.path.join(transform_dir, 'FA_2_MNI0GenericAffine.mat'), '-n', 'BSpline']
    subprocess.check_call(command)    
    
    shutil.copy(bval_path, os.path.join(transform_dir, 'Diffusion_MNI_dwi.bvals'))
    
    rotate_bvecs(bvec_path, os.path.join(transform_dir, 'FA_2_MNI.tfm'),
                 input_image, refer_MNI_FA,
                 os.path.join(transform_dir, 'Diffusion_MNI_dwi.bvecs'))
    
    command = ['antsApplyTransforms', '-d', '3', '-i', brain_mask_path, '-r', refer_MNI_FA, '-o', os.path.join(transform_dir, 'brain_mask_MNI.nii.gz'), '-t', os.path.join(transform_dir, 'FA_2_MNI0GenericAffine.mat'), '-n', 'NearestNeighbor']
    subprocess.check_call(command)
    
    if wm_mask_path is not None:
        command = ['antsApplyTransforms', '-d', '3', '-i', wm_mask_path, '-r', refer_MNI_FA, '-o', os.path.join(transform_dir, 'wm_mask_MNI.nii.gz'), '-t', os.path.join(transform_dir, 'FA_2_MNI0GenericAffine.mat'), '-n', 'NearestNeighbor']
        subprocess.check_call(command)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the input image for tracking')
    parser.add_argument('-input_image', type=str, help='Input image path', required=True)
    parser.add_argument('-brain_mask', type=str, help='Brain mask path', required=True)
    parser.add_argument('-template', type=str, help='Template path', required=True)
    parser.add_argument('-wm_mask', type=str, help='White matter mask path', required=False)
    parser.add_argument('-bvals', type=str, help='Explicit b-values file (otherwise match the DWI filename)')
    parser.add_argument('-bvecs', type=str, help='Explicit b-vectors file (otherwise match the DWI filename)')
    args = parser.parse_args()
    
    align_to_MNI(args.input_image, args.template, args.brain_mask, args.wm_mask, args.bvals, args.bvecs)
