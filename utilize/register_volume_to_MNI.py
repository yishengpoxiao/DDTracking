import os, glob, shutil
import subprocess
import argparse
import re
import numpy as np
from nibabel.tmpdirs import InTemporaryDirectory


"""
ismrm dataset preprocessing, freesurfer is needed
"""

def parse_fsl_affine(file):
    with open(file) as f:
        lines = f.readlines()
    entries = [l.split() for l in lines]
    entries = [row for row in entries if len(row) > 0]  # remove empty rows
    return np.array(entries).astype(np.float32)


def read_bvecs(this_fname):
    """
    Adapted from dipy.io.read_bvals_bvecs
    """
    with open(this_fname, 'r') as f:
        content = f.read()
    # We replace coma and tab delimiter by space
    with InTemporaryDirectory():
        tmp_fname = "tmp_bvals_bvecs.txt"
        with open(tmp_fname, 'w') as f:
            f.write(re.sub(r'(\t|,)', ' ', content))
        return np.squeeze(np.loadtxt(tmp_fname)).T


def rotate_bvecs(bvecs_in, affine_in, bvecs_out):
    bvecs = read_bvecs(bvecs_in)

    affine = parse_fsl_affine(affine_in)

    # Almost identical code to img_utils.apply_rotation_to_peaks except for order of peak array dims
    affine = affine[:3, :3]

    # Get rotation component of affine transformation
    len = np.linalg.norm(affine, axis=0)
    rotation = np.zeros((3,3))
    rotation[:, 0] = affine[:, 0] / len[0]
    rotation[:, 1] = affine[:, 1] / len[1]
    rotation[:, 2] = affine[:, 2] / len[2]

    # Apply rotation to bvecs
    # check bvecs shape
    if bvecs.shape[0] == 3:
        bvecs = np.array(bvecs)
    else:        
        bvecs = np.array(bvecs).T  # change shape from [nr_vecs, 3] to [3, nr_vecs]
    rotated_bvecs = np.matmul(rotation, bvecs)  # output shape [3, nr_vecs]
    
    # Normalize bvecs
    rotated_bvecs = rotated_bvecs.copy()  # Avoid in-place modification.
    bvecs_norm = np.linalg.norm(rotated_bvecs, axis=0)
    idx = bvecs_norm != 0
    rotated_bvecs[:, idx] /= bvecs_norm[idx]

    np.savetxt(bvecs_out, rotated_bvecs, fmt='%1.6f')


def align_to_MNI(input_image, refer_MNI_FA, brain_mask_path, wm_mask_path):
    # calculate FA
    basic_dir = os.path.dirname(input_image)
    
    basic_name = os.path.basename(input_image).replace('.nii.gz', '').replace('.nii', '')

    # 同时含有basic_name 和 bval的文件
    bval_path = glob.glob(os.path.join(basic_dir, '*'+basic_name+'*bval*'))[0]
    bvec_path = glob.glob(os.path.join(basic_dir, '*'+basic_name+'*bvec*'))[0]
    
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
    
    rotate_bvecs(bvec_path, os.path.join(transform_dir, 'FA_2_MNI.mat'), os.path.join(transform_dir, 'Diffusion_MNI_dwi.bvecs'))
    
    command = ['antsApplyTransforms', '-d', '3', '-i', brain_mask_path, '-r', refer_MNI_FA, '-o', os.path.join(transform_dir, 'brain_mask_MNI.nii.gz'), '-t', os.path.join(transform_dir, 'FA_2_MNI0GenericAffine.mat'), '-n', 'NearestNeighbor']
    subprocess.check_call(command)
    
    if wm_mask_path is not None:
        command = ['antsApplyTransforms', '-d', '3', '-i', wm_mask_path, '-r', refer_MNI_FA, '-o', os.path.join(transform_dir, 'wm_mask_MNI.nii.gz'), '-t', os.path.join(transform_dir, 'FA_2_MNI0GenericAffine.mat'), '-n', 'NearestNeighbor']
        subprocess.check_call(command)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the input image for tracking')
    parser.add_argument('-input_image', type=str, help='Input image path')
    parser.add_argument('-brain_mask', type=str, help='Brain mask path')
    parser.add_argument('-template', type=str, help='Template path')
    parser.add_argument('-wm_mask', type=str, help='White matter mask path', required=False)
    args = parser.parse_args()
    
    align_to_MNI(args.input_image, args.template, args.brain_mask, args.wm_mask)
