"""FA registration and diffusion reorientation in explicit physical coordinates."""

from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.reconst.dti import TensorModel, design_matrix
from dipy.reconst.shm import sh_to_sf_matrix


RAS_TO_LPS = np.diag([-1., -1., 1., 1.])


def voxel_axes(affine):
    axes = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
        raise ValueError("Sheared image geometry is unsupported; supply an orthogonal NIfTI grid.")
    return axes


def fsl_to_ras_axes(affine):
    """FSL bvecs use image axes with x flipped for a positive determinant."""
    axes = voxel_axes(affine)
    if np.linalg.det(axes) > 0:
        axes = axes @ np.diag([-1., 1., 1.])
    return axes


def transform_ras_matrix(transform):
    """Convert a linear SimpleITK point transform, including its center, to RAS."""
    origin = np.asarray(transform.TransformPoint((0., 0., 0.)))
    matrix = np.eye(4)
    matrix[:3, 3] = origin
    matrix[:3, :3] = np.column_stack([
        np.asarray(transform.TransformPoint(tuple(axis))) - origin for axis in np.eye(3)
    ])
    return RAS_TO_LPS @ matrix @ RAS_TO_LPS


def rotate_bvecs(bvecs, native_affine, template_affine, native_to_mni_ras):
    rotation = (fsl_to_ras_axes(template_affine).T @ native_to_mni_ras[:3, :3]
                @ fsl_to_ras_axes(native_affine))
    return bvecs @ rotation.T


def to_sitk(array, affine):
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32).transpose(2, 1, 0))
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    image.SetSpacing(tuple(spacing))
    image.SetDirection(tuple((RAS_TO_LPS[:3, :3] @ voxel_axes(affine)).ravel()))
    image.SetOrigin(tuple((RAS_TO_LPS @ affine)[:3, 3]))
    return image


def resample(array, source_affine, reference, fixed_to_moving, mask=False):
    image = sitk.Resample(to_sitk(array, source_affine), reference, fixed_to_moving,
                          sitk.sitkNearestNeighbor if mask else sitk.sitkLinear,
                          0., sitk.sitkFloat32)
    return sitk.GetArrayFromImage(image).transpose(2, 1, 0)


def same_grid_data(path, reference, label):
    image = nib.load(str(path))
    if image.shape != reference.shape[:3] or not np.allclose(image.affine, reference.affine, atol=1e-4):
        raise ValueError(f"{label} must have the same shape and affine as the native DWI.")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"{label} contains non-finite values.")
    return data


def read_gradients(bval_path, bvec_path, count, shell, tolerance):
    bvals = np.loadtxt(bval_path).reshape(-1)
    bvecs = np.loadtxt(bvec_path)
    if bvecs.shape == (3, count):
        bvecs = bvecs.T
    if bvals.shape != (count,) or bvecs.shape != (count, 3):
        raise ValueError("b-value and b-vector counts must match the DWI fourth dimension.")
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all() or (bvals < 0).any():
        raise ValueError("Gradients must be finite and b-values nonnegative.")
    b0 = bvals <= 50
    selected = (~b0) & (np.abs(bvals - shell) <= tolerance)
    if not b0.any() or selected.sum() < 6:
        raise ValueError("Need at least one b0 (b <= 50) and six directions in the selected shell.")
    keep = b0 | selected
    bvals, bvecs = bvals[keep], bvecs[keep].copy()
    norms = np.linalg.norm(bvecs, axis=1)
    if np.any(np.abs(norms[bvals > 50] - 1.) > 0.1):
        raise ValueError("Diffusion b-vectors must be unit vectors (within 0.1).")
    bvecs[bvals > 50] /= norms[bvals > 50, None]
    bvecs[bvals <= 50] = 0
    gradients = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    if np.linalg.matrix_rank(design_matrix(gradients)) < 7:
        raise ValueError("Selected gradient directions cannot determine a diffusion tensor.")
    return bvals, bvecs, keep


def validate_sh_sampling(bvals, bvecs, order):
    if order < 0 or order % 2:
        raise ValueError("SH order must be a nonnegative even integer.")
    directions = bvecs[bvals > 50]
    coefficients = (order + 1) * (order + 2) // 2
    if len(directions) < coefficients:
        raise ValueError(f"SH{order} tracking requires at least {coefficients} independent shell directions.")
    basis = sh_to_sf_matrix(Sphere(xyz=directions), order, basis_type="descoteaux07",
                            legacy=True, return_inv=False)
    if np.linalg.matrix_rank(basis) < coefficients:
        raise ValueError(f"Shell directions do not span the SH{order} feature basis.")


def register_fa(fixed, moving):
    """Return the fixed/template-to-moving/native LPS resampling transform."""
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2, seed=20261022)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1., minStep=0.001, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=True)
    registration.Execute(fixed, moving)
    return initial, {
        "metric": float(registration.GetMetricValue()),
        "stop_condition": registration.GetOptimizerStopConditionDescription(),
    }


def save_nifti(data, affine, path):
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, str(path))
    return str(path)


def prepare(args, directory, logger, sh_order=None):
    directory = Path(directory)
    native = nib.load(args.dwi)
    if len(native.shape) != 4:
        raise ValueError("DWI must be a four-dimensional NIfTI image.")
    voxel_axes(native.affine)
    template = nib.load(args.template)
    if len(template.shape) != 3:
        raise ValueError("FA template must be a three-dimensional NIfTI image.")
    for image in (native, template):
        if image.header.get_xyzt_units()[0] not in ("unknown", "mm"):
            raise ValueError("DWI and template spatial units must be millimetres.")
    # The published model was trained on this LAS-oriented, isotropic voxel frame.
    if not np.allclose(voxel_axes(template.affine), np.diag([-1., 1., 1.]), atol=1e-4):
        raise ValueError("The model requires an LAS-oriented MNI template, as bundled in resources/.")
    zooms = np.linalg.norm(template.affine[:3, :3], axis=0)
    if not np.allclose(zooms, zooms[0], atol=1e-4):
        raise ValueError("The model requires an isotropic MNI template grid.")
    bvals, bvecs, keep = read_gradients(args.bvals, args.bvecs, native.shape[3],
                                      args.shell, args.shell_tolerance)
    if sh_order is not None:
        validate_sh_sampling(bvals, bvecs, sh_order)
    brain = same_grid_data(args.brain_mask, native, "Brain mask") > 0.1
    if not brain.any():
        raise ValueError("Brain mask is empty.")
    wm = None
    if args.wm_mask:
        wm = (same_grid_data(args.wm_mask, native, "WM mask") > 0.1) & brain
        if not wm.any():
            raise ValueError("WM mask has no voxels inside the brain mask.")
    logger.info("Selecting %d b0/shell volumes of %d", keep.sum(), native.shape[3])
    data = native.get_fdata(dtype=np.float32)[..., keep]
    if not np.isfinite(data).all():
        raise ValueError("DWI contains non-finite values.")
    data = np.maximum(data, 0)
    data *= brain[..., None]
    if args.fa:
        logger.info("Using supplied native FA")
        fa = same_grid_data(args.fa, native, "FA")
        if (fa < 0).any() or (fa > 1.001).any():
            raise ValueError("FA values must be between zero and one.")
        fa = fa * brain
    else:
        logger.info("Computing native FA with a weighted least-squares tensor fit")
        gradients = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
        fa = np.nan_to_num(TensorModel(gradients).fit(data, mask=brain).fa).astype(np.float32)
        fa = np.clip(fa, 0, 1)
    template_data = template.get_fdata(dtype=np.float32)
    if not np.isfinite(template_data).all() or not (template_data > 0).any() or not (fa > 0).any():
        raise ValueError("FA and template must contain finite, nonzero tissue values.")
    native_fa = save_nifti(fa, native.affine, directory / "native_FA.nii.gz")
    reference = to_sitk(template_data, template.affine)
    logger.info("Rigidly registering native FA to the MNI FA template")
    fixed_to_moving, quality = register_fa(reference, to_sitk(fa, native.affine))
    mni_to_native_ras = transform_ras_matrix(fixed_to_moving)
    native_to_mni_ras = np.linalg.inv(mni_to_native_ras)
    # ITK files store resampling (output-to-input) transforms. For Slicer's
    # native-to-MNI transform node write fixed-to-moving, which Slicer inverts.
    transform_path = directory / "native_to_mni.tfm"
    sitk.WriteTransform(fixed_to_moving, str(transform_path))
    sitk.WriteTransform(fixed_to_moving.GetInverse(), str(directory / "mni_to_native.tfm"))
    np.savetxt(directory / "native_to_mni_RAS.txt", native_to_mni_ras)
    np.savetxt(directory / "mni_to_native_RAS.txt", mni_to_native_ras)
    registered_fa = save_nifti(resample(fa, native.affine, reference, fixed_to_moving),
                               template.affine, directory / "registered_FA.nii.gz")
    aligned_brain = resample(brain, native.affine, reference, fixed_to_moving, mask=True) > 0.5
    if not aligned_brain.any():
        raise ValueError("Registration placed the brain outside the template; inspect the FA images.")
    quality["brain_voxels_in_template"] = int(aligned_brain.sum())
    logger.info("Resampling DWI, brain mask and WM mask; rotating b-vectors")
    aligned = np.empty(template.shape + (len(bvals),), dtype=np.float32)
    for index in range(len(bvals)):
        aligned[..., index] = resample(data[..., index], native.affine, reference, fixed_to_moving)
    aligned *= aligned_brain[..., None]
    dwi_path = save_nifti(aligned, template.affine, directory / "MNI_dwi.nii.gz")
    np.savetxt(directory / "MNI_dwi.bvals", bvals[None, :], fmt="%.8g")
    rotated = rotate_bvecs(bvecs, native.affine, template.affine, native_to_mni_ras)
    np.savetxt(directory / "MNI_dwi.bvecs", rotated.T, fmt="%.10g")
    brain_path = save_nifti(aligned_brain.astype(np.uint8), template.affine,
                           directory / "MNI_brain_mask.nii.gz")
    wm_path = None
    if wm is not None:
        aligned_wm = (resample(wm, native.affine, reference, fixed_to_moving, mask=True) > 0.5) & aligned_brain
        if not aligned_wm.any():
            raise ValueError("No WM remains inside the registered brain mask.")
        wm_path = save_nifti(aligned_wm.astype(np.uint8), template.affine, directory / "MNI_wm_mask.nii.gz")
    return {"dwi": dwi_path, "brain_mask": brain_path, "wm_mask": wm_path,
            "native_fa": native_fa, "registered_fa": registered_fa,
            "template": str(Path(args.template).resolve()),
            "native_to_mni_transform": str(transform_path),
            "mni_to_native_ras": mni_to_native_ras.tolist(), "registration": quality}
