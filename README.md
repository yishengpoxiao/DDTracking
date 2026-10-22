# DDTracking: A Deep Generative Framework for Diffusion MRI Tractography

**DDTracking** is a deep learning-based pipeline for diffusion MRI tractography. It covers the complete workflow—from dataset preparation and model training to fiber tracking using pre-trained models.

## Installation

### 1. Python 3 (Miniconda Recommended)

Download and install [Miniconda for Python 3](https://docs.conda.io/en/latest/miniconda.html) (choose 64-bit or 32-bit based on your system):

```
sh Miniconda3-latest-Linux-x86_64.sh -b  # Automatically accepts license agreement
```

Activate the base environment:

```
source ~/miniconda3/bin/activate
```

> You should see `(base)` appear in your terminal prompt.

### 2. Create and Activate Conda Environment

```
conda env create -f requirements-2026-10-22.yaml
conda activate DDTracking
```

Install PyTorch 2.4 or newer for your CPU/CUDA platform using the
[official installation instructions](https://pytorch.org/get-started/locally/).
The fixes below were tested with PyTorch 2.4.1. PyTorch is installed separately
so the environment does not mix incompatible CUDA and CPU packages. The original
`requirements.yaml` is preserved; the dated environment replaces its machine-specific
package dump and invalid pip `pytorch` entry.

---

###### 3. Download Pre-trained Model Weights

Download the model weights from the [GitHub release page](https://github.com/yishengpoxiao/DDTracking/releases/tag/v0.1):

```
cd DDTracking
wget https://github.com/yishengpoxiao/DDTracking/releases/download/v0.1/weights.tar.gz
tar -xzvf weights.tar.gz
```

---

## Usage

### 3D Slicer tracking module

This repository also includes a 3D Slicer module for running DDTracking from a graphical interface. It launches an external Python backend so that Slicer and the model environment can use their own Python installations. See the [Slicer module guide](slicer/README.md) for installation, environment setup, inputs, defaults, and output details. The focused backend dependency list is [requirements-slicer-2026-10-22.txt](requirements-slicer-2026-10-22.txt); install a CPU or CUDA build of PyTorch separately as described there.

The same source module targets **Slicer 4.11 and 5.x with Python 3/Qt 5**; it is not restricted to 5.2.2 and requires no version-specific binary build. Module discovery, the real module interface, external processes, cancellation/restart, and result loading were tested on **Linux Slicer 4.11.20210226, 5.2.2, and 5.12.4**. Other releases and Windows/macOS have not been individually tested. See the [compatibility table and installation guide](slicer/README.md).

1. Clone or download the **whole repository**, preserving its directory layout.
2. Create the external Python environment described in the guide, install backend dependencies and PyTorch, and download the model weights.
3. In Slicer, open **Edit → Application Settings → Modules → Additional module paths** and add `<repository>/slicer/DDTracking`, the folder containing `DDTracking.py`. Restart Slicer and select **Diffusion → DDTracking**. Check this path again when switching Slicer installations.
4. Set **External Python** to the environment's executable. Select DWI, b-values, b-vectors, brain mask and output folder. Run **Register FA only** to inspect alignment, then select the checkpoint and **Run tracking**.

The Slicer workflow estimates or accepts an FA image, rigidly registers it to the repository's MNI FA template, tracks in template space, and maps streamlines back to native DWI space. This registration implementation is not numerically identical to the original FSL and ANTs preprocessing workflow. Validate registration and tractography quality on the data and model you use; equivalence on real data has not been established here.

---

### Preprocessing: Extract B Shell and Rigid Registration to MNI Space

This step prepares your diffusion data for subsequent analysis.

1. **Extract the desired b-shells**
   Use only single shell data. Refer to the `extract_bshell` function in [`utilize/extract_dwi_bshell.py`](https://github.com/yishengpoxiao/DDTracking/blob/main/utilize/extract_dwi_bshell.py).

2. **Rigid registration to MNI space**  
   The extracted diffusion data needs to be rigidly registered to the MNI space.  
   You can use [`utilize/register_volume_to_MNI.py`](https://github.com/yishengpoxiao/DDTracking/blob/main/utilize/register_volume_to_MNI.py) for this step.  
   This process requires **ANTs** and **FSL**.

   Registration is performed between the FA image computed from your DWI and the [MNI FA template](https://github.com/yishengpoxiao/DDTracking/blob/main/resources/MNI_FA_template.nii.gz).  
   A `transform` folder will be created in the DWI directory to store transformation files.

   **Example command:**
   ```
   python utilize/register_volume_to_MNI.py \
       -input_image <dwi_path> \
       -brain_mask <brain_mask_path> \
       -template <MNI_FA_template_path> \
       -wm_mask <wm_mask_path>
   ```

### Inference: Fiber Tracking with Pre-trained Model

To perform tractography using a pre-trained model:

```
python tractography.py --config example_tracking_config-2026-10-22.yaml --track
```

Copy the dated configuration to a new local file and set its input paths before
running. DWI gradients must have exactly the same filename stem, with `.bval`/`.bvals`
and `.bvec`/`.bvecs` extensions. Masks must match the DWI shape and affine. The
original command expects preprocessed, single-shell MNI data; use the Slicer
backend for automatic shell selection and FA registration. Requesting an unavailable
CUDA device now raises an error; choose `cpu` explicitly for CPU tracking.

---

### Training: Train Your Own Model

To train a model using your own dataset, structure your data as follows:

```
dataset/
├── sub-001/
│   ├── dwi/
│   │   ├── sub-001_dwi.nii.gz
│   │   ├── sub-001_dwi.bval
│   │   └── sub-001_dwi.bvec
│   ├── mask/
│   │   ├── sub-001_brain-mask.nii.gz
│   │   └── sub-001_wm_mask.nii.gz
│   └── merged_tract/
│       └── tract_whole_tract.trk
├── sub-002/
│   └── ...
```

Then run distributed training:

```
CUDA_VISIBLE_DEVICES=0,1,... torchrun --nproc_per_node=<NUM_GPUS> tractography.py --config example_training_config-2026-10-22.yaml --train
```

Replace `<NUM_GPUS>` with the number of available GPUs and use a local copy of the
dated training template with your dataset paths. For a single process, run
`python tractography.py --config <your-training-config.yaml> --train`.
`num_workers: 0` is supported for small datasets or CPU debugging.

Training resumes from the latest `<model_name>.pt`, restoring optimizer, scheduler,
mixed-precision scaler, best validation loss and early-stopping state. Resume
requires the current checkpoint format, including `gradscaler`, `best_val_loss`
and `bad_epochs`; earlier training checkpoints are not accepted for resume.
The published inference weights still load without changing their network tensors.

Dataset directories now cache each feature/SH-order/step-size combination separately,
for example `trainset_sh_sh6_step0.5_dataset.h5` with its matching `.npz` file.
An existing HDF5 dataset must be selected by its explicit file path when no matching
cache or subject directories are available. Regenerate older training caches to use
the corrected endpoint-inclusive streamline resampling: a 3 mm line at 1 mm spacing
now has four points. Splitting a dataset no longer removes arrays from the source,
and numeric HDF5 volume ordering preserves the correct subject mapping beyond ten
subjects.

## Regression checks

```bash
python -m unittest discover -s tests -v
python tests/synthetic_training.py
python tests/distributed_validation.py
python tests/synthetic_pipeline.py --model weights/DDTracking.pt
```

These exercise data pairing and geometry, normalization and interpolation, masked
losses, reverse tracking, registration transforms, distributed sampling, checkpoint
resume, actual-network CPU training, two-process CPU/Gloo validation and early
stopping, and tracking with the published model. CUDA multi-GPU training and the
legacy FSL/ANTs command-line registration were not run in this validation; the
Slicer backend's SimpleITK registration was exercised end to end.
The voxel-center mask and reverse-turn corrections can change tracked fibers;
the resampling and loss corrections can change newly trained models. Validate
these changes on your own data before comparing scientific results.

## Acknowledgments

This work is supported by:

- **National Key R&D Program of China** (No. 2023YFE0118600)
  
- **National Natural Science Foundation of China** (No. 62371107)

## Citation

If you find **DDTracking** useful in your research, please cite our paper:

**DDTracking: A diffusion model-based deep generative framework with local-global spatiotemporal modeling for diffusion MRI tractography**
*Medical Image Analysis, 2026*

```bibtex
@article{LI2026103967,
  title   = {DDTracking: A diffusion model-based deep generative framework with local-global spatiotemporal modeling for diffusion MRI tractography},
  journal = {Medical Image Analysis},
  volume  = {110},
  pages   = {103967},
  year    = {2026},
  issn    = {1361-8415},
  doi     = {https://doi.org/10.1016/j.media.2026.103967},
  url     = {https://www.sciencedirect.com/science/article/pii/S1361841526000368},
  author  = {Yijie Li and Wei Zhang and Xi Zhu and Ye Wu and Yogesh Rathi and Lauren J. O'Donnell and Fan Zhang},
  keywords = {Tractography, Deep learning, Diffusion model, Diffusion MRI}
}
```

## Contact

For questions, suggestions, or contributions, please create an issue or pull request on GitHub.
