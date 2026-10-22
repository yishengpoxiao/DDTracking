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
conda env create -f requirements.yaml
conda activate DDTracking
```

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
   python register_volume_to_MNI.py \
       -input_image <dwi_path> \
       -brain_mask <brain_mask_path> \
       -template <MNI_FA_template_path> \
       -wm_mask <wm_mask_path>
   ```

### Inference: Fiber Tracking with Pre-trained Model

To perform tractography using a pre-trained model:

```
python tractography.py --config track_config.yaml --track
```

Before running, make sure to modify `track_config.yaml` to match your input paths and desired parameters.

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
CUDA_VISIBLE_DEVICES=0,1,... torchrun --nproc_per_node=<NUM_GPUS> tractography.py --config train_config.yaml --train
```

Replace `<NUM_GPUS>` with the number of available GPUs.

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
