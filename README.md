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

### Inference: Fiber Tracking with Pre-trained Model

To perform tractography using a pre-trained model:

```
python tractography.py --config track_config.yaml --track
```

Modify `track_config.yaml` to match your input paths and desired parameters.

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
│   │   └── sub-001_wm_mask.nii.gz│
   └── merged_tract/
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
  

## Contact

For questions, suggestions, or contributions, please create an issue or pull request on GitHub.
