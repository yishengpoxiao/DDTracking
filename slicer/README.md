# Using DDTracking in 3D Slicer

This repository includes a 3D Slicer module that starts fiber tracking from the Slicer interface. Slicer provides the interface and loads results in native space. Registration, feature preparation, and model inference run in a separate Python environment, so the model dependencies do not need to be installed into Slicer's bundled Python.

## Compatibility and source installation

DDTracking is a Python scripted module. It has no C++ extension that must be compiled for a particular Slicer version, and the code does not check for version 5.2.2 specifically. The compatibility target is Slicer **4.11 and 5.x** builds that use **Python 3 and Qt 5** and provide the `ScriptedLoadableModule`, Qt, and MRML/VTK APIs used by the module. SimpleITK, DIPY, and PyTorch run in the separate backend and do not depend on Slicer's bundled Python version. Python 2-era Slicer releases are unsupported; builds that have not been released yet or use a different major Python/Qt version need to be checked separately.

The following official Linux builds have been tested: **4.11.20210226, 5.2.2, and 5.12.4**. See the test matrix below; other releases and platforms have not all been checked. Installation follows Slicer's source-module workflow: add the directory containing the module `.py` file to Additional module paths. See the official [Slicer 4.11 settings guide](https://slicer.readthedocs.io/en/v4.11/user_guide/settings.html), [current settings guide](https://slicer.readthedocs.io/en/latest/user_guide/settings.html), and [Python FAQ](https://slicer.readthedocs.io/en/5.6/developer_guide/python_faq.html).

This guide describes running the module from source. It is not an Extensions Manager package and does not require building a binary with the Slicer SDK. Keep the repository layout intact so the module can locate `slicer_backend/run.py` and `resources/MNI_FA_template.nii.gz` relative to `slicer/DDTracking`.

## Requirements

- An installed 3D Slicer within the compatibility target above, with access to the `slicer/DDTracking` module directory.
- A separate **Python 3.10** environment for the backend, with the dependencies in [`../requirements-slicer.txt`](../requirements-slicer.txt). This Python version is recommended for the backend; Slicer's bundled Python does not need to be changed.
- **PyTorch 2.4 or later**, installed separately. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose a CPU wheel or a CUDA wheel appropriate for your operating system and CUDA setup. The selected build must run in the backend environment. PyTorch is not required inside Slicer.
- A DDTracking checkpoint that matches your data. Model weights are available from the [GitHub v0.1 release](https://github.com/yishengpoxiao/DDTracking/releases/tag/v0.1) (archive approximately 154 MB). Download and extract the checkpoint, then select it in the module. Preserve the checkpoint's source and applicable terms.

Clone or download the complete repository from GitHub:

```bash
git clone https://github.com/yishengpoxiao/DDTracking.git
cd DDTracking
```

Create the backend environment from the repository root:

```bash
conda create -n ddtracking-slicer python=3.10 pip -y
conda activate ddtracking-slicer
python -m pip install -r requirements-slicer.txt
# Install the CPU or CUDA build of PyTorch using the official selector above.
python slicer_backend/run.py --check-dependencies
python -c "import sys; print(sys.executable)"
```

PyTorch is not listed in the requirements file because the right package depends on the device. You do not need the full `requirements.yaml` to run the Slicer module; it contains packages for training and other uses as well.

## Install the module in Slicer

1. Clone or download this repository and keep it in a location that remains available while Slicer is running.
2. In Slicer, open **Edit → Application Settings → Modules → Additional module paths**. Depending on the operating system, the settings entry may be under the application menu or named Preferences. Add the **module directory** shown below:

   | Operating system | Directory to add to Additional module paths |
   | --- | --- |
   | Windows | `C:\path\to\DDTracking\slicer\DDTracking` |
   | Linux | `/path/to/DDTracking/slicer/DDTracking` |
   | macOS | `/path/to/DDTracking/slicer/DDTracking` |

   Add the `slicer/DDTracking` folder that directly contains `DDTracking.py`—**not** the repository root and not its `slicer` parent folder. Slicer scans additional module directories at startup; see the official settings guide for your version.
3. Restart Slicer and open **Diffusion → DDTracking** from the module selector. After upgrading or installing another Slicer version, check that version's module-path settings and add the same directory if the setting was not carried over.
4. Under **External Python**, select the backend environment's Python **executable**, not the environment directory. A Conda environment is usually `<env>/bin/python` on Linux/macOS and `<env>\python.exe` on Windows. If you are unsure of the exact path, activate the backend environment and run `python -c "import sys; print(sys.executable)"`, then enter the printed path. Do not use `sys.executable` from Slicer's Python console: that points to Slicer's bundled Python. Slicer and the backend use separate Python environments, so PyTorch, DIPY, and the backend requirements do not need to be installed in Slicer.
5. Select the required input files and an output directory that already exists. We recommend clicking **Register FA only** first and checking the registered FA against the template. After confirming registration QC, select the matching model checkpoint and click **Run tracking**. FA-only mode does not require a checkpoint.

The source tree should remain in this layout:

```text
DDTracking/                 # Keep the complete repository layout
├── resources/              # Default MNI template
├── slicer_backend/         # External Python backend
└── slicer/
    └── DDTracking/         # Add this directory to Slicer's module paths
        └── DDTracking.py
```

The module launches the backend asynchronously, so Slicer remains responsive during a run. The log area displays backend output and errors; **Cancel** requests that the process stop. On success, the module loads the native-space fibers into the current scene. Images and transform files are saved in the output directory; enable the QC option to also load the registered FA and MNI-space fibers.

Paths containing spaces are supported. Use the full path in the file chooser and **do not** add quotation marks around it. The module uses Qt `QProcess` to pass the program and arguments separately. If the backend cannot be found, make sure you did not copy only the `slicer/DDTracking` subdirectory and that `slicer_backend/run.py` and `resources/MNI_FA_template.nii.gz` are still present at the repository root.

### Troubleshooting installation

- **DDTracking is missing from the module list:** Check that you added `slicer/DDTracking`, the directory containing `DDTracking.py`, then restart Slicer. In the Modules settings, check whether the module was ignored or failed to load. Slicer's startup log may contain a Python error.
- **The external process will not start or dependency checks fail:** **External Python** must point to an executable in the backend environment. Run `python slicer_backend/run.py --check-dependencies` using that environment. The CPU or CUDA build of PyTorch must be installed in the same environment.
- **The backend or template cannot be found:** Keep the complete repository layout and load the module from that checkout. Do not copy the module files by themselves to a location outside the repository.
- **The module disappears after changing or upgrading Slicer:** Add the module directory in the new Slicer installation's settings and restart it. Source installation requires no version-specific binary, but the target Slicer build still needs local verification.

## Input data

Required inputs are a single-subject DWI NIfTI image (`.nii` or `.nii.gz`), its matching FSL-format `.bval` and `.bvec` text files, and a brain mask in the same native space. The number of b-values and b-vectors must match the number of volumes in the fourth DWI dimension. The b-vector file may have three rows or three columns; its directions must follow the FSL coordinate convention associated with that image. Confirm that the image, gradients, and mask come from the same acquisition and have matching geometry.

A white-matter mask and a precomputed FA image are optional. If FA is not provided, the backend fits a tensor and computes FA from the b=0 volumes and the selected single diffusion shell. If no white-matter mask is supplied, the tracker uses **GA > 0.18** for seeding and **FA > 0.15** as the stopping boundary. If a white-matter mask is supplied, it is used for both seeding and the tracking boundary. A brain mask is required in either case.

The default shell is **1000 s/mm²** with a tolerance of **±100 s/mm²**. For tensor fitting, b=0 volumes default to b≤**50 s/mm²**. The default settings cannot process data without a b=0 volume or diffusion directions in the selected shell. Check the acquisition's b-values before running.

The default model uses SH order 6 and requires at least **28 independent directions** in the selected shell that span the SH feature space. A sufficient count with degenerate sampling directions is also rejected. Tensor fitting for FA requires at least 6 diffusion directions that determine the tensor, so preprocessing-only mode has a lower direction-sampling requirement.

## Default tracking parameters

Defaults are **1 seed per eligible voxel**, a step size of **0.5 voxel**, a maximum turning angle of **45°**, a fiber length range of **40–200 mm**, and **1024 seeds per batch**. The default device is CPU. If the backend environment has a working CUDA build of PyTorch, select `cuda:0` in the device field. Reduce the batch size if GPU memory is insufficient; CPU runs may take longer. These values can be changed in the module before a run.

## Registration and outputs

The default template is `resources/MNI_FA_template.nii.gz` in the repository. The backend uses SimpleITK to estimate a rigid transform from native FA to the template, resamples the DWI and masks into template space, and rotates the diffusion gradients with the same rigid transform. Tracking then runs in template space, and fibers are mapped back to native DWI space for display in Slicer and downstream analysis.

Each run creates a unique `DDTracking-<random suffix>` subdirectory inside the selected output directory. It contains `inputs.json`, `run.log`, `result.json`, native FA, registered FA, and transforms. Tracking runs also write `tracking.yaml` and native- and MNI-space VTP and TRK fibers. VTP uses RAS coordinates and includes a `SPACE=RAS` field; the Slicer module loads its point coordinates directly as RAS. The result manifest includes paths for `native_fa`, `registered_fa`, `template`, and `native_to_mni_transform`; tracking results also include `native_vtp`, `mni_vtp`, `native_trk`, and `mni_trk`. Once results are ready, the backend prints a line of the form `DDTRACKING_RESULT=/path/to/result.json`.

Compare registered FA with the template and inspect native-space fibers before interpreting tracking results. This new rigid-registration workflow is not numerically identical to the repository's original FSL/ANTs preprocessing, and equivalence between the two has not been measured on real data. Registration errors, an incorrect gradient-coordinate convention, or preprocessing that differs from that used for the checkpoint can affect the fibers.

## Backend command line

The graphical interface calls `slicer_backend/run.py`. You can also run the backend from the repository root to troubleshoot:

```bash
python slicer_backend/run.py \
  --dwi /data/sub-01_dwi.nii.gz \
  --bvals /data/sub-01_dwi.bval \
  --bvecs /data/sub-01_dwi.bvec \
  --brain-mask /data/sub-01_brain-mask.nii.gz \
  --model /models/ddtracking.pt \
  --output-dir /data/ddtracking-output \
  --device cpu
```

Optional arguments include `--wm-mask`, `--fa`, `--template`, `--tracking-config`, `--shell`, `--shell-tolerance`, `--seeds`, `--step-size`, `--max-angle`, `--min-length`, `--max-length`, `--batch-size`, and `--device`. Run `python slicer_backend/run.py --help` for the full argument list supported by the current code.

Run `python slicer_backend/run.py --check-dependencies` to check backend dependencies. To compute or load FA and perform registration for QC without tracking, add `--preprocess-only`; this mode does not need a model checkpoint. The official checkpoint uses the default architecture. If you use a checkpoint trained with a different architecture, provide its matching tracking configuration with `--tracking-config`, or its model parameters may not load correctly. The configuration specifies model architecture and input-feature parameters; paths and runtime tracking options come from the command line or Slicer interface.

## Validation status

The following official Linux x86_64 builds passed testing under Xvfb, and each process exited normally. The same plugin source and a separate backend environment were used:

| Slicer | Bundled Python | Qt | VTK | Module discovery and UI test | Synthetic tracking with real backend |
| --- | --- | --- | --- | --- | --- |
| 4.11.20210226 | 3.6.7 | 5.15.1 | 8.2.0 | Passed | Passed, including FA-only registration |
| 5.2.2 | 3.9.10 | 5.15.2 | 9.1.0 | Passed | Passed |
| 5.12.4 | 3.12.10 | 5.15.2 | 9.6.2 | Passed | Passed, including FA-only registration |

The UI test covers discovery through Additional module paths, creation of the actual Slicer module interface, paths containing spaces, chunked QProcess result markers, cancel-and-restart behavior, RAS VTP loading, ITK transform inversion, and empty fiber output. The test uses an isolated settings namespace and does not overwrite the user's saved DDTracking paths. The real-backend test computes FA and performs registration from the UI, tracks with the published checkpoint, and loads nonempty fibers into the Slicer scene. The tested external backend used Python 3.9.19 and PyTorch 2.4.1; the new-environment example above uses Python 3.10.

The CPU synthetic end-to-end test uses the published checkpoint, computes FA, performs rigid registration and tracking, exports nonempty native/MNI fibers, and checks fiber lengths and coordinates in both spaces. Rerun it with `--model` pointing to the downloaded checkpoint:

```bash
python tests/synthetic_pipeline.py --model /path/to/weights.pt
python -m unittest discover -s tests -v
```

To run the UI test with a target Slicer build, replace both repository paths below with your absolute paths. On Windows, use `Slicer.exe`; on macOS, the executable is usually `/Applications/Slicer.app/Contents/MacOS/Slicer`.

```bash
/path/to/Slicer --ignore-slicerrc --disable-settings --no-main-window --testing \
  --additional-module-paths /absolute/path/to/DDTracking/slicer/DDTracking \
  --python-script /absolute/path/to/DDTracking/tests/slicer_smoke.py
```

The test needs the standalone Python executable to be available on Slicer's startup `PATH`, or set `DDTRACKING_TEST_PYTHON` to its path. This lightweight test runs a standard-library helper process and does not need a model or backend dependencies. It prints the Slicer/Python/Qt/VTK versions, prints `DDTracking Slicer smoke test passed` on success, and exits with code 0; on failure it exits nonzero. A success message without a normal process exit does not count as a complete pass.

On a headless Linux system with Xvfb installed, run:

```bash
xvfb-run -a /path/to/Slicer --ignore-slicerrc --disable-settings --no-main-window --testing \
  --additional-module-paths /absolute/path/to/DDTracking/slicer/DDTracking \
  --python-script /absolute/path/to/DDTracking/tests/slicer_smoke.py
```

Source installation has been validated. CMake extension build files are provided, but formal packaging with the Slicer SDK and submission to the Extensions Index have not been completed. Building a binary extension package requires the matching Slicer SDK; use source installation to share the module across Slicer versions.

Windows/macOS and other Slicer versions can use the same source-install procedure and run this repository's UI test to record the result for that platform/version. The compatibility target does not mean that every past or future version has been tested.

CUDA, Windows, and macOS have not been tested. Registration/tracking equivalence has not been evaluated on real-subject data.

## License and research use

The repository currently has no license file; public availability does not grant permission to redistribute the source code or model weights. For research use, inspect the outputs and cite the DDTracking paper listed in the main README.
