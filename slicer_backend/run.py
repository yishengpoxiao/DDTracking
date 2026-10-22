"""External process entry point; no Slicer Python packages are required."""

import argparse
from datetime import datetime
import importlib
import json
import logging
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DEFAULTS = {
    "num_layers": 2, "num_cells": 512, "cell_type": "GRU", "dropout_prob": 0.5,
    "loss_type": "l1", "sh_order": 6, "use_previous_dire": 0,
    "input_type": "sh", "look_ahead": True,
}


def parser():
    result = argparse.ArgumentParser(description="DDTracking with automatic native FA to MNI registration")
    result.add_argument("--check-dependencies", action="store_true")
    result.add_argument("--preprocess-only", action="store_true", help="Save registration outputs without tracking")
    for name in ("dwi", "bvals", "bvecs", "brain-mask", "wm-mask", "fa", "model", "output-dir", "tracking-config"):
        result.add_argument("--" + name)
    result.add_argument("--template", default=str(ROOT / "resources" / "MNI_FA_template.nii.gz"))
    result.add_argument("--shell", type=float, default=1000)
    result.add_argument("--shell-tolerance", type=float, default=100)
    result.add_argument("--seeds", type=int, default=1)
    result.add_argument("--step-size", type=float, default=0.5)
    result.add_argument("--max-angle", type=float, default=45)
    result.add_argument("--min-length", type=float, default=40)
    result.add_argument("--max-length", type=float, default=200)
    result.add_argument("--batch-size", type=int, default=1024)
    result.add_argument("--device", default="cpu")
    return result


def check_dependencies():
    for name in ("numpy", "scipy", "nibabel", "dipy", "h5py", "yaml", "einops", "SimpleITK", "vtk", "torch"):
        module = importlib.import_module(name)
        print(f"{name}: {getattr(module, '__version__', 'available')}", flush=True)
    importlib.import_module("tracker.tracker")
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)


def validate_arguments(args):
    import math

    required = ["dwi", "bvals", "bvecs", "brain_mask", "template"]
    if not args.preprocess_only:
        required.append("model")
    for name in required:
        value = getattr(args, name)
        if not value or not Path(value).is_file():
            raise ValueError(f"--{name.replace('_', '-')} must name an existing file.")
    for name in ("wm_mask", "fa", "tracking_config"):
        value = getattr(args, name)
        if value and not Path(value).is_file():
            raise ValueError(f"--{name.replace('_', '-')} file does not exist: {value}")
    if not args.output_dir:
        raise ValueError("--output-dir is required.")
    for name in ("shell", "shell_tolerance", "step_size", "max_angle", "min_length", "max_length"):
        if not math.isfinite(getattr(args, name)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if args.shell - args.shell_tolerance <= 50 or args.shell_tolerance < 0:
        raise ValueError("The selected shell interval must be above b=50; tolerance must be nonnegative.")
    if args.seeds < 1 or args.batch_size < 1 or args.step_size <= 0:
        raise ValueError("Seeds, batch size and step size must be positive.")
    if not 0 < args.max_angle <= 180 or not 0 <= args.min_length < args.max_length:
        raise ValueError("Use 0 < angle <= 180 and 0 <= minimum length < maximum length.")


def tracking_parameters(args):
    import torch
    import yaml

    params = dict(MODEL_DEFAULTS)
    if args.tracking_config:
        with open(args.tracking_config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("Tracking configuration must be a YAML mapping.")
        # Paths and UI tracking parameters always come from this run's inputs.
        params.update({key: config[key] for key in MODEL_DEFAULTS if key in config})
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda:<index>.")
    if device.type == "cuda" and (not torch.cuda.is_available() or
            (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"Requested GPU {args.device} is unavailable; select cpu or an available CUDA GPU.")
    params.update({"model_file": str(Path(args.model).resolve()), "num_seeds": args.seeds,
                   "step_size": args.step_size, "max_angle": args.max_angle,
                   "min_length": args.min_length, "max_length": args.max_length,
                   "track_batch_size": args.batch_size, "device": args.device})
    return params


def run(args):
    import yaml
    from slicer_backend.preprocessing import prepare

    validate_arguments(args)
    params = None if args.preprocess_only else tracking_parameters(args)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    directory = Path(tempfile.mkdtemp(prefix=f"DDTracking-{date}-", dir=output))
    logger = logging.getLogger("DDTracking")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logfile = logging.FileHandler(directory / "run.log", encoding="utf-8")
    logfile.setFormatter(formatter)
    logger.handlers[:] = [console, logfile]
    logger.propagate = False
    logger.info("Output directory: %s", directory)
    with (directory / f"inputs-{date}.json").open("x", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2)
    try:
        # Fail on a mismatched checkpoint before spending time registering DWI.
        tracker = None
        if params is not None:
            from tracker.tracker import Tracker
            params.update({"dwi_path": args.dwi, "brain_mask_dir": args.brain_mask,
                           "wm_mask_dir": args.wm_mask, "out_dir": str(directory),
                           "out_name": "MNI_tracks.trk"})
            tracker = Tracker(logger=logger, params=params)
        sh_order = int(params["sh_order"]) if params and params["input_type"] == "sh" else None
        prepared = prepare(args, directory, logger, sh_order=sh_order)
        result = dict(prepared)
        if tracker is not None:
            from slicer_backend.tractograms import export
            tracker.dwi_path = params["dwi_path"] = prepared["dwi"]
            tracker.brain_mask_path = params["brain_mask_dir"] = prepared["brain_mask"]
            tracker.wm_mask_path = params["wm_mask_dir"] = prepared["wm_mask"]
            with (directory / f"tracking-{date}.yaml").open("x", encoding="utf-8") as stream:
                yaml.safe_dump(params, stream, sort_keys=False)
            logger.info("Tracking in MNI space on %s", args.device)
            tracker.track()
            logger.info("Mapping fibers to native RAS coordinates and exporting VTP/TRK")
            result.update(export(directory / "MNI_tracks.trk", args.dwi,
                                 prepared["mni_to_native_ras"], directory))
            if result["streamline_count"] == 0:
                logger.warning("No streamlines passed the length/mask criteria; inspect registration and tracking parameters.")
        result["output_dir"] = str(directory)
        result["preprocess_only"] = args.preprocess_only
        result_path = directory / "result.json"
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        logger.info("Completed. Inspect registered FA against the template before interpreting tracking.")
        print("DDTRACKING_RESULT=" + str(result_path), flush=True)
        return result_path
    except Exception:
        logger.exception("Pipeline failed; partial outputs are preserved in %s", directory)
        raise
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def main():
    args = parser().parse_args()
    try:
        if args.check_dependencies:
            check_dependencies()
        else:
            run(args)
    except Exception as error:
        print(f"DDTracking failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
