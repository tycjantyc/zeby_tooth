"""Public pipeline: CBCT + teeth mask -> NIfTI masks."""

from pathlib import Path
from threading import Lock

import numpy as np
from scipy import ndimage
import torch
from torch.nn import functional as F

from inference.volume import predict_volume, read_volume, require_same_geometry, save_mask

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "weights"


def _load_model(path, device):
    """Infer the architecture/channels from a plain state_dict; load strictly."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model weights: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "embeddings.stem.weight" in state and "decoder.out.conv.weight" in state:
        from models.tooth_fairy_1 import build_hyspark_tooth_fairy_1
        inputs = state["embeddings.stem.weight"].shape[1]
        outputs = state["decoder.out.conv.weight"].shape[0]
        model = build_hyspark_tooth_fairy_1(in_channel=inputs, n_classes=outputs, freeze=False)
    elif "final_conv.weight" in state:
        from models.class_segmentor_model import build_class_segmentor_model
        stem = next(v for k, v in state.items() if k.startswith("encoders.0") and v.ndim == 5)
        inputs, outputs = stem.shape[1], state["final_conv.weight"].shape[0]
        model = build_class_segmentor_model(inchannels=inputs, outchannels=outputs)
    else:
        raise ValueError(f"Unsupported checkpoint (expected a HySpark or ResidualUNetSE3D state_dict): {path}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), inputs, outputs


def _bbox(mask):
    points = np.nonzero(mask)
    return tuple(slice(int(v.min()), int(v.max()) + 1) for v in points)


def _resize(volume, shape, mode):
    tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
    return F.interpolate(tensor, size=shape, mode=mode)[0, 0].numpy()


def _centred_patch(volume, centre, size=128):
    """Return a padded cube and slices for pasting predictions back at edges."""
    start = [int(c) - size // 2 for c in centre]
    source = tuple(slice(max(0, v), min(n, v + size)) for v, n in zip(start, volume.shape))
    target = tuple(slice(s.start - v, s.stop - v) for s, v in zip(source, start))
    patch = np.zeros((size,) * 3, dtype=np.float32)
    patch[target] = volume[source]
    return patch, source, target


class SegmentationPipeline:
    """Reuse one instance per worker; calls on the same instance are serialized.

    Available bundled weights start from a teeth mask, not raw CBCT.
    """

    def __init__(self, weights_dir=DEFAULT_WEIGHTS, *, device="auto",
                 threshold=0.5, stride=48):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='cpu' or install CUDA-enabled PyTorch")
        if not 0 < threshold < 1 or not 1 <= stride <= 96:
            raise ValueError("Expected 0 < threshold < 1 and 1 <= stride <= 96")
        self.weights_dir = Path(weights_dir)
        self.threshold, self.stride = threshold, stride
        self._models = {}
        self._lock = Lock()

    def _model(self, name):
        if name not in self._models:
            paths = {"centroid": "agh_centroid_best_s2.pt",
                     "separator": "dl_separator_best_v2.pt",
                     "canals": "tooth_fairy_3d_best_agh.pt"}
            path = self.weights_dir / paths[name]
            model, inputs, outputs = _load_model(path, self.device)
            expected = {"centroid": (1, 1), "separator": (1, 3), "canals": (1, 1)}
            if name in expected and (inputs, outputs) != expected[name]:
                raise ValueError(f"Incompatible {name} checkpoint: {inputs} inputs, {outputs} outputs")
            self._models[name] = (model, inputs, outputs)
        return self._models[name]

    def _separate_teeth(self, teeth):
        output = np.zeros(teeth.shape, dtype=np.uint16)
        if not teeth.any():
            return output
        model, _, _ = self._model("centroid")
        # Preserve the orientation and resize conventions used to train the centroid model.
        flipped = np.flip(teeth, axis=(0, 2))
        region = _bbox(flipped)
        snippet = _resize(flipped[region], (128,) * 3, "nearest-exact")
        tensor = torch.from_numpy(snippet[None, None]).to(self.device)
        seeds_small = (model(tensor).sigmoid()[0, 0].cpu().numpy() >= self.threshold)
        seeds = np.zeros(teeth.shape, dtype=bool)
        seeds[region] = _resize(seeds_small, flipped[region].shape, "nearest") > 0
        seeds = np.flip(seeds, axis=(0, 2))
        markers, count = ndimage.label(seeds, structure=np.ones((3, 3, 3)))
        if count == 0:
            raise RuntimeError("Centroid model found no teeth in a non-empty mask")
        if count > np.iinfo(np.uint16).max:
            raise RuntimeError("Too many detected tooth instances")
        separator, _, _ = self._model("separator")
        for index, bounds in enumerate(ndimage.find_objects(markers), start=1):
            centre = [(s.start + s.stop - 1) // 2 for s in bounds]
            patch, source, target = _centred_patch(teeth, centre)
            tensor = torch.from_numpy(patch[None, None]).to(self.device)
            labels = separator(tensor).argmax(dim=1)[0].cpu().numpy()
            selected = (labels[target] == 2) & teeth[source]
            output[source][selected] = index
        if not output.any():
            raise RuntimeError("Separator produced an empty instance mask for non-empty teeth")
        return output

    def _segment_canals(self, image, instances):
        output = np.zeros(image.shape, dtype=np.uint8)
        ids = np.unique(instances)
        ids = ids[ids != 0]
        if len(ids) == 0:
            return output
        model, _, _ = self._model("canals")
        for tooth_id in ids:
            mask = instances == tooth_id
            region = _bbox(mask)
            local_mask = mask[region]
            snippet = image[region].astype(np.float32, copy=True)
            values = snippet[local_mask]
            snippet[local_mask] = (values - values.min()) / (np.ptp(values) + 1e-5)
            snippet[~local_mask] = 0
            # Sliding windows cover large teeth along every axis, without truncation.
            scores = predict_volume(model, snippet[None], (96,) * 3, (self.stride,) * 3,
                                    1, device=self.device)
            output[region][(scores[0] >= self.threshold) & local_mask] = 1
        return output

    @torch.inference_mode()
    def run(self, cbct_path, output_dir, *, teeth_mask_path=None, instances_path=None):
        """Write three masks into an empty output directory and return their absolute paths.

        Inputs must be scalar 3D NIfTI files with matching geometry. Instances
        are arbitrary positive integer IDs, not anatomical tooth numbers.
        """
        with self._lock:
            if teeth_mask_path and instances_path:
                raise ValueError("Supply either teeth_mask_path or instances_path")
            if not teeth_mask_path and not instances_path:
                raise ValueError("Supply teeth_mask_path or instances_path; bundled weights start from a teeth mask")
            reference, image = read_volume(cbct_path)
            input_mask = None
            mask_path = instances_path or teeth_mask_path
            if mask_path:
                mask_image, input_mask = read_volume(mask_path)
                require_same_geometry(reference, mask_image)
                if (input_mask < 0).any() or not np.equal(input_mask, np.floor(input_mask)).all():
                    raise ValueError("Mask must contain nonnegative integer labels")
                if instances_path and input_mask.max() > np.iinfo(np.uint16).max:
                    raise ValueError("Instance IDs must fit uint16")
            out = Path(output_dir).resolve()
            out.mkdir(parents=True, exist_ok=True)
            if any(out.iterdir()):
                raise FileExistsError(f"Output directory must be empty: {out}")
            teeth = input_mask > 0
            instances = (input_mask.astype(np.uint16) if instances_path
                         else self._separate_teeth(teeth))
            canals = self._segment_canals(image, instances)
            return {
                "teeth_mask": save_mask(teeth.astype(np.uint8), reference, out / "teeth_mask.nii.gz"),
                "teeth_instances": save_mask(instances, reference, out / "teeth_instances.nii.gz"),
                "canals_mask": save_mask(canals, reference, out / "canals_mask.nii.gz"),
            }
