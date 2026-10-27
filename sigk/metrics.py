"""Metryki pełnoreferencyjne RGB, bez przycinania brzegów."""

import math
import numpy as np


def errors(reference, prediction, sne_definition=None):
    if reference.shape != prediction.shape:
        raise ValueError("Metryki wymagają jednakowych wymiarów.")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Obrazy zawierają NaN lub Inf.")
    delta = reference.astype(np.float64) - prediction.astype(np.float64)
    sse = float(np.square(delta).sum())
    mse = sse / reference.size
    result = {"MSE": mse, "SSE": sse,
              "PSNR": float("inf") if mse == 0 else -10 * math.log10(mse)}
    if sne_definition == "sse":
        result["SNE"] = sse
    elif sne_definition == "mse":
        result["SNE"] = mse
    elif sne_definition == "relative-squared-error":
        energy = float(np.square(reference.astype(np.float64)).sum())
        result["SNE"] = sse / energy if energy else (0.0 if sse == 0 else float("inf"))
    elif sne_definition is not None:
        raise ValueError(f"Nieznana definicja SNE: {sne_definition}")
    return result


class Metrics:
    def __init__(self, device="cpu", lpips_net="squeeze", skip_lpips=False, sne_definition=None):
        self.device = device
        self.sne_definition = sne_definition
        self.perceptual = None
        if not skip_lpips:
            import lpips
            # Nigdy nie używamy losowych wag jako zamiennika LPIPS.
            self.perceptual = lpips.LPIPS(net=lpips_net, version="0.1", verbose=False).to(device).eval()
            self.perceptual.requires_grad_(False)

    def __call__(self, reference, prediction):
        from skimage.metrics import structural_similarity

        result = errors(reference, prediction, self.sne_definition)
        result["SSIM"] = float(structural_similarity(
            reference, prediction, channel_axis=-1, data_range=1.0,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        result["LPIPS"] = None
        if self.perceptual is not None:
            import torch
            def tensor(array):
                return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()[None].to(self.device) * 2 - 1
            with torch.inference_mode():
                result["LPIPS"] = float(self.perceptual(tensor(reference), tensor(prediction)).item())
        return result
