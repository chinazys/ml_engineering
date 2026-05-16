"""Grad-CAM implementation for CifarCNN."""

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.model import CifarCNN

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for the last convolutional layer of CifarCNN (features[8])."""

    def __init__(self, model: CifarCNN) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hook_handles: list = []  # type: ignore[type-arg]

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the last conv layer."""
        target_layer = self.model.features[8]

        def forward_hook(module: object, inp: object, out: torch.Tensor) -> None:
            self._activations = out.detach()

        def backward_hook(module: object, grad_in: object, grad_out: Tuple[torch.Tensor, ...]) -> None:
            self._gradients = grad_out[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(
        self, image_chw: np.ndarray, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Generate a Grad-CAM heatmap.

        Args:
            image_chw: Float32 CHW image in [0, 1] (normalized with CIFAR-10 stats).
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            heatmap: HW float32 array in [0, 1].
            predicted_class: Predicted class index.
            probabilities: Softmax probability array, shape (n_classes,).
        """
        self.model.eval()
        self._register_hooks()

        try:
            tensor = torch.from_numpy(image_chw).unsqueeze(0).to(next(self.model.parameters()).device)
            tensor.requires_grad_(False)

            # Forward pass
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            predicted_class = int(np.argmax(probs))

            if class_idx is None:
                class_idx = predicted_class

            # Backward pass for the target class
            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward()

            if self._activations is None or self._gradients is None:
                logger.error("Hooks did not capture activations/gradients")
                return np.zeros((32, 32), dtype=np.float32), predicted_class, probs

            # Pool gradients over spatial dimensions (GAP)
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = F.relu(cam)

            # Upsample to input size
            cam_up = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
            cam_np = cam_up.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            cam_min, cam_max = cam_np.min(), cam_np.max()
            if cam_max - cam_min > 1e-8:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_np = np.zeros_like(cam_np)

            return cam_np, predicted_class, probs

        finally:
            self._remove_hooks()
            self._activations = None
            self._gradients = None


def overlay_heatmap(image_chw: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        image_chw: Float32 CHW image in [0, 1] (raw, not normalized).
        heatmap: Float32 HW array in [0, 1].
        alpha: Blend weight for the heatmap.

    Returns:
        HWC uint8 blended image.
    """
    import matplotlib.cm as cm  # type: ignore[import-untyped]

    # Convert image to HWC uint8
    img_hwc = np.transpose(image_chw, (1, 2, 0))
    img_hwc = np.clip(img_hwc * 255, 0, 255).astype(np.uint8)

    # Apply colormap to heatmap
    colormap = cm.get_cmap("jet")
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    blended = (alpha * heatmap_rgb + (1 - alpha) * img_hwc).astype(np.uint8)
    return blended
