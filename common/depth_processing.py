"""The parkour RealSense preprocessing, without camera or DDS dependencies."""

import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def preprocess_depth_array(
    raw_depth,
    depth_scale,
    output_shape=(48, 64),
    depth_range_m=(0.0, 3.0),
    cropping=(48, 0, 28, 36),
    rotate_180=False,
):
    """Process an already filtered Z16 frame into normalized float32 [H, W].

    Use the reference's crop endpoints and adaptive average pooling. Rotation
    corrects an inverted camera mount; it is not a property of the policy.
    RealSense depth_scale is meters per raw unit (the reference assumes .001).
    """
    raw_depth = np.asarray(raw_depth)
    if raw_depth.ndim != 2:
        raise ValueError("Expected a two-dimensional depth frame.")
    near, far = depth_range_m
    if not (np.isfinite(depth_scale) and depth_scale > 0):
        raise ValueError("depth_scale must be finite and positive.")
    if not (np.isfinite(near) and np.isfinite(far) and 0 <= near < far):
        raise ValueError("Depth range must satisfy 0 <= near < far.")
    if len(output_shape) != 2 or any(int(v) != v or v <= 0 for v in output_shape):
        raise ValueError("Output shape must contain two positive integers.")
    if len(cropping) != 4 or any(int(v) != v or v < 0 for v in cropping):
        raise ValueError("Cropping must contain four nonnegative integers.")
    top, bottom, left, right = map(int, cropping)
    height, width = raw_depth.shape
    # Literal parkour slices: top:-bottom-1, left:-right-1. A zero bottom
    # crop still removes the last row; do not silently substitute H-bottom.
    height_stop, width_stop = height - bottom - 1, width - right - 1
    if top >= height_stop or left >= width_stop:
        raise ValueError("Cropping removes the entire depth frame.")
    if rotate_180:
        raw_depth = np.rot90(raw_depth, k=2)
    # astype copies the rotated array, removing NumPy's negative strides.
    depth = torch.from_numpy(raw_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    depth = depth[:, :, top:height_stop, left:width_stop]
    # Keep the reference's millimeter arithmetic when scale == .001, while
    # honoring cameras configured with a different RealSense depth unit.
    depth = depth * (depth_scale * 1000.0)
    near_mm, far_mm = near * 1000.0, far * 1000.0
    depth = (depth.clamp(near_mm, far_mm) - near_mm) / (far_mm - near_mm)
    depth = F.adaptive_avg_pool2d(depth, tuple(map(int, output_shape)))
    return depth[0, 0].numpy()
