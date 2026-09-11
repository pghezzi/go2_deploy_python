"""Portable terrain selection for DepthWaQ hardware deployment.

The exported classifier has a uniform TorchScript signature:
``depth, rpy, angular_velocity -> class logits``.  This module applies the
paper's instantaneous, EMA, and Bayes selection modes and maps class labels to
the LoRA slots used by the deployed DepthWaQ actor.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


DEFAULT_LABEL_TO_LORA = {
    "rough": -1, "random_uniform": -1, "pyramid_sloped": -1, "plane": -1,
    "gap": 0, "leap": 0,
    "stairs": 1, "upwards_stairs": 1, "all_stairs": 1,
    "pit": 2, "center_platform": 2, "climb": 2,
}


class TerrainSelector:
    def __init__(self, model_path, *, mode="instantaneous", label_to_lora=None,
                 ema_alpha=0.6, change_patience=1, stable_stay=0.9):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Terrain selector model not found: {self.model_path}")
        manifest_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Terrain selector manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "depthwaq-terrain-selector-v1":
            raise ValueError(f"Unsupported terrain-selector bundle: {manifest_path}")
        self.class_ids = [str(value) for value in self.manifest["class_ids"]]
        self.input_shape = tuple(self.manifest["input_shape"])
        if mode not in ("instantaneous", "ema", "bayes"):
            raise ValueError("terrain selector mode must be instantaneous, ema, or bayes")
        self.mode = mode
        self.ema_alpha = float(ema_alpha)
        self.change_patience = int(change_patience)
        self.stable_stay = float(stable_stay)
        if not 0 < self.ema_alpha <= 1 or self.change_patience < 1 or not 0 < self.stable_stay <= 1:
            raise ValueError("invalid terrain selector filter configuration")
        mapping = dict(DEFAULT_LABEL_TO_LORA)
        mapping.update({str(key).lower(): int(value) for key, value in (label_to_lora or {}).items()})
        missing = [label for label in self.class_ids if label.lower() not in mapping]
        if missing:
            raise ValueError(f"No LoRA mapping configured for classifier labels: {missing}")
        self.label_to_lora = mapping
        self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
        self.reset()

    def reset(self):
        self.ema_logits = None
        self.selected_index = None
        self.pending_index = None
        self.pending_count = 0
        count = len(self.class_ids)
        self.belief = torch.full((count,), 1.0 / count)
        self.transition = torch.full((count, count), (1.0 - self.stable_stay) / max(count - 1, 1))
        self.transition.fill_diagonal_(self.stable_stay)

    def _ema(self, logits):
        if self.ema_logits is None:
            self.ema_logits = logits.clone()
            self.selected_index = int(logits.argmax())
            return self.selected_index
        self.ema_logits = self.ema_alpha * logits + (1.0 - self.ema_alpha) * self.ema_logits
        candidate = int(self.ema_logits.argmax())
        if candidate == self.selected_index:
            self.pending_index, self.pending_count = None, 0
        elif candidate == self.pending_index:
            self.pending_count += 1
        else:
            self.pending_index, self.pending_count = candidate, 1
        if self.pending_count >= self.change_patience:
            self.selected_index, self.pending_index, self.pending_count = candidate, None, 0
        return self.selected_index

    def _bayes(self, logits):
        probabilities = torch.softmax(logits, dim=0)
        predicted = self.belief @ self.transition
        self.belief = predicted * probabilities
        self.belief /= self.belief.sum().clamp_min(1e-8)
        return int(self.belief.argmax())

    @torch.inference_mode()
    def update(self, depth, orientation_rpy, angular_velocity):
        depth = torch.as_tensor(depth, dtype=torch.float32).reshape(1, *self.input_shape)
        rpy = torch.as_tensor(orientation_rpy, dtype=torch.float32).reshape(1, 3)
        omega = torch.as_tensor(angular_velocity, dtype=torch.float32).reshape(1, 3)
        logits = self.model(depth, rpy, omega).reshape(-1)
        if logits.numel() != len(self.class_ids) or not torch.isfinite(logits).all():
            raise RuntimeError("Terrain selector returned invalid logits")
        instant = int(logits.argmax())
        selected = instant if self.mode == "instantaneous" else (self._ema(logits) if self.mode == "ema" else self._bayes(logits))
        label = self.class_ids[selected]
        return {
            "label": label, "lora_index": self.label_to_lora[label.lower()],
            "confidence": float(torch.softmax(logits, dim=0)[selected]),
            "instantaneous_label": self.class_ids[instant],
        }
