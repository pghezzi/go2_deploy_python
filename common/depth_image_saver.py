"""Save random samples of network-ready depth without blocking the publisher."""

import queue
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class DepthImageSaver:
    def __init__(self, directory, probability=0.1):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("image_save_probability must be between 0 and 1.")
        self.probability = probability
        self.directory = Path(directory) / f"session_{time.time_ns()}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._random = random.Random()
        self._queue = queue.Queue(maxsize=2)
        self._failed = False
        self._worker = threading.Thread(target=self._save_images, daemon=True)
        self._worker.start()
        print(f"Saving {probability:.0%} of processed depth frames to {self.directory}", flush=True)

    def sample(self, depth):
        if self._failed or self._random.random() >= self.probability:
            return
        # Copy the exact processed values before the caller can reuse its array.
        # A full queue drops a sample instead of delaying the next camera frame.
        try:
            self._queue.put_nowait((time.time_ns(), depth.copy()))
        except queue.Full:
            pass

    def _save_images(self):
        while True:
            sample = self._queue.get()
            try:
                if sample is None:
                    return
                if self._failed:
                    continue
                timestamp, depth = sample
                stem = self.directory / f"depth_{timestamp}"
                # Fixed scale across all frames: near=black, far=white.
                # Keep the original float array alongside the viewable PNG.
                pixels = np.rint(np.clip(depth, 0.0, 1.0) * 255).astype(np.uint8)
                if not cv2.imwrite(str(stem.with_suffix('.png')), pixels):
                    raise OSError("PNG writer returned failure")
                np.save(stem.with_suffix('.npy'), depth, allow_pickle=False)
            except Exception as error:
                self._failed = True
                print(f"Depth image saving disabled after write error: {error}", flush=True)
            finally:
                self._queue.task_done()

    def close(self):
        # Permit a short drain on a normal shutdown; never wait indefinitely
        # for a stalled filesystem during a hardware test.
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            return
        self._worker.join(timeout=1.0)
