#!/usr/bin/env python3
"""Display normalized depth frames received on the Unitree RT DDS topic."""

import argparse
import threading

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

from common.depth_image_idl import DepthImage_


TOPIC_DEPTHIMAGE = "rt/depthimage"


class RTDepthImageViewer:
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._lock = threading.Lock()
        self._image = None

        self.subscriber = ChannelSubscriber(TOPIC_DEPTHIMAGE, DepthImage_)
        self.subscriber.Init(self._on_depth_image, 1)
        print(f"Subscribed to {TOPIC_DEPTHIMAGE}; expecting {width}x{height} frames.")

    def _on_depth_image(self, message: DepthImage_) -> None:
        values = np.asarray(message.normalized_value, dtype=np.float32)
        expected_size = self.height * self.width
        if (
            message.height != self.height
            or message.width != self.width
            or values.size != expected_size
        ):
            print(
                "Ignoring depth image with unexpected shape "
                f"{message.width}x{message.height} ({values.size} values); "
                f"expected {self.width}x{self.height} ({expected_size} values)."
            )
            return

        # Copy so DDS can safely reuse the message buffer after this callback.
        image = values.reshape(self.height, self.width).copy()
        print(image, flush=True)
        with self._lock:
            self._image = image

    def run(self) -> None:
        window_name = "RT Depth Image (normalized)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width * 8, self.height * 8)

        while True:
            with self._lock:
                image = None if self._image is None else self._image.copy()

            if image is not None:
                # The publisher sends values normalized to [0, 1]. Invalid or
                # transient values are clipped only for display.
                gray = np.clip(image * 255.0, 0, 255).astype(np.uint8)
                display = cv2.resize(
                    gray,
                    (self.width * 8, self.height * 8),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imshow(window_name, display)

            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), 27):
                break

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", "-i", default="lo", help="DDS network interface")
    parser.add_argument("--height", type=int, default=48, help="expected image height")
    parser.add_argument("--width", type=int, default=64, help="expected image width")
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")

    # Match deploy.py: domain 1 is used for simulation on loopback, domain 0
    # for the physical robot interface.
    if args.interface == "lo":
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, args.interface)

    RTDepthImageViewer(args.height, args.width).run()


if __name__ == "__main__":
    main()
