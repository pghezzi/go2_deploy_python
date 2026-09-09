import time
import numpy as np
import torch
import yaml
import pyrealsense2 as rs

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from common.depth_image_idl import DepthImage_
from common.depth_processing import preprocess_depth_array
from common.depth_image_saver import DepthImageSaver

TOPIC_DEPTHIMAGE = "rt/depthimage"


class DepthImagePublisher:
    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        depth_range=(0.0, 3000.0),   # mm
        depth_image_shape=(48, 64),  # H, W

        # Same cropping as VisualHandlerNode
        crop_top=48,
        crop_bottom=0,
        crop_left=28,
        crop_right=36,
        rotate_180=False,
        interface=None,
        save_processed_images=False,
        image_save_probability=0.1,
        image_save_dir="logs/depth_images",
    ):
        self.rs_width = width
        self.rs_height = height
        self.rs_fps = fps

        self.depth_min, self.depth_max = depth_range

        self.out_height = depth_image_shape[0]
        self.out_width = depth_image_shape[1]

        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.rotate_180 = rotate_180
        self._last_stats_time = time.monotonic()
        self._published_frames = 0

        # ============================================================
        # RealSense setup
        # ============================================================

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.depth,
            self.rs_width,
            self.rs_height,
            rs.format.z16,
            self.rs_fps,
        )

        self.profile = self.pipeline.start(self.config)

        # Get actual depth scale
        self.depth_sensor = (
            self.profile.get_device().first_depth_sensor()
        )

        self.depth_scale = self.depth_sensor.get_depth_scale()

        print("Depth scale:", self.depth_scale)
        print(
            "1 raw depth unit =",
            self.depth_scale,
            "meters",
        )
        print(
            "1 raw depth unit =",
            self.depth_scale * 1000,
            "mm",
        )

        # ============================================================
        # RealSense filters
        #
        # Same filters/settings as VisualHandlerNode
        # ============================================================

        self.rs_hole_filling_filter = rs.hole_filling_filter()

        self.rs_spatial_filter = rs.spatial_filter()

        self.rs_spatial_filter.set_option(
            rs.option.filter_magnitude,
            5,
        )

        self.rs_spatial_filter.set_option(
            rs.option.filter_smooth_alpha,
            0.75,
        )

        self.rs_spatial_filter.set_option(
            rs.option.filter_smooth_delta,
            1,
        )

        self.rs_spatial_filter.set_option(
            rs.option.holes_fill,
            4,
        )

        self.rs_temporal_filter = rs.temporal_filter()

        self.rs_temporal_filter.set_option(
            rs.option.filter_smooth_alpha,
            0.75,
        )

        self.rs_temporal_filter.set_option(
            rs.option.filter_smooth_delta,
            1,
        )

        # Exact same order as VisualHandlerNode
        self.rs_filters = [
            self.rs_hole_filling_filter,
            self.rs_spatial_filter,
            self.rs_temporal_filter,
        ]

        # ============================================================
        # DDS setup
        # ============================================================

        ChannelFactoryInitialize(1 if interface == "lo" else 0, interface)

        self.publisher = ChannelPublisher(
            TOPIC_DEPTHIMAGE,
            DepthImage_,
        )

        self.publisher.Init()

        print()

        self.image_saver = (
            DepthImageSaver(image_save_dir, image_save_probability)
            if save_processed_images else None
        )
        print("DepthImagePublisher initialized")
        print("RealSense resolution:", self.rs_width, "x", self.rs_height)
        print("RealSense FPS:", self.rs_fps)
        print("Rotate 180 degrees:", self.rotate_180)
        print("DDS interface:", interface or "automatic")
        print("Resize: adaptive average pooling (parkour reference)")
        print("Crop endpoints: reference top:-bottom-1, left:-right-1")
        print(
            "Crop:",
            "top =", self.crop_top,
            "bottom =", self.crop_bottom,
            "left =", self.crop_left,
            "right =", self.crop_right,
        )
        print(
            "Output resolution:",
            self.out_width,
            "x",
            self.out_height,
        )
        print(
            "Depth range:",
            self.depth_min,
            "-",
            self.depth_max,
            "mm",
        )
        print()

    # ================================================================
    # Preprocessing
    # ================================================================

    def preprocess_depth(self, depth_frame):
        """Apply parkour's RealSense filters, then its tensor preprocessing."""
        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)
        return preprocess_depth_array(
            np.asanyarray(depth_frame.get_data()),
            depth_scale=self.depth_scale,
            output_shape=(self.out_height, self.out_width),
            depth_range_m=(self.depth_min / 1000.0, self.depth_max / 1000.0),
            cropping=(self.crop_top, self.crop_bottom, self.crop_left, self.crop_right),
            rotate_180=self.rotate_180,
        )

    # ================================================================
    # Publish
    # ================================================================

    def publish_frame(self):
        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            print("No depth frame")
            return

        # Preprocess exactly like the deployment pipeline
        normalized = self.preprocess_depth(
            depth_frame
        )

        # ------------------------------------------------------------
        # Sanity checks
        # ------------------------------------------------------------

        expected_size = (
            self.out_height * self.out_width
        )

        if normalized.shape != (
            self.out_height,
            self.out_width,
        ):
            print(
                "ERROR: unexpected output shape:",
                normalized.shape,
            )
            return

        if normalized.size != expected_size:
            print(
                "ERROR: unexpected output size:",
                normalized.size,
                "expected:",
                expected_size,
            )
            return

        # ------------------------------------------------------------
        # Convert numpy array -> Python list
        #
        # This is important because your IDL is:
        #
        # sequence<float32>
        # ------------------------------------------------------------

        flat = (
            normalized
            .reshape(-1)
            .tolist()
        )

        # ------------------------------------------------------------
        # DDS message
        # ------------------------------------------------------------

        msg = DepthImage_(
            width=self.out_width,
            height=self.out_height,
            normalized_value=flat,
        )

        self.publisher.Write(msg)
        if self.image_saver is not None:
            self.image_saver.sample(normalized)
        self._published_frames += 1
        now = time.monotonic()
        elapsed = now - self._last_stats_time
        if elapsed >= 1.0:
            print(
                f"Depth: {self._published_frames / elapsed:.1f} Hz, "
                f"{self.out_width}x{self.out_height}, "
                f"range=[{normalized.min():.3f}, {normalized.max():.3f}], "
                f"zero_fraction={np.mean(normalized == 0):.3f}",
                flush=True,
            )
            self._last_stats_time = now
            self._published_frames = 0

    # ================================================================
    # Main loop
    # ================================================================

    def spin(self, rate_hz=30):
        period = 1.0 / rate_hz

        try:
            while True:
                t0 = time.monotonic()

                self.publish_frame()

                elapsed = time.monotonic() - t0

                time.sleep(
                    max(
                        0.0,
                        period - elapsed,
                    )
                )

        except KeyboardInterrupt:
            print("\nStopping depth publisher...")

        finally:
            try:
                self.pipeline.stop()
            finally:
                if self.image_saver is not None:
                    self.image_saver.close()


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Publish parkour-preprocessed RealSense depth over DDS.")
    parser.add_argument("--interface", "-i", default=None, help="DDS interface, e.g. eth0")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "configs" / "depthwaq.yaml",
        help="Deployment YAML containing depth_camera settings",
    )
    args = parser.parse_args()
    with args.config.open() as config_file:
        config = yaml.safe_load(config_file)
    camera = config.get("depth_camera", {})
    height, width = camera.get("resolution", [480, 640])
    top, bottom, left, right = camera.get("cropping", [48, 0, 28, 36])
    near, far = camera.get("depth_range_m", [0.0, 3.0])
    # This is a separate process from the controller; cap its Torch pool too.
    torch.set_num_threads(config.get("torch_num_threads", 1))
    torch.set_num_interop_threads(config.get("torch_num_interop_threads", 1))
    publisher = DepthImagePublisher(
        width=width,
        height=height,
        fps=camera.get("fps", 30),
        depth_range=(near * 1000.0, far * 1000.0),
        depth_image_shape=config.get("depth_image_shape", [48, 64]),
        crop_top=top, crop_bottom=bottom, crop_left=left, crop_right=right,
        rotate_180=camera.get("rotate_180", False),
        interface=args.interface,
        save_processed_images=camera.get("save_processed_images", False),
        image_save_probability=camera.get("image_save_probability", 0.1),
        image_save_dir=camera.get("image_save_dir", "logs/depth_images"),
    )
    # Like VisualHandlerNode, acquire/process on the embedding refresh period
    # while the camera itself streams at its configured (normally 30 Hz) rate.
    publisher.spin(rate_hz=config.get("cnn_rate_hz", 10.0))


if __name__ == "__main__":
    main()
