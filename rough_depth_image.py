import time
import numpy as np
import cv2
import pyrealsense2 as rs

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from depth_image_idl import DepthImage_

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

        ChannelFactoryInitialize(0)

        self.publisher = ChannelPublisher(
            TOPIC_DEPTHIMAGE,
            DepthImage_,
        )

        self.publisher.Init()

        print()
        print("DepthImagePublisher initialized")
        print("RealSense resolution:", self.rs_width, "x", self.rs_height)
        print("RealSense FPS:", self.rs_fps)
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
        """
        Match VisualHandlerNode.get_depth_frame() preprocessing.

        Input:
            RealSense depth_frame

        Output:
            normalized depth image with shape (48, 64)
            values in [0, 1]
        """

        # ------------------------------------------------------------
        # Apply RealSense filters
        # ------------------------------------------------------------

        for rs_filter in self.rs_filters:
            depth_frame = rs_filter.process(depth_frame)

        # ------------------------------------------------------------
        # Convert to numpy
        # ------------------------------------------------------------

        depth_image = np.asanyarray(
            depth_frame.get_data()
        )
        
        # ------------------------------------------------------------
        # Crop
        #
        # Same intended region as VisualHandlerNode
        # ------------------------------------------------------------

        h_end = depth_image.shape[0] - self.crop_bottom
        w_end = depth_image.shape[1] - self.crop_right

        depth_image = depth_image[
            self.crop_top:h_end,
            self.crop_left:w_end,
        ]

        # ------------------------------------------------------------
        # Convert to float32
        #
        # RealSense z16 values are raw depth units.
        # On your camera:
        #
        # depth_scale ~= 0.001 m
        #
        # therefore raw value ~= mm.
        # ------------------------------------------------------------

        depth_image = depth_image.astype(np.float32)

        # Convert raw RealSense units to mm using actual scale.
        #
        # This makes the code robust if another RealSense has
        # a different depth scale.
        depth_image_mm = depth_image * (
            self.depth_scale * 1000.0
        )

        # ------------------------------------------------------------
        # Clip to depth range
        # ------------------------------------------------------------

        depth_image_mm = np.clip(
            depth_image_mm,
            self.depth_min,
            self.depth_max,
        )

        # ------------------------------------------------------------
        # Normalize to [0, 1]
        #
        # Same operation as VisualHandlerNode:
        #
        # torch.clip(...) / (max - min)
        # ------------------------------------------------------------

        depth_normalized = (
            depth_image_mm - self.depth_min
        ) / (
            self.depth_max - self.depth_min
        )

        # ------------------------------------------------------------
        # Resize to 48x64
        #
        # IMPORTANT:
        #
        # VisualHandlerNode uses:
        #
        # F.adaptive_avg_pool2d(...)
        #
        # INTER_AREA is close conceptually, but not exactly the
        # same operation.
        #
        # For this standalone publisher, use INTER_AREA because
        # it performs area averaging when shrinking.
        # ------------------------------------------------------------

        depth_small = cv2.resize(
            depth_normalized,
            (
                self.out_width,
                self.out_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        depth_small = np.clip(
            depth_small,
            0.0,
            1.0,
        ).astype(np.float32)

        return depth_small

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

        # Debug information
        print(
            "Publishing:",
            "width =", msg.width,
            "height =", msg.height,
            "length =", len(msg.normalized_value),
            "min =", min(msg.normalized_value),
            "max =", max(msg.normalized_value),
        )

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        self.publisher.Write(msg)

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
            self.pipeline.stop()


if __name__ == "__main__":
    publisher = DepthImagePublisher(
        width=640,
        height=480,
        fps=30,

        # Same default as deployment:
        # config [0.0, 3.0] meters -> [0, 3000] mm
        depth_range=(0.0, 3000.0),

        # Same network input resolution
        depth_image_shape=(48, 64),

        # Same deployment cropping
        crop_top=48,
        crop_bottom=0,
        crop_left=28,
        crop_right=36,
    )

    publisher.spin(rate_hz=30)