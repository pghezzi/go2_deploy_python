import time
import numpy as np
import cv2
import pyrealsense2 as rs

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from common.depth_image_idl import DepthImage_

TOPIC_DEPTHIMAGE = "rt/depthimage"

#this a rough idea of how it would work in the actual thing
#notes: 
#   - python3.8 is the version we should use
#   - I'm not 100% if the main computer will recive a message from the robot, can't test
#        - correction tested, works (python /home/oscaryoungquist/go2_deploy_python/scanner.py here and ~/bridge_test/build/publisher (executable) on robot)
#   - instalations needed: unitree_sdk2py and pyrealsense2,
#       - unitree_sdk2py instruction here [https://github.com/unitreerobotics/unitree_sdk2_python]
#       - pyrealsense2 is pip available [https://pypi.org/project/pyrealsense2/]
#   - take also from common.depth_image_idl import DepthImage_ as it will be the used format for now
#   should be plug and play but im not sure. Once done, confirm using a simple python script running on this computer similar to scanner.py

class DepthImagePublisher:
    def __init__(self, width=640, height=480, fps=30,
                 depth_range=(0.0, 3.0),
                 depth_image_shape=(48, 64)):  # (height, width) of the published image
        """
        depth_range: (min_mm, max_mm) used to normalize raw depth into [0, 1]
        depth_image_shape: (H, W) to resize/compress the raw depth image to
                            before publishing.
        """
        self.rs_width = width
        self.rs_height = height
        self.depth_min, self.depth_max = depth_range
        self.out_height, self.out_width = depth_image_shape

        # Realsense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(self.config)

        # DDS channel setup
        ChannelFactoryInitialize(0)  # use ChannelFactoryInitialize(0, "eth0") if a specific NIC is needed
        self.publisher = ChannelPublisher(TOPIC_DEPTHIMAGE, DepthImage_)
        self.publisher.Init()

    def publish_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return

        # Raw 16-bit depth data, in millimeters, shape (H, W)
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        # Downsample to (out_height, out_width). INTER_AREA is best for shrinking
        # since it averages pixels rather than just sampling/skipping them.
        depth_small = cv2.resize(
            depth_image,
            (self.out_width, self.out_height),  # cv2 wants (width, height)
            interpolation=cv2.INTER_AREA,
        )

        # Normalize into [0, 1], clip
        normalized = (depth_small - self.depth_min) / (self.depth_max - self.depth_min)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Flatten row-major to match sequence[float32]
        flat = normalized.flatten().tolist()

        msg = DepthImage_(
            width=self.out_width,
            height=self.out_height,
            normalized_value=flat,
        )
        self.publisher.Write(msg)

    def spin(self, rate_hz=30):
        period = 1.0 / rate_hz
        try:
            while True:
                t0 = time.time()
                self.publish_frame()
                time.sleep(max(0, period - (time.time() - t0)))
        finally:
            self.pipeline.stop()


if __name__ == "__main__":
    node = DepthImagePublisher()
    node.spin()