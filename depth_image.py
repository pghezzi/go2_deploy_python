import cv2
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# Unitree connection
ChannelFactoryInitialize(0, "enp3s0")

print("Available cameras:")

for i in range(20):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera {i}: AVAILABLE - {frame.shape}")
        else:
            print(f"Camera {i}: device opened, but no frame")

        cap.release()

# USB camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("USB camera not found")

while True:

    # Get camera frame
    ret, frame = cap.read()

    if not ret:
        continue

    # frame is a NumPy array
    print("Camera shape:", frame.shape)

    # Example:
    # frame -> (480, 640, 3)
    #
    # You can now feed `frame` to:
    # - PyTorch
    # - YOLO
    # - depth estimation
    # - your policy
    # - image processing, etc.

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()