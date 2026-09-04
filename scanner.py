#!/usr/bin/env python3
"""
depth_listener.py
"""

import sys
import time

# Point this at the actual directory containing the idl package
sys.path.insert(0, "/home/pablo/Documents/go2_deploy_python/unitree_mujoco/simulate_python/image_publisher")

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
except ImportError:
    print("ERROR: unitree_sdk2py not found.")
    sys.exit(1)

try:
    from idl._DepthImage_ import DepthImage_
except ImportError as e:
    print(f"ERROR importing DepthImage_: {e}")
    print("Check the actual class/module name inside _DepthImage_.py")
    sys.exit(1)

TOPIC_DEPTHIMAGE = "rt/depthimage"

msg_count = 0
last_info = None


def handler(msg: DepthImage_):
    global msg_count, last_info
    msg_count += 1
    n = len(msg.normalized_value)
    first_val = msg.normalized_value[0] if n > 0 else None
    last_info = (msg.width, msg.height, n, first_val)
    print(f"  received #{msg_count}: {msg.width}x{msg.height}, "
          f"len={n}, first_val={first_val}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <domain_id> <interface> [timeout_seconds]")
        sys.exit(1)

    domain_id = int(sys.argv[1])
    interface = sys.argv[2]
    timeout_s = int(sys.argv[3]) if len(sys.argv) >= 4 else 15

    print(f"Listening: domain_id={domain_id} interface={interface} "
          f"timeout={timeout_s}s topic={TOPIC_DEPTHIMAGE}")

    ChannelFactoryInitialize(domain_id, interface)
    sub = ChannelSubscriber(TOPIC_DEPTHIMAGE, DepthImage_)
    sub.Init(handler, 10)

    start = time.time()
    while time.time() - start < timeout_s:
        time.sleep(0.5)

    if msg_count > 0:
        print(f"\nPASS: received {msg_count} message(s)")
        sys.exit(0)
    else:
        print(f"\nFAIL: no messages received within {timeout_s}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
