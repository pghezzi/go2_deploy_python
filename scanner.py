import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

TOPIC_NAME = "rt/hello"


def message_handler(msg: String_):
    print(f"Received: {msg.data}")


def main():
    # Init the DDS channel factory. Domain id 0, optional NIC name (e.g. "eth0")
    # -- must match whatever domain id the C++ publisher used.
    ChannelFactoryInitialize(0)

    subscriber = ChannelSubscriber(TOPIC_NAME, String_)
    subscriber.Init(message_handler)

    print(f"Listening on {TOPIC_NAME}... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()