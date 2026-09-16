"""Optional local raw-camera stream, with a bounded queue and no DDS control API.

The producer never waits for the consumer. Connection/queue losses are counted
in subsequent packets. Unix socket access is restricted by the collector.
"""
import io
import json
import queue
import socket
import struct
import threading
import time

import numpy as np

MAX_PACKET = 32 * 1024 * 1024


class CameraTap:
    def __init__(self, path):
        self.path = path
        self.queue = queue.Queue(maxsize=2)
        self.dropped = 0
        self.offered = 0
        self.identity = {}
        self.stats_lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def offer(self, frame, scale, receipt_ns, filtered_depth=None, filter_ms=None, processed_depth=None):
        with self.stats_lock:
            self.offered += 1
            offered, dropped = self.offered, self.dropped
        metadata = {'frame_id': int(frame.get_frame_number()), 'receipt_ns': receipt_ns,
                    'camera_identity': self.identity, 'realsense_filter_ms': filter_ms,
                    'filter_history': 'shared publisher lifetime; filters are not reset by the shadow collector',
                    'sensor_timestamp_ms': frame.get_timestamp(),
                    'sensor_clock': str(frame.get_frame_timestamp_domain()),
                    'depth_scale_m': scale, 'tap_offered': offered, 'tap_dropped': dropped,
                    'rgb_timestamp_ms': None, 'source': 'publisher_tap',
                    'raw_units': 'uint16 sensor units; multiply depth_scale_m for meters'}
        raw = np.asanyarray(frame.get_data()).copy()
        try:
            self.queue.put_nowait((metadata, raw, filtered_depth, processed_depth))
        except queue.Full:
            self.record_drop()

    def record_drop(self):
        with self.stats_lock:
            self.dropped += 1

    def _run(self):
        connection = None
        while not self.stop.is_set():
            try:
                metadata, raw, filtered, processed = self.queue.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                if connection is None:
                    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    connection.settimeout(.5)
                    connection.connect(self.path)
                buf = io.BytesIO()
                extra = {} if filtered is None else {'filtered_depth': filtered}
                if processed is not None:
                    extra['processed_depth'] = processed
                np.savez(buf, **extra, raw=raw, metadata=np.frombuffer(json.dumps(metadata).encode(), np.uint8))
                payload = buf.getvalue()
                connection.sendall(struct.pack('!I', len(payload)) + payload)
            except OSError:
                self.record_drop()
                if connection is not None:
                    connection.close()
                connection = None
        if connection is not None:
            connection.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def recv_packet(connection):
    def exact(size):
        pieces = bytearray()
        while len(pieces) < size:
            data = connection.recv(size - len(pieces))
            if not data:
                raise EOFError('Camera tap disconnected')
            pieces.extend(data)
        return pieces
    size = struct.unpack('!I', exact(4))[0]
    if size > MAX_PACKET:
        raise ValueError('Oversized camera packet')
    with np.load(io.BytesIO(exact(size)), allow_pickle=False) as data:
        metadata = json.loads(data['metadata'].tobytes())
        if 'filtered_depth' in data:
            metadata['_filtered_depth'] = data['filtered_depth'].copy()
        if 'processed_depth' in data:
            metadata['_processed_depth'] = data['processed_depth'].copy()
        return metadata, data['raw'].copy(), None
