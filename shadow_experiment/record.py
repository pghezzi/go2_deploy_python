"""Run the deployment depth node and sensor-only collector in separate processes."""
from common.dds_runtime import configure_dds_runtime

if __name__ == '__main__':
    configure_dds_runtime()

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import yaml

from .core import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--trial-id', required=True)
    parser.add_argument('--interface', default='eth0')
    args = parser.parse_args()
    c = load_config(args.config, args.trial_id)
    c['runtime']['record_only'] = True
    c['camera']['source'] = 'publisher_tap'
    if c['camera']['rgb'] or not c['camera']['realsense_filters'] or c['camera']['preprocessing'] != 'deployment_tensor_only':
        raise ValueError('Deployment node requires RGB disabled, RealSense filters enabled, deployment_tensor_only')
    processes = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    with tempfile.TemporaryDirectory(prefix='shadow_record_') as directory:
        d = Path(directory)
        # A fresh socket and camera process isolate temporal-filter history per trial.
        c['camera']['socket_path'] = str(d/'camera.sock')
        (d/'trial.yaml').write_text(yaml.safe_dump(c))
        camera = {'depth_camera': c['camera'], 'depth_image_shape': [48,64],
                  'cnn_rate_hz': c['runtime']['update_hz'], 'torch_num_threads': 1,
                  'torch_num_interop_threads': 1}
        (d/'camera.yaml').write_text(yaml.safe_dump(camera))
        try:
            collector = subprocess.Popen([sys.executable, '-u', '-m', 'shadow_experiment.collect',
                '--config', str(d/'trial.yaml'), '--interface', args.interface], cwd=ROOT)
            processes.append(collector)
            deadline = time.monotonic()+30
            while not Path(c['camera']['socket_path']).exists():
                if collector.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Collector failed to open camera socket')
                time.sleep(.1)
            publisher = subprocess.Popen([sys.executable, '-u', str(ROOT/'rough_depth_image.py'),
                '--config', str(d/'camera.yaml'), '--interface', args.interface,
                '--shadow-socket', c['camera']['socket_path']], cwd=ROOT, stdin=subprocess.DEVNULL)
            processes.append(publisher)
            while collector.poll() is None:
                if publisher.poll() is not None:
                    raise RuntimeError('Depth publisher exited; see its output above')
                time.sleep(.2)
            if collector.returncode:
                raise SystemExit(collector.returncode)
        except KeyboardInterrupt:
            pass
        finally:
            for process in processes:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            for process in processes:
                # The collector must finish draining its bounded writer queue.
                # A camera shutdown timeout must not truncate saved frames.
                if process is collector:
                    process.wait()
                    continue
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    main()
