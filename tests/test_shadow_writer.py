"""Bounded writer overload, ownership, shutdown and disk failure coverage."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shadow_experiment.core import AsyncTrialWriter, GroundTruth, load_config, read_trial

ROOT = Path(__file__).resolve().parents[1]


class AsyncWriterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = load_config(ROOT/'configs/shadow/rough_to_gap.yaml', 'writer_test')
        self.config['output_root'] = self.tmp.name
        self.config['runtime'].update(writer_queue_size=1, chunk_frames=1)
        self.truth = GroundTruth(self.config, time.monotonic_ns())

    def row(self, index):
        return ({'frame_id': index, 'state': {'nested': [index]}},
                np.full((2,2), index, np.uint16), np.zeros((48,64), np.float32), [0]*3, [0]*3)

    def test_bounded_admission_snapshots_and_shutdown_drains(self):
        entered, release = threading.Event(), threading.Event()
        original = np.savez_compressed
        def slow_write(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('Test writer was not released')
            return original(*args, **kwargs)
        with patch('shadow_experiment.core.np.savez_compressed', side_effect=slow_write):
            writer = AsyncTrialWriter(self.config, {}, time.monotonic_ns())
            try:
                self.assertTrue(writer.add(*self.row(1)))
                self.assertTrue(entered.wait(3))
                row = self.row(2)
                self.assertTrue(writer.add(*row))
                row[0]['state']['nested'][0] = 99
                row[1][:] = 99
                self.assertFalse(writer.add(*self.row(3)))
                closer = threading.Thread(target=writer.close, args=('test', self.truth, {}))
                closer.start()
                self.assertTrue(closer.is_alive())
                release.set()
                closer.join(5)
                self.assertFalse(closer.is_alive())
            finally:
                release.set()
                writer.close('test', self.truth, {})
        manifest, rows, issues = read_trial(writer.path)
        self.assertFalse(issues)
        self.assertTrue(manifest['complete'])
        self.assertEqual([r[0]['frame_id'] for r in rows], [1,2])
        self.assertEqual(rows[1][0]['state']['nested'], [2])
        self.assertTrue((rows[1][1]['raw_depth'] == 2).all())
        self.assertEqual(manifest['writer']['queue_dropped'], 1)

    def test_partial_chunk_is_committed_on_close(self):
        self.config['runtime']['chunk_frames'] = 8
        writer = AsyncTrialWriter(self.config, {}, time.monotonic_ns())
        writer.add(*self.row(1))
        writer.close('interrupted', self.truth, {})
        manifest, rows, issues = read_trial(writer.path)
        self.assertFalse(issues)
        self.assertEqual(len(rows), 1)
        self.assertTrue(manifest['complete'])

    def test_failed_write_retains_committed_data_and_marks_incomplete(self):
        writer = AsyncTrialWriter(self.config, {}, time.monotonic_ns())
        writer.add(*self.row(1))
        writer.queue.join()
        with patch('shadow_experiment.core.np.savez_compressed', side_effect=OSError('disk full')):
            writer.add(*self.row(2))
            with self.assertRaisesRegex(RuntimeError, 'disk full'):
                writer.close('test', self.truth, {})
        manifest, rows, issues = read_trial(writer.path)
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['termination_reason'], 'writer_error')
        self.assertEqual([r[0]['frame_id'] for r in rows], [1])
        self.assertIn('unclosed_trial', issues)
