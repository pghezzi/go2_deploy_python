"""Offline checks: no camera, DDS participant, or robot commands are started."""

import copy
import importlib
import sys
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, call, patch

import numpy as np
import torch
import torch.nn.functional as F

import controller
from common.depth_processing import preprocess_depth_array
from config import Config


ROOT = Path(__file__).resolve().parents[1]


class DepthProcessingTests(unittest.TestCase):
    def test_matches_supplied_parkour_tensor_path(self):
        # Nonuniform data exposes interpolation differences and crop offsets.
        raw = np.random.default_rng(5).integers(0, 6000, (480, 640), dtype=np.uint16)
        for inverted in (False, True):
            with self.subTest(inverted=inverted):
                reference = np.rot90(raw, k=2) if inverted else raw
                reference = torch.from_numpy(reference.astype(np.float32))[None, None]
                reference = reference[:, :, 48:-1, 28:-37]
                reference = torch.clip(reference, 0.0, 3000.0) / 3000.0
                reference = F.adaptive_avg_pool2d(reference, (48, 64))[0, 0].numpy()
                actual = preprocess_depth_array(raw, .001, rotate_180=inverted)
                np.testing.assert_array_equal(actual, reference)
                self.assertEqual(actual.dtype, np.float32)

    def test_honors_camera_units_and_training_normalization(self):
        raw = np.full((480, 640), 1000, dtype=np.uint16)
        actual = preprocess_depth_array(raw, .002)
        np.testing.assert_allclose(actual, 2.0 / 3.0, atol=1e-6)
        actual = preprocess_depth_array(raw, .002, depth_range_m=(1.0, 3.0))
        np.testing.assert_array_equal(actual, np.full((48, 64), .5, dtype=np.float32))

    def test_upright_keeps_top_and_bottom_orientation(self):
        raw = np.zeros((480, 640), dtype=np.uint16)
        raw[240:] = 3000
        image = preprocess_depth_array(raw, .001)
        self.assertEqual(image[0, 0], 0.0)
        self.assertEqual(image[-1, 0], 1.0)

    def test_rejects_empty_crop(self):
        with self.assertRaises(ValueError):
            preprocess_depth_array(np.zeros((20, 20), dtype=np.uint16), .001)


class CommandPublicationTests(unittest.TestCase):
    def test_in_flight_command_remains_consistent_during_control_update(self):
        instance = controller.TSController.__new__(controller.TSController)
        instance.low_cmd = controller.unitree_go_msg_dds__LowCmd_()
        instance.InitLowCmd()
        instance.low_cmd.motor_cmd[0].q = .25
        instance._prepare_command_for_publish()
        entered_write, finish_write = threading.Event(), threading.Event()
        captured = []

        def write(message):
            entered_write.set()
            if finish_write.wait(2):
                captured.append(copy.deepcopy(message))

        instance.lowcmd_publisher_ = Mock()
        instance.lowcmd_publisher_.Write.side_effect = write
        sender = threading.Thread(target=instance.LowCmdHandler)
        sender.start()
        try:
            self.assertTrue(entered_write.wait(2))
            instance.low_cmd.motor_cmd[0].q = -.5
            instance.low_cmd.motor_cmd[1].kp = 30
            instance._prepare_command_for_publish()
        finally:
            finish_write.set()
            sender.join(2)
        self.assertFalse(sender.is_alive())
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].motor_cmd[0].q, .25)
        self.assertEqual(captured[0].crc, controller.CRC().Crc(captured[0]))
        self.assertEqual(instance._command_to_publish.motor_cmd[0].q, -.5)
        self.assertEqual(
            instance._command_to_publish.crc,
            controller.CRC().Crc(instance._command_to_publish),
        )


class PublisherWiringTests(unittest.TestCase):
    def test_reference_filters_and_configured_dds_interface(self):
        rs = MagicMock()
        rs.pipeline.return_value.start.return_value.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = .001
        frame = Mock()
        raw = np.full((480, 640), 1500, dtype=np.uint16)
        frame.get_data.return_value = raw
        rs.pipeline.return_value.wait_for_frames.return_value.get_depth_frame.return_value = frame
        filter_order = []
        for name in ['hole_filling_filter', 'spatial_filter', 'temporal_filter']:
            def process(value, name=name):
                filter_order.append(name)
                return value
            getattr(rs, name).return_value.process.side_effect = process
        # Import the publisher without requiring the RealSense binary package.
        with patch.dict(sys.modules, {'pyrealsense2': rs}):
            publisher_module = importlib.import_module('rough_depth_image')
        with patch.object(publisher_module, 'rs', rs), \
                patch.object(publisher_module, 'ChannelFactoryInitialize') as initialize, \
                patch.object(publisher_module, 'ChannelPublisher') as publisher:
            instance = publisher_module.DepthImagePublisher(interface='eth0')
            instance.publish_frame()
        initialize.assert_called_once_with(0, 'eth0')
        self.assertEqual(filter_order, ['hole_filling_filter', 'spatial_filter', 'temporal_filter'])
        self.assertEqual(rs.spatial_filter.return_value.set_option.call_args_list, [
            call(rs.option.filter_magnitude, 5),
            call(rs.option.filter_smooth_alpha, .75),
            call(rs.option.filter_smooth_delta, 1),
            call(rs.option.holes_fill, 4),
        ])
        self.assertEqual(rs.temporal_filter.return_value.set_option.call_args_list, [
            call(rs.option.filter_smooth_alpha, .75),
            call(rs.option.filter_smooth_delta, 1),
        ])
        message = publisher.return_value.Write.call_args[0][0]
        self.assertEqual((message.height, message.width), (48, 64))
        np.testing.assert_array_equal(message.normalized_value, np.full(48 * 64, .5))


class DepthControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = Config(ROOT / 'configs' / 'depthwaq.yaml')
        config.timing_log_path = str(Path(self.directory.name) / 'timing.log')
        config.cnn_path = str(ROOT / config.cnn_path)
        config.actor_path = str(ROOT / config.actor_path)
        config.policy_path = str(ROOT / config.policy_path)
        if not Path(config.actor_path).exists():
            self.skipTest('Local exported depth checkpoint is unavailable')
        # Real TorchScript and SDK message/CRC code; all transport and scheduled
        # threads are replaced so construction cannot interact with a robot.
        with patch.object(controller, 'ChannelPublisher'), \
                patch.object(controller, 'ChannelSubscriber'), \
                patch.object(controller, 'RecurrentThread'), \
                patch.object(torch, 'set_num_interop_threads'), \
                patch.object(controller, 'HAS_DISPLAY', False):
            self.instance = controller.DepthWaQController(config, 'lo')

    def send_standing_state(self):
        instance = self.instance
        state = controller.unitree_go_msg_dds__LowState_()
        state.imu_state.quaternion = [1., 0., 0., 0.]
        state.wireless_remote = bytes(40)
        for i, motor_index in enumerate(instance.config.leg_joint2motor_idx):
            state.motor_state[motor_index].q = float(instance.config.default_angles[i])
        instance.LowStateGoHandler(state)

    def send_depth(self, value=.5):
        message = controller.DepthImage_(width=64, height=48, normalized_value=[value] * (48 * 64))
        with patch.object(controller, 'HAS_DISPLAY', False):
            self.instance.DepthImageHandler(message)

    def test_conflicting_rate_limit_preserves_estimated_torque_bounds(self):
        instance = self.instance
        instance.qj = instance.config.default_angles + .5
        instance.dqj[:] = 0
        instance.prev_actions_scaled.fill_(-.5)
        instance.prev_pd_torque.zero_()
        instance.limit_position_actions(torch.zeros(12))
        self.assertGreater(instance._limiter_conflicts, 0)
        self.assertTrue(torch.all(instance.prev_pd_torque.abs() <= 10.0 + 1e-5))
        self.assertTrue(torch.all(instance.prev_pd_torque.abs() <= instance.torque_limits + 1e-5))

    def test_actual_split_policy_and_published_crc(self):
        instance = self.instance
        self.send_standing_state()
        self.send_depth()
        instance.cnnHandler()
        instance.state = 'ctrl'
        instance.mainControlStep()
        instance.LowCmdHandler()
        message = instance.lowcmd_publisher_.Write.call_args[0][0]
        self.assertEqual(message.crc, controller.CRC().Crc(message))
        self.assertTrue(np.isfinite(instance.action).all())
        np.testing.assert_array_equal(instance.cur_obs[3:6], [0, 0, -1])
        np.testing.assert_array_equal(instance.cur_obs[9:21], np.zeros(12))
        self.assertEqual(instance._visual_sample[1], instance._depth_sample[1])
        self.assertFalse(instance._visual_sample[0].requires_grad)

    def test_action_history_matches_training_clipping(self):
        instance = self.instance
        self.send_standing_state()
        instance.policy = Mock(return_value=torch.full((1, 12), 20.0))
        instance.calculate()
        np.testing.assert_array_equal(instance.action, np.full(12, 10.0))
        instance.calculate()
        np.testing.assert_array_equal(instance.cur_obs[33:45], np.full(12, 10.0))

    def test_depth_diagnostics_distinguish_missing_and_stale_frames(self):
        instance = self.instance
        instance._last_timing_log_time -= 2
        instance.cnnHandler()
        text = instance._timing_log_path.read_text()
        self.assertIn('depth: 0.0 Hz age=missing', text)
        self.send_depth()
        accepted_image, _ = instance._depth_sample
        self.send_depth(float('nan'))
        self.assertIs(instance._depth_sample[0], accepted_image)
        instance._depth_sample = (accepted_image, time.monotonic() - 2)
        instance._last_timing_log_time -= 2
        instance.cnnHandler()
        self.assertGreater(time.monotonic() - instance._visual_sample[1], 2)
        text = instance._timing_log_path.read_text()
        self.assertIn('visual source age=', text)
        self.assertIn('zero_fraction=0.000', text)


if __name__ == '__main__':
    unittest.main()
