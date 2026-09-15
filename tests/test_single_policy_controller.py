"""Single-policy debugging checks with transport and control threads mocked."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import controller
from config import Config
from single_policy_controller import SinglePolicyController


ROOT = Path(__file__).resolve().parents[1]


class SinglePolicyControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Config(ROOT / 'configs/single_policy.yaml')
        self.config.policy_path = str(ROOT / self.config.policy_path)
        self.config.timing_log_path = str(Path(self.directory.name) / 'timing.log')
        if not Path(self.config.policy_path).is_file():
            self.skipTest('Local combined policy checkpoint is unavailable')
        for name in ('ChannelPublisher', 'ChannelSubscriber', 'RecurrentThread', 'TerrainSelector'):
            patcher = patch.object(controller, name)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        patcher = patch.object(controller, 'HAS_DISPLAY', False)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(torch, 'set_num_interop_threads')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_loads_one_model_and_disables_routing_without_mutating_config(self):
        self.config.fixed_policy_index = 2
        self.config.split = True
        self.config.terrain_selector_enabled = True
        with patch.object(torch.jit, 'load', wraps=torch.jit.load) as load:
            instance = SinglePolicyController(self.config)
        self.assertEqual(load.call_count, 1)
        self.assertFalse(instance.split)
        self.assertIsNone(instance.terrain_selector)
        self.TerrainSelector.assert_not_called()
        self.assertEqual(instance.active_lora_index, 2)
        self.assertEqual(instance.fixed_policy_index, 2)
        self.assertFalse(hasattr(instance, 'cnnThread'))
        self.assertFalse(hasattr(instance, 'selectorThread'))
        self.assertTrue(self.config.split)
        self.assertTrue(self.config.terrain_selector_enabled)
        with self.assertRaisesRegex(RuntimeError, 'fixed to policy 2'):
            instance.swap_policy(0)
        instance.remote_controller.button[controller.KeyMap.L1].pressed = True
        instance.remote_controller.button[controller.KeyMap.right].on_press = True
        instance.updateStateMachine()
        self.assertEqual(instance.active_lora_index, 2)
        instance.remote_controller.button[controller.KeyMap.Y].on_press = True
        instance.updateStateMachine()
        self.assertEqual(instance.state, 'damping')

    def test_all_slots_match_direct_model_inference(self):
        reference = torch.jit.load(self.config.policy_path, map_location='cpu').eval()
        for index in (-1, 0, 1, 2):
            with self.subTest(index=index):
                instance = SinglePolicyController(self.config, policy_index=index)
                state = controller.unitree_go_msg_dds__LowState_()
                state.imu_state.quaternion = [1., 0., 0., 0.]
                for joint, motor in enumerate(instance.config.leg_joint2motor_idx):
                    state.motor_state[motor].q = float(instance.config.default_angles[joint])
                instance.LowStateGoHandler(state)
                frame = torch.linspace(0.05, 0.95, 48 * 64).reshape(48, 64)
                instance.DepthImageHandler(controller.DepthImage_(
                    width=64, height=48, normalized_value=frame.flatten().tolist()))
                reference.swap(index)
                torch.manual_seed(10)
                instance.calculate()
                torch.manual_seed(10)
                with torch.inference_mode():
                    expected = reference(torch.from_numpy(instance.cur_obs)[None],
                                         torch.from_numpy(instance.obs_history)[None],
                                         instance._visual_sample[0]).squeeze(0)
                np.testing.assert_allclose(instance.action, expected.clamp(-10, 10).numpy(),
                                           rtol=1e-5, atol=1e-5)
                self.assertEqual(instance.active_lora_index, index)
                self.assertEqual(instance._visual_sample[1], instance._depth_sample[1])

    def test_invalid_slot_fails_before_model_load_or_transport(self):
        for index in (-2, 3, 0.5, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                SinglePolicyController(self.config, policy_index=index)
        self.ChannelPublisher.assert_not_called()
        self.ChannelSubscriber.assert_not_called()


if __name__ == '__main__':
    unittest.main()
