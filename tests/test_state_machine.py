"""Exercise remote state transitions without DDS or running motor threads."""
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from controller import TSController, DepthWaQController
from single_policy_controller import SinglePolicyController
from common.remote_controller import RemoteController, KeyMap


class StateMachineTests(unittest.TestCase):
    def make_controller(self, cls):
        c = cls.__new__(cls)
        c.state = 'zero_torque'
        c.remote_controller = RemoteController()
        c.config = SimpleNamespace(leg_joint2motor_idx=[1, 0], num_loras=3)
        c.low_state = SimpleNamespace(motor_state=[SimpleNamespace(q=.3), SimpleNamespace(q=-.4)])
        c.transition2sit_init_dof_pos = np.zeros(2)
        c.transition2stand_init_dof_pos = np.zeros(2)
        c.transition2sit_step = 100
        c.transition2stand_step = 100
        c.active_lora_index = -1
        c.swap_policy = Mock()
        return c

    def press(self, c, *keys):
        c.remote_controller.set(bytes(40))
        payload = bytearray(40)
        struct.pack_into('H', payload, 2, sum(1 << k for k in keys))
        c.remote_controller.set(payload)
        c.updateStateMachine()

    def test_forward_and_reverse_with_current_pose_capture(self):
        for cls in (TSController, DepthWaQController, SinglePolicyController):
            with self.subTest(controller=cls.__name__):
                c = self.make_controller(cls)
                for key, state in ((KeyMap.Y, 'damping'), (KeyMap.R1, 'sit'),
                                   (KeyMap.R2, 'stand'), (KeyMap.A, 'ctrl'),
                                   (KeyMap.R2, 'stand'), (KeyMap.R1, 'sit'),
                                   (KeyMap.Y, 'damping'), (KeyMap.X, 'zero_torque')):
                    c.low_state.motor_state[1].q += .1
                    self.press(c, KeyMap.L1, key)
                    self.assertEqual(c.state, state)
                    if state in ('sit', 'stand'):
                        np.testing.assert_allclose(
                            getattr(c, 'transition2' + state + '_init_dof_pos'),
                            [c.low_state.motor_state[1].q, .3])
                        self.assertEqual(getattr(c, 'transition2' + state + '_step'), 0)

    def test_damping_overrides_other_commands_in_every_state(self):
        for cls in (TSController, DepthWaQController, SinglePolicyController):
            c = self.make_controller(cls)
            for state in ('zero_torque', 'damping', 'sit', 'stand', 'ctrl'):
                c.state = state
                self.press(c, KeyMap.L1, KeyMap.Y, KeyMap.R1, KeyMap.R2,
                           KeyMap.A, KeyMap.X, KeyMap.right)
                self.assertEqual(c.state, 'damping')
            c.swap_policy.assert_not_called()

    def test_unavailable_transition_and_missing_modifier_are_ignored(self):
        for cls in (TSController, DepthWaQController, SinglePolicyController):
            c = self.make_controller(cls)
            self.press(c, KeyMap.L1, KeyMap.A)
            self.assertEqual(c.state, 'zero_torque')
            c.state = 'ctrl'
            self.press(c, KeyMap.L1, KeyMap.R1)
            self.assertEqual(c.state, 'ctrl')
            self.press(c, KeyMap.Y)
            self.assertEqual(c.state, 'ctrl')
