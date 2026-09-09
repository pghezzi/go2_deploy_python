from typing import Union
import numpy as np
import time
import torch
import threading
import copy
import platform
from collections import deque
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.command_helper import create_damping_cmd, create_zero_cmd
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config

locker = threading.Lock()

class TSController:
    def __init__(self, config: Config, interface: str) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        print("Loading policy network from:", config.policy_path)
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self._reported_invalid_action = False
        self.target_dof_pos = config.default_angles.copy()
        self.obs_deque = deque(maxlen=config.frame_stack)
        for _ in range(config.frame_stack):
            self.obs_deque.append(np.zeros(config.num_single_obs, dtype=np.float32))
        self.cur_obs = np.zeros(config.num_single_obs, dtype=np.float32) # current obs
        self.euler = np.zeros(3, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])

        # State Machine
        self.state = "zero_torque"  # initial state
        state_transition_total_time = 2.0 # seconds
        self.state_transition_total_steps = int(state_transition_total_time / self.config.control_dt)
        self.transition2sit_step = 0
        self.transition2sit_init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        self.transition2stand_step = 0
        self.transition2stand_init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        self.control_step_count = 0
        
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.InitLowCmd()
        
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        
        if interface != "lo":
            # Disable MCF mode to enable custom control
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(5.0)
            self.sport_client.Init()
            
            self.motion_switcher_client = MotionSwitcherClient()
            self.motion_switcher_client.SetTimeout(5.0)
            self.motion_switcher_client.Init()
            
            status, result = self.motion_switcher_client.CheckMode()
            while result['name']:
                self.sport_client.StandDown()
                self.motion_switcher_client.ReleaseMode()
                status, result = self.motion_switcher_client.CheckMode()
                time.sleep(1.0)
            
            print("Release mcf mode")
        
        self.lowCmdThread = RecurrentThread(
            interval=self.config.communication_dt, 
            target=self.LowCmdHandler,
            name="LowCmdThread")
        self.lowCmdThread.Start()
        
        self.mainControlThread = RecurrentThread(
            interval=self.config.control_dt,
            target=self.mainControlStep,
            name="MainControlThread")
        self.mainControlThread.Start()

    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        PosStopF = 2.146e9
        VelStopF = 16000.0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0
        self._prepare_command_for_publish()

    def _prepare_command_for_publish(self):
        # Only the control thread mutates low_cmd. Publish a separate, complete
        # snapshot so CRC calculation and DDS serialization see identical data.
        # Build at 50 Hz; the sender can keep transmitting the previous snapshot
        # at 500 Hz while inference or the next command update is in progress.
        command = copy.deepcopy(self.low_cmd)
        command.crc = CRC().Crc(command)
        self._command_to_publish = command

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        
    def LowCmdHandler(self):
        command = self._command_to_publish
        self.lowcmd_publisher_.Write(command)
    
    def damping_state(self):
        create_damping_cmd(self.low_cmd)

    def zero_torque_state(self):
        create_zero_cmd(self.low_cmd)
    
    def move_to_sit_pos(self):
        dof_idx = self.config.leg_joint2motor_idx
        sit_pos = self.config.sit_angles
        
        # move to sit pos
        alpha = min(self.transition2sit_step / self.state_transition_total_steps, 1)
        for j in range(self.config.num_actions):
            motor_idx = dof_idx[j]
            target_pos = sit_pos[j]
            self.low_cmd.motor_cmd[motor_idx].q = self.transition2sit_init_dof_pos[j] * (1 - alpha) + target_pos * alpha
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stand_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.stand_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        self.transition2sit_step += 1

    def move_to_stand_pos(self):
        dof_idx = self.config.leg_joint2motor_idx
        stand_pos = self.config.default_angles
        
        # move to stand pos
        alpha = min(self.transition2stand_step / self.state_transition_total_steps, 1)
        for j in range(self.config.num_actions):
            motor_idx = dof_idx[j]
            target_pos = stand_pos[j]
            self.low_cmd.motor_cmd[motor_idx].q = self.transition2stand_init_dof_pos[j] * (1 - alpha) + target_pos * alpha
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stand_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.stand_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.transition2stand_step += 1
    
    def updateStateMachine(self):
        if self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.R1].on_press:
            if self.state == "damping": # damping to sit
                print("Moving to sit pos.")
                # record the current pos
                dof_idx = self.config.leg_joint2motor_idx
                for i in range(self.config.num_actions):
                    self.transition2sit_init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
                # reset step counter
                self.transition2sit_step = 0
                self.state = "sit"
            else:
                raise NotImplementedError("Cannot transition from damping to states other than sit.")
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.R2].on_press:
            if self.state == "sit": # sit to stand
                print("Moving to stand pos.")
                # record the current pos
                dof_idx = self.config.leg_joint2motor_idx
                for i in range(self.config.num_actions):
                    self.transition2stand_init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
                # reset step counter
                self.transition2stand_step = 0
                self.state = "stand"
            else:
                raise NotImplementedError("Cannot transition from sit to states other than stand.")
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.A].on_press:
            if self.state == "stand": # stand to ctrl
                print("Entering control state.")
                self.state = "ctrl"
            else:
                raise NotImplementedError("Cannot transition from stand to states other than ctrl.")
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.Y].on_press:
            print("Enter damping state.")
            self.state = "damping"
            # back to damping from any state
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.X].on_press:
            if self.state == "damping":
                print("Enter zero torque state.")
                self.state = "zero_torque"
            else:
                raise NotImplementedError("Can only enter zero torque state from damping state.")
        else:
            pass

    def mainControlStep(self):
        if self.control_step_count == 50:
            print(f"Current State: {self.state}")
            self.control_step_count = 0
        # Update remote controller state
        self.remote_controller.set(self.low_state.wireless_remote)
        # Update State Machine
        self.updateStateMachine()
        with locker:
            if self.state == "zero_torque":
                self.zero_torque_state()
            elif self.state == "damping":
                self.damping_state()
            elif self.state == "sit":
                self.move_to_sit_pos()
            elif self.state == "stand":
                self.move_to_stand_pos()
            elif self.state == "ctrl":
                self.calculate()
            else:
                raise ValueError("Invalid state.")
            self._prepare_command_for_publish()
        
        self.control_step_count += 1
        
    def calculate(self):
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        self.euler = self.low_state.imu_state.rpy

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        self.cur_obs[:3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.cur_obs[3:6] = gravity_orientation
        self.cur_obs[6:9] = ang_vel
        self.cur_obs[9 : 9 + num_actions] = qj_obs
        self.cur_obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.cur_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        
        self.obs_deque.append(self.cur_obs.copy())
        self.obs_history = np.concatenate([self.obs_deque[i] for i in range(len(self.obs_deque))], 
                                  axis=0)

        # Get the action from the policy network
        cur_obs_tensor = torch.from_numpy(self.cur_obs).unsqueeze(0)
        obs_history_tensor = torch.from_numpy(self.obs_history).unsqueeze(0)
        self.action = self.policy(cur_obs_tensor, obs_history_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.ctrl_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.ctrl_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0
    
# Explicit Estimator Controller
class EEController(TSController):
    # ovveride calculate function
    def calculate(self):
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32)
        self.euler = self.low_state.imu_state.rpy

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        self.cur_obs[:3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.cur_obs[3:6] = gravity_orientation
        self.cur_obs[6:9] = ang_vel
        self.cur_obs[9 : 9 + num_actions] = qj_obs
        self.cur_obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.cur_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        
        self.obs_deque.append(self.cur_obs.copy())
        self.obs_history = np.concatenate([self.obs_deque[i] for i in range(len(self.obs_deque))], 
                                  axis=0)

        # Get the action from the policy network
        obs_history_tensor = torch.from_numpy(self.obs_history).unsqueeze(0)
        self.action = self.policy(obs_history_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.ctrl_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.ctrl_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0

class WaQController(TSController):
    pass


import cv2
import os

HAS_DISPLAY = os.environ.get("DISPLAY")

from common.depth_image_idl import DepthImage_

TOPIC_DEPTHIMAGE = "rt/depthimage"

class DepthWaQController(TSController):
    def __init__(self, config: Config, interface: str) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Configure this before invoking either TorchScript model. Without a
        # cap, concurrent actor/CNN forwards can oversubscribe the robot CPU.
        torch.set_num_threads(config.torch_num_threads)
        try:
            torch.set_num_interop_threads(config.torch_num_interop_threads)
        except RuntimeError as error:
            # This setting is process-global and can only be set before Torch
            # begins inter-op work. Retain the runtime default if it is late.
            print(f"Warning: could not set PyTorch inter-op threads: {error}")

        # Initialize the policy network
        self.split = config.split
        if self.split:
            print("Loading cnn network from:", config.cnn_path)
            print("Loading actor network from:", config.actor_path)
            self.depth_cnn = torch.jit.load(config.cnn_path, map_location="cpu").eval()
            self.cnn = lambda depth_image: self.depth_cnn(depth_image.unsqueeze(0))
            self.policy = torch.jit.load(config.actor_path, map_location="cpu").eval()
        else:
            print("Loading policy network from:", config.policy_path)
            self.policy = torch.jit.load(config.policy_path, map_location="cpu").eval()
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self._reported_invalid_action = False
        self.target_dof_pos = config.default_angles.copy()
        self.obs_deque = deque(maxlen=config.frame_stack)
        for _ in range(config.frame_stack):
            self.obs_deque.append(np.zeros(config.num_single_obs, dtype=np.float32))
        self.cur_obs = np.zeros(config.num_single_obs, dtype=np.float32) # current obs
        self.euler = np.zeros(3, dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0])

        self.torque_limits = torch.from_numpy(config.torque_limits)
        self.qd_rate_limits = torch.from_numpy(config.qd_rate_limits)
        self.torque_slew_limits = config.torque_slew_limits

        # This tracks position offsets from the default pose, matching the
        # input/output convention of limit_position_actions().
        self.prev_actions_scaled = torch.zeros(config.num_actions)
        self.prev_pd_torque = torch.zeros((12))

        self._timing_lock = threading.Lock()
        self._last_inference_time = None
        self._last_cnn_time = None
        self._last_timing_log_time = time.monotonic()
        self._inference_interval_s = None
        self._cnn_interval_s = None
        self._inference_duration_s = None
        self._actor_forward_duration_s = None
        self._cnn_duration_s = None
        self._lowstate_sample = (None, None)
        self._last_publish_time = None
        self._max_publish_interval_s = 0.0
        self._received_states = 0
        self._received_depth_frames = 0
        self._control_diagnostics = {}
        self._limiter_conflicts = 0
        self._timing_log_path = Path(config.timing_log_path)
        self._timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._timing_log_path.open("a", encoding="utf-8") as timing_log:
            timing_log.write(
                "# DepthWaQ timing session started "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            timing_log.write(
                f"# interface={interface} host={platform.node()} arch={platform.machine()} "
                f"policy={config.actor_path if config.split else config.policy_path} "
                f"joint_mapping={config.leg_joint2motor_idx}\n"
            )


        # State Machine
        self.state = "zero_torque"  # initial state
        state_transition_total_time = 2.0 # seconds
        self.state_transition_total_steps = int(state_transition_total_time / self.config.control_dt)
        self.transition2sit_step = 0
        self.transition2sit_init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        self.transition2stand_step = 0
        self.transition2stand_init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        self.control_step_count = 0
        
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.InitLowCmd()
        
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        self.depth_image = torch.zeros((1, *config.depth_image_shape))
        self._depth_sample = (self.depth_image, None)
        self.depth_width = config.depth_image_shape[1]
        self.depth_height = config.depth_image_shape[0]

        if self.split:
            with torch.inference_mode():
                self.visual_latent = self.cnn(self.depth_image)
        else:
            self.visual_latent = self.depth_image.unsqueeze(0)
        self._visual_sample = (self.visual_latent, None)
        self.active_lora_index = -1

        self.depth_subscriber = ChannelSubscriber(TOPIC_DEPTHIMAGE, DepthImage_)
        self.depth_subscriber.Init(self.DepthImageHandler, 1)
        
        if interface != "lo":
            # Disable MCF mode to enable custom control
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(5.0)
            self.sport_client.Init()
            
            self.motion_switcher_client = MotionSwitcherClient()
            self.motion_switcher_client.SetTimeout(5.0)
            self.motion_switcher_client.Init()
            
            status, result = self.motion_switcher_client.CheckMode()
            while result['name']:
                self.sport_client.StandDown()
                self.motion_switcher_client.ReleaseMode()
                status, result = self.motion_switcher_client.CheckMode()
                time.sleep(1.0)
            
            print("Release mcf mode")
        
        self.lowCmdThread = RecurrentThread(
            interval=self.config.communication_dt, 
            target=self.LowCmdHandler,
            name="LowCmdThread")
        self.lowCmdThread.Start()
        
        self.mainControlThread = RecurrentThread(
            interval=1.0 / self.config.inference_rate_hz,
            target=self.mainControlStep,
            name="MainControlThread")
        self.mainControlThread.Start()

        if self.split:
            self.cnnThread = RecurrentThread(
                        interval=1.0 / self.config.cnn_rate_hz,
                        target=self.cnnHandler,
                        name="DepthCNNThread")
            self.cnnThread.Start()

    def LowStateGoHandler(self, msg: LowStateGo):
        received_at = time.monotonic()
        self._lowstate_sample = (msg, received_at)
        self.low_state = msg
        with self._timing_lock:
            self._received_states += 1

    def LowCmdHandler(self):
        super().LowCmdHandler()
        now = time.monotonic()
        with self._timing_lock:
            if self._last_publish_time is not None:
                self._max_publish_interval_s = max(
                    self._max_publish_interval_s, now - self._last_publish_time
                )
            self._last_publish_time = now

    @torch.inference_mode()
    def cnnHandler(self):
        start = time.perf_counter()
        depth_image, received_at = self._depth_sample
        self.visual_latent = self.cnn(depth_image)
        self._visual_sample = (self.visual_latent, received_at)
        self._record_timing("cnn", time.perf_counter() - start)

    def _record_timing(self, task, duration_s, forward_duration_s=None):
        """Log model execution time and completion intervals at a bounded rate."""
        now = time.monotonic()
        timing_line = None
        with self._timing_lock:
            if task == "inference":
                if self._last_inference_time is not None:
                    self._inference_interval_s = now - self._last_inference_time
                self._last_inference_time = now
                self._inference_duration_s = duration_s
                self._actor_forward_duration_s = forward_duration_s
            elif task == "cnn":
                if self._last_cnn_time is not None:
                    self._cnn_interval_s = now - self._last_cnn_time
                self._last_cnn_time = now
                self._cnn_duration_s = duration_s
            else:
                raise ValueError(f"Unknown timing task: {task}")

            if now - self._last_timing_log_time < self.config.timing_log_interval_s:
                return

            def format_interval(interval_s, target_hz):
                if interval_s is None:
                    return f"waiting (target {target_hz:.1f} Hz)"
                return (
                    f"{interval_s * 1000.0:.1f} ms "
                    f"({1.0 / interval_s:.1f} Hz; target {target_hz:.1f} Hz)"
                )

            def format_duration(duration_s):
                return "waiting" if duration_s is None else f"{duration_s * 1000.0:.2f} ms"

            def format_age(received_at):
                return "missing" if received_at is None else f"{max(0.0, now - received_at) * 1000.0:.1f} ms"

            elapsed = now - self._last_timing_log_time
            depth_image, depth_received_at = self._depth_sample
            depth_array = depth_image.numpy()
            timing_line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [Timing] "
                "actor completion interval: "
                f"{format_interval(self._inference_interval_s, self.config.inference_rate_hz)}; "
                f"actor step: {format_duration(self._inference_duration_s)}; "
                f"actor forward: {format_duration(self._actor_forward_duration_s)}; "
                "CNN completion interval: "
                f"{format_interval(self._cnn_interval_s, self.config.cnn_rate_hz)}; "
                f"CNN forward: {format_duration(self._cnn_duration_s)}; "
                f"state={self.state} policy_index={self.active_lora_index}; "
                f"lowstate: {self._received_states / elapsed:.1f} Hz "
                f"age={format_age(self._lowstate_sample[1])}; "
                f"depth: {self._received_depth_frames / elapsed:.1f} Hz "
                f"age={format_age(depth_received_at)} "
                f"range=[{depth_array.min():.3f},{depth_array.max():.3f}] "
                f"zero_fraction={np.mean(depth_array == 0):.3f}; "
                f"visual source age={format_age(self._visual_sample[1])}; "
                f"lowcmd max publish gap={self._max_publish_interval_s * 1000.0:.2f} ms; "
                f"control={self._control_diagnostics}"
            )
            self._last_timing_log_time = now
            self._received_states = 0
            self._received_depth_frames = 0
            self._max_publish_interval_s = 0.0

        # Disk I/O must not block the actor or CNN thread while it holds the
        # shared timing lock.
        if timing_line is not None:
            with self._timing_log_path.open("a", encoding="utf-8") as timing_log:
                timing_log.write(timing_line + "\n")
    
    def DepthImageHandler(self, msg: DepthImage_):
        depth = np.asarray(msg.normalized_value, dtype=np.float32)

        expected_height, expected_width = self.config.depth_image_shape
        expected_size = expected_height * expected_width

        if (msg.height, msg.width) != (expected_height, expected_width) or depth.size != expected_size:
            print(
                "Ignoring depth image with unexpected shape "
                f"{msg.height}x{msg.width}; expected {expected_height}x{expected_width}."
            )
            return

        if not np.all(np.isfinite(depth)) or np.any((depth < 0) | (depth > 1)):
            print("Ignoring depth image outside the finite normalized [0, 1] range.")
            return

        # Shape: [1, H, W]
        self.depth_image = torch.from_numpy(
            depth.reshape(expected_height, expected_width).copy()
        ).unsqueeze(0)
        received_at = time.monotonic()
        self._depth_sample = (self.depth_image, received_at)
        with self._timing_lock:
            self._received_depth_frames += 1

        if not self.split:
            self.visual_latent = self.depth_image.unsqueeze(0)
            self._visual_sample = (self.visual_latent, received_at)
        
        if HAS_DISPLAY:
            # -----------------------------
            # Display depth image with OpenCV
            # -----------------------------
            depth_np = self.depth_image.squeeze(0).numpy()
            depth_8u = np.clip(depth_np * 255.0, 0, 255).astype(np.uint8)
            cv2.imshow("Depth Image", depth_8u)
            cv2.waitKey(1)

        #print("Depth Received")

    def swap_policy(self, index):
        if self.split:
            self.depth_cnn.swap(index)
        self.policy.swap(index)
        self.active_lora_index = index
        if self.split:
            depth_image, received_at = self._depth_sample
            with torch.inference_mode():
                self.visual_latent = self.cnn(depth_image)
            self._visual_sample = (self.visual_latent, received_at)
        print(f"Switched depth policy to {index}")

    def updateStateMachine(self):
            if self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.left].on_press:
                self.swap_policy(-1)
            elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.right].on_press:
                self.swap_policy((self.active_lora_index + 1) % self.config.num_loras)
            elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.R1].on_press:
                if self.state == "damping": # damping to sit
                    print("Moving to sit pos.")
                    # record the current pos
                    dof_idx = self.config.leg_joint2motor_idx
                    for i in range(self.config.num_actions):
                        self.transition2sit_init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
                    # reset step counter
                    self.transition2sit_step = 0
                    self.state = "sit"
                else:
                    raise NotImplementedError("Cannot transition from damping to states other than sit.")
            elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.R2].on_press:
                if self.state == "sit": # sit to stand
                    print("Moving to stand pos.")
                    # record the current pos
                    dof_idx = self.config.leg_joint2motor_idx
                    for i in range(self.config.num_actions):
                        self.transition2stand_init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
                    # reset step counter
                    self.transition2stand_step = 0
                    self.state = "stand"
                    print("here")
                else:
                    raise NotImplementedError("Cannot transition from sit to states other than stand.")
            elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.A].on_press:
                if self.state == "stand": # stand to ctrl
                    print("Entering control state.")
                    self.state = "ctrl"
                else:
                    raise NotImplementedError("Cannot transition from stand to states other than ctrl.")
            elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.Y].on_press:
                print("Enter damping state.")
                self.state = "damping"
                # back to damping from any state
            elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.X].on_press:
                if self.state == "damping":
                    print("Enter zero torque state.")
                    self.state = "zero_torque"
                else:
                    raise NotImplementedError("Can only enter zero torque state from damping state.")
            else:
                pass

    def limit_position_actions(self, actions_scaled):
        qj = torch.from_numpy(self.qj).float()
        dqj = torch.from_numpy(self.dqj).float()
        default_angles = torch.from_numpy(
            self.config.default_angles
        ).float()
        kp = torch.as_tensor(self.config.ctrl_kp, dtype=torch.float32)
        kd = torch.as_tensor(self.config.ctrl_kd, dtype=torch.float32)

        # ----- 1. Absolute PD torque bounds -----
        torque_low = -self.torque_limits
        torque_high = self.torque_limits

        abs_low = (
            (torque_low + kd * dqj) / kp
            + qj
            - default_angles
        )

        abs_high = (
            (torque_high + kd * dqj) / kp
            + qj
            - default_angles
        )

        # ----- 2. Desired-position rate bounds -----
        max_daction = self.qd_rate_limits * self.config.control_dt

        rate_low = self.prev_actions_scaled - max_daction
        rate_high = self.prev_actions_scaled + max_daction

        # ----- 3. PD torque-slew bounds -----
        max_dtau = self.torque_slew_limits * self.config.control_dt

        previous_torque = torch.clamp(
            self.prev_pd_torque, -self.torque_limits, self.torque_limits
        )
        slew_tau_low = torch.maximum(
            -self.torque_limits,
            previous_torque - max_dtau,
        )

        slew_tau_high = torch.minimum(
            self.torque_limits,
            previous_torque + max_dtau,
        )

        slew_low = (
            (slew_tau_low + kd * dqj) / kp
            + qj
            - default_angles
        )

        slew_high = (
            (slew_tau_high + kd * dqj) / kp
            + qj
            - default_angles
        )

        # ----- Intersect all allowable intervals -----
        actions_low = torch.maximum(
            abs_low,
            torch.maximum(rate_low, slew_low),
        )

        actions_high = torch.minimum(
            abs_high,
            torch.minimum(rate_high, slew_high),
        )

        # A moving joint can make the position-rate and torque intervals
        # disjoint. torch.clamp(min > max) silently returns max, which can
        # violate the torque bound. In that case prioritize the torque/slew
        # interval and report that the position-rate bound could not be met.
        conflicts = actions_low > actions_high
        self._limiter_conflicts = int(conflicts.sum().item())
        actions_low = torch.where(conflicts, slew_low, actions_low)
        actions_high = torch.where(conflicts, slew_high, actions_high)

        actions_limited = torch.clamp(
            actions_scaled,
            min=actions_low,
            max=actions_high,
        )

        # ----- Estimate torque actually commanded -----
        q_des = default_angles + actions_limited

        pd_torque = (
            kp * (q_des - qj)
            - kd * dqj
        )

        self.prev_actions_scaled = actions_limited.detach().clone()
        self.prev_pd_torque = pd_torque.detach().clone()

        return actions_limited

    
    @torch.inference_mode()
    def calculate(self):
        step_start = time.perf_counter()
        low_state, state_received_at = self._lowstate_sample
        if low_state is None:
            raise RuntimeError("No low-state sample received for depth-policy inference.")
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = low_state.imu_state.quaternion
        ang_vel = np.asarray(low_state.imu_state.gyroscope, dtype=np.float32)
        self.euler = low_state.imu_state.rpy

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        raw_command = np.array(
            [
                self.remote_controller.ly,
                -self.remote_controller.lx,
                -self.remote_controller.rx,
            ],
            dtype=np.float32,
        )
        # Never feed NaN/Inf or out-of-distribution joystick commands to the
        # policy. The configured bounds are the unscaled training ranges.
        command_lower, command_upper = self.config.command_bounds(
            self.active_lora_index
        )
        raw_command = np.nan_to_num(
            raw_command,
            nan=0.0,
            posinf=command_upper,
            neginf=command_lower,
        )
        self.cmd = np.clip(
            raw_command,
            command_lower,
            command_upper,
        )

        commands = self.cmd * self.config.max_cmd * self.config.cmd_scale

        num_actions = self.config.num_actions
        self.cur_obs[:3] = commands
        self.cur_obs[3:6] = gravity_orientation
        self.cur_obs[6:9] = ang_vel
        self.cur_obs[9 : 9 + num_actions] = qj_obs
        self.cur_obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.cur_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        
        self.obs_deque.append(self.cur_obs.copy())
        self.obs_history = np.concatenate([self.obs_deque[i] for i in range(len(self.obs_deque))], axis=0)

        # Get the action from the policy network
        cur_obs_tensor = torch.from_numpy(self.cur_obs).unsqueeze(0)
        obs_history_tensor = torch.from_numpy(self.obs_history).unsqueeze(0)
        visual_latent, visual_received_at = self._visual_sample

        # Get the action from the policy network
        #policy_action = self.policy(
        #    cur_obs_tensor,
        #    obs_history_tensor,
        #    visual_latent,
        #).detach().squeeze(0)
        policy_start = time.perf_counter()
        policy_action = self.policy(
            cur_obs_tensor, obs_history_tensor, visual_latent
        ).detach().numpy().squeeze()
        actor_forward_duration = time.perf_counter() - policy_start

        if not np.all(np.isfinite(policy_action)):
            if not self._reported_invalid_action:
                print("Invalid depth-policy action; commanding the neutral pose.")
                self._reported_invalid_action = True
            policy_action = np.zeros(self.config.num_actions, dtype=np.float32)
        else:
            self._reported_invalid_action = False
            policy_action = np.clip(
                policy_action, -self.config.action_clip, self.config.action_clip
            ).astype(np.float32)

        # limit_position_actions expects position offsets from default_angles.
        # Convert normalized policy actions to offsets before limiting.
        action_offset = torch.from_numpy(
            policy_action * self.config.action_scale
        ).float()
        limited_action_offset = self.limit_position_actions(action_offset)
        limited_action_offset = limited_action_offset.detach().cpu().numpy()

        # Training stores the clipped policy request before simulator torque
        # limits and delay. Keep that convention; log how much the additional
        # deployment limiter changes the requested position below.
        self.action = policy_action
        target_dof_pos = self.config.default_angles + limited_action_offset
        limiter_delta = np.abs(limited_action_offset - policy_action * self.config.action_scale)
        now = time.monotonic()
        self._control_diagnostics = {
            "state_age_at_inference_ms": round((now - state_received_at) * 1000, 2),
            "visual_source_age_ms": None if visual_received_at is None else round((now - visual_received_at) * 1000, 2),
            "quat_norm": round(float(np.linalg.norm(quat)), 4),
            "gravity": np.round(gravity_orientation, 3).tolist(),
            "max_abs_dq": round(float(np.max(np.abs(self.dqj))), 3),
            "max_abs_action": round(float(np.max(np.abs(policy_action))), 3),
            "limited_joints": int(np.count_nonzero(limiter_delta > 1e-5)),
            "max_limiter_delta_rad": round(float(limiter_delta.max()), 4),
            "limiter_conflicts": self._limiter_conflicts,
            "max_tracking_error_rad": round(float(np.max(np.abs(target_dof_pos - self.qj))), 4),
        }

        # Build low cmd
        for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
            #self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
            #self.low_cmd.motor_cmd[motor_idx].dq = 0
            #self.low_cmd.motor_cmd[motor_idx].kp = self.config.stand_kp / 2
            #self.low_cmd.motor_cmd[motor_idx].kd = self.config.stand_kd / 2
            #
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.ctrl_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.ctrl_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self._record_timing(
            "inference",
            time.perf_counter() - step_start,
            actor_forward_duration,
        )
