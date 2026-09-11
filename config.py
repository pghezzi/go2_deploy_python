import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]
            self.communication_dt = config["communication_dt"]
            self.inference_rate_hz = float(
                config.get("inference_rate_hz", 1.0 / self.control_dt)
            )
            self.cnn_rate_hz = float(config.get("cnn_rate_hz", 10.0))
            self.timing_log_interval_s = float(
                config.get("timing_log_interval_s", 1.0)
            )
            self.timing_log_path = config.get(
                "timing_log_path", "logs/depthwaq_timing.log"
            )
            # Keep PyTorch from creating a worker pool per concurrent model
            # thread. This matters on the robot, where the actor, depth CNN,
            # DDS callbacks, and camera pipeline share CPUs.
            self.torch_num_threads = int(config.get("torch_num_threads", 1))
            self.torch_num_interop_threads = int(
                config.get("torch_num_interop_threads", 1)
            )
            if (
                self.inference_rate_hz <= 0
                or self.cnn_rate_hz <= 0
                or self.timing_log_interval_s <= 0
                or self.torch_num_threads <= 0
                or self.torch_num_interop_threads <= 0
            ):
                raise ValueError("Timing rates and PyTorch thread counts must be positive.")
            if not np.isclose(self.control_dt, 1.0 / self.inference_rate_hz):
                raise ValueError(
                    "control_dt must equal 1 / inference_rate_hz so action limits "
                    "use the actual policy timestep."
                )

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.policy_path = config["policy_path"]

            self.split = config.get("split", False)
            self.cnn_path = config.get("cnn_path", None)
            self.actor_path = config.get("actor_path", None)
            self.depth_image_shape = config.get("depth_image_shape", [48, 64])
            selector = config.get("terrain_selector", {})
            self.terrain_selector_enabled = bool(selector.get("enabled", False))
            self.terrain_selector_model_path = selector.get("model_path")
            self.terrain_selector_mode = selector.get("mode", "instantaneous")
            self.terrain_selector_ema_alpha = float(selector.get("ema_alpha", 0.6))
            self.terrain_selector_change_patience = int(selector.get("change_patience", 1))
            self.terrain_selector_stable_stay = float(selector.get("stable_stay", 0.9))
            self.terrain_selector_label_to_lora = selector.get("label_to_lora", {})
            if self.terrain_selector_enabled:
                if not self.terrain_selector_model_path:
                    raise ValueError("terrain_selector.model_path is required when enabled")
                if self.terrain_selector_mode not in ("instantaneous", "ema", "bayes"):
                    raise ValueError("terrain_selector.mode must be instantaneous, ema, or bayes")
                if not 0 < self.terrain_selector_ema_alpha <= 1:
                    raise ValueError("terrain_selector.ema_alpha must be in (0, 1]")
                if self.terrain_selector_change_patience < 1:
                    raise ValueError("terrain_selector.change_patience must be >= 1")
                if not 0 < self.terrain_selector_stable_stay <= 1:
                    raise ValueError("terrain_selector.stable_stay must be in (0, 1]")
            self.num_loras = config.get("num_loras", 0)
            self.action_clip = config.get("action_clip", 10.0)
            command_ranges = config.get("command_ranges", {})
            self.command_ranges = {}
            for policy_index, bounds in command_ranges.items():
                lower = np.array(bounds["lower"], dtype=np.float32)
                upper = np.array(bounds["upper"], dtype=np.float32)
                if (
                    lower.shape != (3,)
                    or upper.shape != (3,)
                    or np.any(lower > upper)
                ):
                    raise ValueError(
                        f"Command bounds for policy {policy_index} must be "
                        "three-element vectors with lower <= upper."
                    )
                self.command_ranges[int(policy_index)] = (lower, upper)
            self.torque_limits = np.array(
                config.get("torque_limits", [23.0, 23.0, 40.0] * 4),
                dtype=np.float32,
            )
            self.qd_rate_limits = np.array(
                config.get("qd_rate_limits", [8.0, 12.0, 16.0] * 4),
                dtype=np.float32,
            )
            self.torque_slew_limits = float(config.get("torque_slew_limits", 500.0))


            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
            self.stand_kp = config["stand_kp"]
            self.stand_kd = config["stand_kd"]
            self.ctrl_kp = config["ctrl_kp"]
            self.ctrl_kd = config["ctrl_kd"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.sit_angles = np.array(config["sit_angles"], dtype=np.float32)

            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)
            
            self.num_actions = config["num_actions"]
            self.num_single_obs = config["num_single_obs"]
            self.frame_stack = config["frame_stack"]

    def command_bounds(self, policy_index):
        """Return the unscaled training bounds for a base or LoRA policy."""
        try:
            return self.command_ranges[policy_index]
        except KeyError as exc:
            raise ValueError(
                f"No command range configured for policy index {policy_index}."
            ) from exc
