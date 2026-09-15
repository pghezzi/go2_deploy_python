"""Run one fixed DepthWaQ locomotion policy for debugging."""

import copy

from controller import DepthWaQController, TSController


class SinglePolicyController(DepthWaQController):
    """Load one combined TorchScript model and keep its policy slot fixed.

    Inputs are the same observations, history, and depth used by DepthWaQController.
    A bundle's swap() is called once before control starts. A standalone exported
    policy without swap() is used directly; policy_index selects its command bounds.
    """

    def __init__(self, config, interface="lo", *, policy_index=None, model_path=None):
        if policy_index is None:
            policy_index = config.fixed_policy_index
        if isinstance(policy_index, bool) or not isinstance(policy_index, int):
            raise ValueError("policy_index must be an integer")
        if policy_index < -1 or policy_index >= config.num_loras:
            raise ValueError(f"Invalid fixed policy index {policy_index}")
        self.fixed_policy_index = policy_index
        single_config = copy.deepcopy(config)
        single_config.split = False
        single_config.terrain_selector_enabled = False
        if model_path is not None:
            single_config.policy_path = str(model_path)
        super().__init__(single_config, interface)

    def _load_policy_models(self):
        super()._load_policy_models()
        if hasattr(self.policy, "swap"):
            self.policy.swap(self.fixed_policy_index)
        self.active_lora_index = self.fixed_policy_index
        print(f"Single-policy debugging: fixed policy index {self.fixed_policy_index}")

    def updateStateMachine(self):
        # Keep sit/stand/control/damping buttons, omitting DepthWaQ's swap buttons.
        TSController.updateStateMachine(self)

    def swap_policy(self, index, source="remote"):
        if index != self.fixed_policy_index:
            raise RuntimeError(
                f"This controller is fixed to policy {self.fixed_policy_index}; "
                "restart it to select a different policy")
