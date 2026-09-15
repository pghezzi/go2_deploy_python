from controller import *
from single_policy_controller import SinglePolicyController

if __name__ == "__main__":
    import argparse

    # run ts controller in sim by default
    parser = argparse.ArgumentParser()
    parser.add_argument('--interface', '-i', type=str, default="lo", help="network interface")
    parser.add_argument("--config", type=str, help="config file name in configs; single_policy.yaml for single_policy, otherwise ts.yaml")
    parser.add_argument("--type", type=str, default="ts",
                        choices=("ts", "ee", "waq", "depthwaq", "single_policy"))
    parser.add_argument("--policy-index", type=int,
                        help="single_policy slot: -1 rough/base, 0 gap, 1 stairs, 2 pit")
    parser.add_argument("--model", help="single_policy combined TorchScript path; defaults to config policy_path")
    args = parser.parse_args()
    if args.config is None:
        args.config = "single_policy.yaml" if args.type == "single_policy" else "ts.yaml"

    # Load config
    config_path = f"./configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    if args.interface == "lo":
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, args.interface)
        
    if args.type == "ts":
        controller = TSController(config, args.interface)
    elif args.type == "ee":
        controller = EEController(config, args.interface)
    elif args.type == "waq":
        controller = WaQController(config, args.interface)
    elif args.type == "depthwaq":
        controller = DepthWaQController(config, args.interface)
    elif args.type == "single_policy":
        controller = SinglePolicyController(
            config, args.interface, policy_index=args.policy_index, model_path=args.model)
    else:
        raise ValueError(f"Unsupported controller type: {args.type}")
    
    while True:
        time.sleep(1)
