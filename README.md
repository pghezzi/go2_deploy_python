This project is an modification of [original python deployment code provided by unitree](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_real)

## Installation

1. Install [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)
   ```bash
   git clone https://github.com/unitreerobotics/unitree_sdk2.git
   cd unitree_sdk2/
   mkdir build
   cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
   sudo make install
   ```

2. Clone unitree_mujoco and compile (for simulation in mujoco)
   
   1. install mujoco
      ```bash
      sudo apt install libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev

      git clone https://github.com/pghezzi/mujoco/tree/fixes
      mkdir build && cd build
      cmake ..
      make -j4
      sudo make install

      sudo apt install libyaml-cpp-dev
      ```
   2. install unitree_mujoco
      ```bash
      git clone https://github.com/pghezzi/unitree_mujoco
      cd unitree_mujoco/simulate
      mkdir build && cd build
      cmake ..
      make -j4
      ```

3. Install via pip
- [pytorch](https://pytorch.org/)
- scipy
- pyyaml

## Usage

-Start sim
   ```bash
   cd unitree_mujoco/simulate/build
   ./unitree_mujoco
   ```

- TS Controller (sim)
    
  ```bash
  python deploy.py --config=ts.yaml --type=ts
  ```
- TS Controller (real)
  
  ```bash
  python deploy.py --config=ts.yaml --type=ts --interface=your_ethernet
  ```
  The interface should be the name of your ethernet card. It can be seen by running `ifconfig` in the terminal.
- EE Controller (sim)

  ```bash
  python deploy.py --config=ee.yaml --type=ee
  ```

## DepthWaQ hardware deployment and diagnostics

```bash
./deploy_depthwaq.sh eth0
```

The launcher sets `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before
starting Python, alongside the controller's configured PyTorch thread limits.
The robot's NumPy and system OpenBLAS libraries otherwise keep separate
four-thread pools even when PyTorch reports one thread. If launching the Python
programs directly, set these environment variables before starting them too.

The launcher passes the same DDS interface and deployment YAML to the camera
publisher and controller. Camera settings live in `configs/depthwaq.yaml` under
`depth_camera`; `depth_image_shape` controls the network input size.

The RealSense pipeline follows the supplied Robot Parkour Learning deployment
reference: hole filling, spatial filtering, temporal filtering, cropping,
normalization, and PyTorch adaptive average pooling. The camera is upright, so
`rotate_180` is false. The range is 0–3 meters and the output is 48×64, matching
the saved training configuration. The publisher honors the actual RealSense
depth unit and processes frames at `cnn_rate_hz` (10 Hz), with the camera
streaming at 30 Hz.

Two reference details differ from the saved training implementation: its
`top:-bottom-1` / `left:-right-1` slices remove an extra bottom row and right
column, and it uses adaptive average pooling instead of training's bicubic
resize. Deployment deliberately follows those reference operations. With the
configured crops, 480×640 becomes 431×575 before pooling. Artificial training
noise is not added to real camera measurements.

To inspect randomly sampled processed frames, set
`depth_camera.save_processed_images: true` (enabled for the current test).
`image_save_probability: 0.1` saves about one frame per second at 10 Hz.
Samples go under `image_save_dir` (`logs/depth_images`) in a new session folder
on each launch. Each sample has a native-resolution grayscale PNG, with
0 meters black and 3 meters white for the current range, and a matching `.npy`
containing the exact normalized float32 values sent over DDS. File writes run
in a background thread; samples are dropped if its bounded queue is full.
Set the flag to false to disable saving.

The publisher logs to `/tmp/depthwaq_depth_publisher.log`. The controller logs
to `logs/depthwaq_timing.log`, including:

- Received low-state and depth frame rates and ages. `age=missing` means no
  valid sample has arrived; a running CNN alone does not establish camera input.
- The receipt age of the frame used for the current visual embedding. These
  ages measure time since the DDS callback, not time since camera exposure.
- Maximum command publication gap in each reporting interval.
- Quaternion norm, projected gravity, joint speed, tracking error, and how
  much the action limiter changed the requested joint targets. Limiter conflicts
  indicate that its position-rate and torque constraints could not both be met.

Commands are published as complete snapshots with matching CRCs. Action history
contains the clipped policy request, as in training; the additional deployment
limiter's effects are reported separately. These fixes do not establish the
cause of a particular hardware oscillation without a corresponding run log.

Offline verification (requires PyTorch and the Unitree Python SDK):

```bash
python -m unittest discover -s tests -v
```

The tests do not open DDS channels or start camera/robot control. Checkpoint
integration checks are skipped if the local exported depth models are absent.

## Demo

| Controller Type | GIF | Training Code |
|--- | --- | --- |
|  Teacher-Student  |  ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/ts_demo.gif)   |   [genesis_lr/go2_ts](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ts)  |
| Explicit Estimator | ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/ee_demo.gif) | [genesis_lr/go2_ee](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ee) |
| DreamWaQ | ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/dreamwaq_demo.gif) | [genesis_lr/go2_dreamwaq](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_dreamwaq) |
