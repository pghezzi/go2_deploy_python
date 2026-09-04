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

## Demo

| Controller Type | GIF | Training Code |
|--- | --- | --- |
|  Teacher-Student  |  ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/ts_demo.gif)   |   [genesis_lr/go2_ts](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ts)  |
| Explicit Estimator | ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/ee_demo.gif) | [genesis_lr/go2_ee](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ee) |
| DreamWaQ | ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/dreamwaq_demo.gif) | [genesis_lr/go2_dreamwaq](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_dreamwaq) |
