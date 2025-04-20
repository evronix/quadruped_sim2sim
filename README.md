# Isaacgym-Gazebo Sim2Sim for RL Quadruped Robot

Deploying quadruped robot policies trained in Isaac Gym to ROS2 Gazebo for Sim-to-Sim verification.

---

## Overview

This repository provides an easy-to-use Sim2Sim pipeline for testing and validating reinforcement learning (RL) policies of quadruped robots within the **ROS2 Gazebo** environment.

Currently, two pretrained policies are supported:

**[Walk-These-Ways]**  
<img src="https://github.com/user-attachments/assets/9a4557a4-4edb-4af4-af57-21c00eb30f29" width="400" alt="Walk-These-Ways Demo"/>

  Margolis, Gabriel B., and Pulkit Agrawal.  
  _"Walk these ways: Tuning robot control for generalization with multiplicity of behavior."_  
  Conference on Robot Learning, PMLR, 2023.  
  👉 [Project Website](https://gmargo11.github.io/walk-these-ways/)

**[DreamWaQ]**  
<img src="https://github.com/user-attachments/assets/86fa960f-357e-4b9b-a901-a41f0dd0195d" width="600" alt="DreamWaQ Demo"/>

  Nahrendra, I. Made Aswin, Byeongho Yu, and Hyun Myung.  
  _"DreamWaQ: Learning robust quadrupedal locomotion with implicit terrain imagination via deep reinforcement learning."_  
  ICRA 2023, IEEE.  
  👉 [Project Website](https://sites.google.com/view/dreamwaq)

---

## Installation

Tested on **ROS2 Humble** and **Ubuntu 22.04**.

### 1. Install Required ROS2 Packages

```bash
sudo apt install ros-humble-gazebo-ros2-control ros-humble-gazebo-ros2-control-demos
sudo apt install ros-humble-ros2-control ros-humble-controller-manager
sudo apt install ros-humble-gazebo-ros ros-humble-joint-state-publisher
sudo apt install ros-humble-gazebo-ros-pkgs
```

### 2. Install PyTorch (CUDA 11.7)

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
  --index-url https://download.pytorch.org/whl/cu117
```

### 3. Clone and Build the Repository

```bash
git clone https://github.com/evronix/quadruped_sim2sim.git
cd quadruped_sim2sim
colcon build --symlink-install
source install/setup.bash
```

---

## Running the Controller

### 1. Launch the Gazebo Simulation

```bash
ros2 launch robot_gazebo go1_gazebo.launch.py
```

### 2. Run the Policy Controller

- **Walk-These-Ways**

```bash
ros2 run robot_controller run_wtw_policy
```

- **DreamWaQ**

```bash
ros2 run robot_controller run_dreamwaq_policy
```

### 3.Launch the UI

```bash
ros2 run robot_UI run_ui
```

With the UI, you can control the robot's `command_velocity` interactively:

<img src="https://github.com/user-attachments/assets/47d30867-16ad-4d53-af50-28cf05a7e717" width="400" alt="UI Demo"/>




---

## Real Robot Integration

Coming soon...

