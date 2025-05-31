DRL SETUP
# 🧠 drl_drone_px4

An advanced ROS 2 package for autonomous drone navigation using **Soft Actor-Critic (SAC)** with **Prioritized Experience Replay (PER)**. It integrates **PX4**, **Gazebo Harmonic**, and **ROS 2 Humble**, enabling obstacle avoidance and learning-based navigation in simulation. This package also supports communication with real hardware via **Micro XRCE-DDS**.

---

## 📁 Repository Structure

drl_drone_px4/
├── drl_px4/ # Core RL training and UAV environment code
├── config/ # ROS-Gazebo-PX4 bridge config
├── test/ # PEP8 / PEP257 / linting tests
├── resource/ # Package resources
├── setup.py # Python package setup
├── package.xml # ROS 2 package manifest
└── README.md

yaml
Copy
Edit

---

## 🚀 Features

- 🧠 SAC + Prioritized Experience Replay for efficient learning  
- 🛩️ PX4-Gazebo-ROS 2 integration  
- 🪄 Modular training and testing scripts  
- 🔄 Micro XRCE-DDS communication with Pixhawk  
- 🖥️ QGroundControl support for telemetry visualization  

---

## 🧰 Requirements

- ROS 2 Humble  
- PX4 Autopilot (v1.14+)  
- Gazebo Harmonic  
- Micro XRCE-DDS agent  
- QGroundControl (latest)  
- Python 3.10  
- `px4_ros_com`, `px4_msgs`, `ros_gz`, etc.

---

## 🔧 Installation & Build

### 1. Clone the repo into your ROS 2 workspace
```bash
cd ~/ros2_ws/src
git clone https://github.com/mohshibinroshankt/drl_drone_px4.git
2. Clone PX4 ROS 2 packages
bash
Copy
Edit
git clone https://github.com/PX4/px4_ros_com.git --recursive
3. Install dependencies
bash
Copy
Edit
sudo apt update
rosdep install --from-paths src --ignore-src -y
4. Build the workspace
bash
Copy
Edit
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
📡 Micro XRCE-DDS Agent Setup (for Pixhawk ↔ Raspberry Pi communication)
1. Install agent on Raspberry Pi
bash
Copy
Edit
sudo apt install micro-xrce-dds-agent
2. Start the agent (assuming Pixhawk is on /dev/ttyUSB0)
bash
Copy
Edit
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600
This bridges DDS topics between Pixhawk and ROS 2.

🧪 Run Simulation
1. Launch Gazebo with PX4 drone
bash
Copy
Edit
cd ~/PX4-Autopilot
make px4_sitl gz_x500
2. Run ROS-Gazebo bridge
bash
Copy
Edit
ros2 launch px4_ros_com sensor_combined_listener.launch.py
3. Run training node (example: SAC in maze)
bash
Copy
Edit
ros2 run drl_px4 train_maze.py
You can modify and run:

train_uav.py

train_sac.py

test_sac.py

test_maze.py

🛰️ Workflow Summary
Start PX4 SITL or connect Pixhawk via Micro XRCE-DDS

Run ROS 2 launch files to bridge PX4 ↔ ROS

Train or test DRL agents using provided scripts

Use QGroundControl for telemetry/monitoring

Evaluate obstacle avoidance and path learning performance

📺 QGroundControl Setup
Download from qgroundcontrol.com

Connect via USB or UDP telemetry

Use to monitor PX4 parameters, position, and sensor data

