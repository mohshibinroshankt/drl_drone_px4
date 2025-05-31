# 🧠 drl_drone_px4

An advanced ROS 2 package for autonomous drone navigation using **Soft Actor-Critic (SAC)** with **Prioritized Experience Replay (PER)**. It integrates **PX4**, **Gazebo Harmonic**, and **ROS 2 Humble**, enabling obstacle avoidance and learning-based navigation in simulation, with support for real hardware via **Micro XRCE-DDS**. Monitor and control the drone using **QGroundControl**.

---

## 📁 Repository Structure

```
drl_drone_px4/
├── drl_px4/            # Core RL training and UAV environment code
├── config/             # ROS-Gazebo-PX4 bridge config
├── test/               # PEP8 / PEP257 / linting tests
├── resource/           # Package resources
├── setup.py            # Python package setup
├── package.xml         # ROS 2 package manifest
└── README.md           # Project documentation
```

---

## 🚀 Features

- 🧠 SAC + Prioritized Experience Replay for efficient learning  
- 🛩️ PX4-Gazebo-ROS 2 integration  
- 🪄 Modular training and testing scripts  
- 🔄 Micro XRCE-DDS communication with Pixhawk  
- 🖥️ QGroundControl support for telemetry visualization  

---

## 🧰 Requirements

- **Operating System**: Ubuntu 22.04  
- **ROS 2**: Humble  
- **PX4 Autopilot**: v1.15.4 (recommended for stability)  
- **Gazebo**: Harmonic  
- **Micro XRCE-DDS Agent**: v2.4.2  
- **QGroundControl**: Latest version  
- **Python**: 3.10  
- **Dependencies**: `px4_ros_com`, `px4_msgs`, `ros_gzharmonic`, etc.  

---

## 📦 Creating the Repository and Package

### 1. Create the Repository

Create a new repository on GitHub (or your preferred platform) named `drl_drone_px4`. Clone it to your local machine:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/mohshibinroshankt/drl_drone_px4.git
cd drl_drone_px4
```

### 2. Set Up the Package Structure

Create the necessary directories and files for the ROS 2 package:

```bash
mkdir -p drl_px4 config test resource
touch setup.py package.xml README.md
```

### Create package.xml

Edit `package.xml` with the following content:

```xml
<?xml version="1.0"?>
<package format="3">
  <name>drl_px4</name>
  <version>0.1.0</version>
  <description>ROS 2 package for autonomous drone navigation using SAC and PER</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_python</buildtool_depend>
  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>px4_msgs</depend>
  <depend>ros_gzharmonic</depend>
  <exec_depend>python3-numpy</exec_depend>
  <exec_depend>python3-torch</exec_depend>
  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>
</package>
```

### Create setup.py

Edit `setup.py` with the following content:

```python
from setuptools import setup

package_name = 'drl_px4'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS 2 package for autonomous drone navigation using SAC and PER',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_maze = drl_px4.train_maze:main',
            'train_uav = drl_px4.train_uav:main',
            'train_sac = drl_px4.train_sac:main',
            'test_sac = drl_px4.test_sac:main',
            'test_maze = drl_px4.test_maze:main',
        ],
    },
)
```

### Create a Sample Script

Create placeholder scripts in the `drl_px4` directory to ensure the package builds:

```bash
touch drl_px4/train_maze.py drl_px4/train_uav.py drl_px4/train_sac.py drl_px4/test_sac.py drl_px4/test_maze.py
```

Add a minimal implementation to `drl_px4/train_maze.py` (you can expand this later):

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class TrainMazeNode(Node):
    def __init__(self):
        super().__init__('train_maze')
        self.get_logger().info('Train Maze Node Started')

def main(args=None):
    rclpy.init(args=args)
    node = TrainMazeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Make the script executable:

```bash
chmod +x drl_px4/*.py
```

---

## 🔧 Installation & Build

### 1. Install Prerequisites

Ensure Ubuntu 22.04 is installed, then set up ROS 2 Humble, Gazebo Harmonic, PX4, and Micro XRCE-DDS.

#### Install ROS 2 Humble

Follow the official ROS 2 Humble installation guide for Ubuntu 22.04:

```bash
sudo apt update && sudo apt install -y curl gnupg2 lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
source /opt/ros/humble/setup.bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

#### Install Gazebo Harmonic and ROS-Gazebo Bridge

Install Gazebo Harmonic and the ROS-Gazebo bridge for ROS 2 Humble:

```bash
sudo apt update
sudo apt install -y ros-humble-ros-gzharmonic
```

#### Install PX4

Clone and set up PX4 Autopilot (v1.15.4 recommended):

```bash
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
git checkout v1.15.4
git submodule update --init --recursive
bash ./Tools/setup/ubuntu.sh
```

#### Install Micro XRCE-DDS Agent

Install the Micro XRCE-DDS Agent for communication between PX4 and ROS 2:

```bash
cd ~
git clone -b v2.4.2 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```

#### Install QGroundControl

Download and install QGroundControl for telemetry and monitoring:

```bash
cd ~
wget https://d176tv9ibo4f19.cloudfront.net/release/QGroundControl.AppImage
chmod +x QGroundControl.AppImage
```

### 2. Clone PX4 ROS 2 Packages

Clone necessary PX4 ROS 2 packages into your workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git --recursive
git clone https://github.com/PX4/px4_ros_com.git --recursive
```

### 3. Install Dependencies

Install ROS 2 dependencies for your workspace:

```bash
cd ~/ros2_ws
sudo apt update
rosdep install --from-paths src --ignore-src -y
```

### 4. Build the Workspace

Build and source the ROS 2 workspace:

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

---

## 📡 Set Up Micro XRCE-DDS Communication

Micro XRCE-DDS bridges PX4 and ROS 2, enabling communication between the Pixhawk (or SITL) and a companion computer (e.g., Raspberry Pi).

### 1. Start the Micro XRCE-DDS Agent

On the companion computer (or your machine for simulation):

**For Simulation (UDP):**
```bash
MicroXRCEAgent udp4 -p 8888
```

**For Hardware (e.g., Pixhawk on /dev/ttyUSB0):**
```bash
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600
```

The PX4 firmware includes the Micro XRCE-DDS client, which connects to the agent automatically in SITL mode or when configured for hardware.

---

## 🧪 Running the Simulation and Nodes

### 1. Launch Gazebo with PX4 SITL

Start the PX4 SITL simulation with a drone model (e.g., gz_x500):

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
```

This launches Gazebo Harmonic with the PX4 SITL drone model.

### 2. Run the ROS-Gazebo Bridge

In a new terminal, launch the ROS 2 bridge to connect PX4 topics to ROS 2:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch px4_ros_com sensor_combined_listener.launch.py
```

This bridges PX4 sensor data (e.g., SensorCombined) to ROS 2 topics.

### 3. Run the Training Node

In another terminal, run the SAC training node for the maze environment:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run drl_px4 train_maze
```

You can also run other scripts as needed:

```bash
ros2 run drl_px4 train_uav
ros2 run drl_px4 train_sac
ros2 run drl_px4 test_sac
ros2 run drl_px4 test_maze
```

---

## 📺 QGroundControl Setup

QGroundControl is used to monitor the drone's telemetry, position, and sensor data.

### 1. Launch QGroundControl

Run QGroundControl on your machine:

```bash
cd ~
./QGroundControl.AppImage
```

### 2. Connect to the Drone

- **For Simulation**: QGroundControl should automatically connect to the PX4 SITL instance via UDP (port 14550).
- **For Hardware**: Connect the Pixhawk to your machine via USB, or configure UDP telemetry in QGroundControl (e.g., `udp://@localhost:14550`).

Use QGroundControl to monitor PX4 parameters, visualize the drone's position, and verify sensor data during training or testing.

---

## 🛰️ Workflow Summary

1. **Set Up Environment:**
   - Install ROS 2 Humble, Gazebo Harmonic, PX4, Micro XRCE-DDS, and QGroundControl.
   - Clone and build the `drl_drone_px4` repository in your ROS 2 workspace.

2. **Start Communication:**
   - Launch the Micro XRCE-DDS Agent to bridge PX4 and ROS 2 (UDP for simulation, serial for hardware).

3. **Run Simulation:**
   - Start PX4 SITL with Gazebo (`make px4_sitl gz_x500`).
   - Launch the ROS-Gazebo bridge (`sensor_combined_listener.launch.py`).

4. **Train or Test:**
   - Run the SAC training/testing nodes (`ros2 run drl_px4 train_maze`).

5. **Monitor with QGroundControl:**
   - Use QGroundControl to visualize telemetry, position, and sensor data.

6. **Evaluate Performance:**
   - Assess the drone's obstacle avoidance and path learning in simulation or on hardware.

---

## 📝 Notes

- Ensure your system has sufficient resources (at least 8GB RAM) for Gazebo Harmonic and PX4 SITL.
- For hardware deployment, verify Pixhawk firmware compatibility with PX4 v1.15.4 and Micro XRCE-DDS.
- Expand the training scripts (`train_maze.py`, etc.) with your SAC + PER implementation, integrating PX4 sensor data and control commands.
