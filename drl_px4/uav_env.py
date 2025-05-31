import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleLocalPosition, TrajectorySetpoint, VehicleCommand, OffboardControlMode, VehicleStatus
import sensor_msgs.msg
import time

class UAVNode(Node):
    """ROS2 Node for UAV control and sensing."""
    def __init__(self):
        super().__init__('uav_env')
        qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)

        # Subscribers
        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_callback, qos)
        self.lidar_sub = self.create_subscription(
            sensor_msgs.msg.LaserScan,
            '/world/iris_maze/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan',
            self.lidar_callback,
            qos
        )
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos)

        # State variables
        self.pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [x, y, z, vx, vy, vz]
        self.heading = 0.0  # Heading from VehicleLocalPosition
        self.lidar_data = []
        self.vehicle_status = VehicleStatus()
        self.xy_valid = False
        self.z_valid = False

        # Timer for heartbeat
        self.timer = self.create_timer(0.05, self.publish_offboard_control_heartbeat)

    def pos_callback(self, msg):
        self.pos = np.array([msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz], dtype=np.float32)
        self.heading = msg.heading
        self.xy_valid = msg.xy_valid
        self.z_valid = msg.z_valid
        print(f"Received position: {self.pos}, heading={self.heading}, xy_valid={self.xy_valid}, z_valid={self.z_valid}")

    def lidar_callback(self, msg):
        self.lidar_data = [min(r, 10.0) for r in msg.ranges if not np.isnan(r)]
        print(f"LIDAR data received: min={min(self.lidar_data) if self.lidar_data else 10.0}, avg={np.mean(self.lidar_data) if self.lidar_data else 10.0}")

    def status_callback(self, msg):
        self.vehicle_status = msg
        print(f"Vehicle status - Arming: {self.vehicle_status.arming_state}, Nav state: {self.vehicle_status.nav_state}")

    def publish_offboard_control_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(msg)
        print(f"Published OffboardControlMode: position={msg.position}, velocity={msg.velocity}")

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

class UAVEnv(gym.Env):
    def __init__(self, max_steps=300, takeoff_height=-3.0):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # [vx, vy]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)  # [x, y, z, vx, vy, vz, lidar_min, lidar_avg, goal_dist, goal_angle]

        # Initialize ROS2 node
        self.node = UAVNode()
        self.max_range = 10.0
        self.goal = np.array([7.0, 0.0, 0.0], dtype=np.float32)  # Goal position [x, y, z]
        self.offboard_setpoint_counter = 0
        self.has_taken_off = False
        self.has_landed = False
        self.takeoff_steps = 0
        self.max_takeoff_steps = 200
        self.takeoff_height = takeoff_height
        self.step_count = 0
        self.max_steps = max_steps
        self.prev_shaping = None
        self.x_bounds = [-10.0, 10.0]
        self.y_bounds = [-10.0, 10.0]
        self.z_bounds = [-30.0, 0.0]
        self.landing_threshold = 0.5
        self.altitude_threshold = 0.5
        self.boundary_margin = 1.5

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.node.lidar_data = []
        self.offboard_setpoint_counter = 0
        self.has_taken_off = False
        self.has_landed = False
        self.takeoff_steps = 0
        self.step_count = 0
        self.prev_shaping = None
        self.node.xy_valid = False
        self.node.z_valid = False

        # Safe starting position
        safe_start = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            self.takeoff_height
        ], dtype=np.float32)

        # Set OFFBOARD mode and arm the drone
        timeout = 90.0
        start_time = self.node.get_clock().now()
        self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        print("Set OFFBOARD mode")
        self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        print("Arming drone")

        while (self.node.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
            self.node.publish_offboard_control_heartbeat()
            rclpy.spin_once(self.node, timeout_sec=0.1)
            print(f"Current nav_state: {self.node.vehicle_status.nav_state}, arming_state: {self.node.vehicle_status.arming_state}")
            if (self.node.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
                self.node.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED):
                print("Drone is armed and in OFFBOARD mode")
                break
        else:
            print("Failed to arm or set OFFBOARD mode within timeout")
            self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)  # Disarm before raising exception
            raise RuntimeError("Failed to arm or set OFFBOARD mode within timeout")

        # Wait for valid position estimate
        start_time = self.node.get_clock().now()
        while (self.node.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
            self.node.publish_offboard_control_heartbeat()
            rclpy.spin_once(self.node, timeout_sec=0.1)
            print(f"Position validity - xy_valid: {self.node.xy_valid}, z_valid={self.node.z_valid}, pos: {self.node.pos}")
            if self.node.xy_valid and self.node.z_valid:
                print("Valid position estimate received")
                break
        else:
            print("Failed to get a valid position estimate within timeout")
            self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
            raise RuntimeError("Failed to get a valid position estimate within timeout")

        # Move to safe starting position if out of bounds
        if (self.node.pos[0] < self.x_bounds[0] or self.node.pos[0] > self.x_bounds[1] or
            self.node.pos[1] < self.y_bounds[0] or self.node.pos[1] > self.y_bounds[1]):
            print(f"Drone out of bounds at {self.node.pos[:3]}, moving to safe start {safe_start}")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.yaw = 0.0
            msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)

            start_time = self.node.get_clock().now()
            while (self.node.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
                dx = safe_start[0] - self.node.pos[0]
                dy = safe_start[1] - self.node.pos[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 0.5:
                    print("Reached safe starting position")
                    break
                vx = 0.5 * dx
                vy = 0.5 * dy
                vx = max(min(vx, 1.0), -1.0)
                vy = max(min(vy, 1.0), -1.0)
                z_error = self.takeoff_height - self.node.pos[2]
                vz = 0.5 * z_error
                vz = max(min(vz, 1.0), -1.0)
                msg.velocity = [float(vx), float(vy), float(vz)]
                for _ in range(5):
                    self.node.traj_pub.publish(msg)
                    rclpy.spin_once(self.node, timeout_sec=0.02)
                self.node.publish_offboard_control_heartbeat()
                rclpy.spin_once(self.node, timeout_sec=0.02)
                print(f"Moving to safe start - Pos: {self.node.pos[:3]}, Vel: [{vx}, {vy}, {vz}], Dist: {dist}")
            else:
                print("Failed to reach safe starting position within timeout")
                self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                raise RuntimeError("Failed to reach safe starting position within timeout")

        # Adjust altitude
        if abs(self.node.pos[2] - self.takeoff_height) > 0.5:
            print(f"Adjusting altitude to {self.takeoff_height}, current z={self.node.pos[2]}")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.yaw = 0.0
            msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)

            start_time = self.node.get_clock().now()
            while (self.node.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
                z_error = self.takeoff_height - self.node.pos[2]
                vz = 0.5 * z_error
                vz = max(min(vz, 1.0), -1.0)
                msg.velocity = [0.0, 0.0, float(vz)]
                for _ in range(5):
                    self.node.traj_pub.publish(msg)
                    rclpy.spin_once(self.node, timeout_sec=0.02)
                self.node.publish_offboard_control_heartbeat()
                rclpy.spin_once(self.node, timeout_sec=0.02)
                altitude_diff = abs(self.node.pos[2] - self.takeoff_height)
                vel_norm = np.linalg.norm(self.node.pos[3:6])
                print(f"Altitude adjustment - Diff: {altitude_diff}, Velocity norm: {vel_norm}, vz: {vz}")
                if altitude_diff < 0.5 and vel_norm < 0.3:
                    print("Drone reached desired altitude")
                    break
                if altitude_diff < 1.0 and (self.node.get_clock().now() - start_time).nanoseconds / 1e9 > timeout / 2:
                    print("Close enough to desired altitude, proceeding")
                    break
            else:
                print("Failed to reach desired altitude within timeout")
                self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                raise RuntimeError("Failed to reach desired altitude within timeout")

        # Check for obstacles
        if self.node.lidar_data:
            min_dist = min(self.node.lidar_data)
            if min_dist < 0.5:
                print(f"Obstacles too close: {min_dist}m, commanding hover")
                msg = TrajectorySetpoint()
                msg.position = [float('nan'), float('nan'), self.takeoff_height]
                msg.velocity = [0.0, 0.0, 0.0]
                msg.yaw = float(self.get_yaw_to_target())
                msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
                for _ in range(10):
                    self.node.traj_pub.publish(msg)
                    rclpy.spin_once(self.node, timeout_sec=0.02)
        else:
            print("No LIDAR data, proceeding from current position")

        print(f"Starting episode from current position: {self.node.pos[:3]}")
        self.has_taken_off = True
        return self._get_obs(), {"position": self.node.pos[:3].tolist()}

    def step(self, action):
        self.step_count += 1
        print(f"Step count: {self.step_count}/{self.max_steps}")
        for _ in range(5):
            self.node.publish_offboard_control_heartbeat()
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Check for obstacles
        min_dist = min(self.node.lidar_data) if self.node.lidar_data else self.max_range
        if min_dist < 0.3:
            print(f"Obstacle too close: {min_dist}m, forcing hover")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            for _ in range(5):
                self.node.traj_pub.publish(msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)
            obs = self._get_obs()
            return obs, -10.0, True, False, {"reason": "obstacle_too_close", "min_dist": min_dist}

        # Apply corrective action if near boundaries
        boundary_threshold = 0.5
        corrective_factor = 0.0
        corrective_action = [0.0, 0.0]
        if self.node.pos[0] < self.x_bounds[0] + boundary_threshold:
            dist_to_boundary = self.x_bounds[0] + boundary_threshold - self.node.pos[0]
            corrective_factor = max(corrective_factor, dist_to_boundary / boundary_threshold)
            corrective_action[0] = 0.3 * dist_to_boundary / boundary_threshold
            print(f"Near left boundary x={self.node.pos[0]}, corrective action: {corrective_action}")
        elif self.node.pos[0] > self.x_bounds[1] - boundary_threshold:
            dist_to_boundary = self.node.pos[0] - (self.x_bounds[1] - boundary_threshold)
            corrective_factor = max(corrective_factor, dist_to_boundary / boundary_threshold)
            corrective_action[0] = -0.3 * dist_to_boundary / boundary_threshold
            print(f"Near right boundary x={self.node.pos[0]}, corrective action: {corrective_action}")
        if self.node.pos[1] < self.y_bounds[0] + boundary_threshold:
            dist_to_boundary = self.y_bounds[0] + boundary_threshold - self.node.pos[1]
            corrective_factor = max(corrective_factor, dist_to_boundary / boundary_threshold)
            corrective_action[1] = 0.3 * dist_to_boundary / boundary_threshold
            print(f"Near bottom boundary y={self.node.pos[1]}, corrective action: {corrective_action}")
        elif self.node.pos[1] > self.y_bounds[1] - boundary_threshold:
            dist_to_boundary = self.node.pos[1] - (self.y_bounds[1] - boundary_threshold)
            corrective_factor = max(corrective_factor, dist_to_boundary / boundary_threshold)
            corrective_action[1] = -0.3 * dist_to_boundary / boundary_threshold
            print(f"Near top boundary y={self.node.pos[1]}, corrective action: {corrective_action}")

        # Terminate if out of bounds
        if (self.node.pos[0] < self.x_bounds[0] or self.node.pos[0] > self.x_bounds[1] or
            self.node.pos[1] < self.y_bounds[0] or self.node.pos[1] > self.y_bounds[1] or
            self.node.pos[2] < self.z_bounds[0] or self.node.pos[2] > self.z_bounds[1]):
            target_dist = np.sqrt((self.node.pos[0] - self.goal[0])**2 + (self.node.pos[1] - self.goal[1])**2 + (self.node.pos[2] - self.goal[2])**2)
            print(f"Drone out of bounds: x={self.node.pos[0]}, y={self.node.pos[1]}, z={self.node.pos[2]}, target_dist={target_dist}. Terminating episode.")
            obs = self._get_obs()
            return obs, -100.0, True, False, {"reason": "out_of_bounds", "position": self.node.pos[:3].tolist()}

        # Terminate if too far from target
        target_dist = np.sqrt((self.node.pos[0] - self.goal[0])**2 + (self.node.pos[1] - self.goal[1])**2 + (self.node.pos[2] - self.goal[2])**2)
        if target_dist > 12.0:
            print(f"Drone too far from target: distance={target_dist} > 12m. Terminating episode.")
            obs = self._get_obs()
            return obs, -100.0, True, False, {"reason": "too_far_from_target", "target_dist": target_dist}

        # Emergency check for vertical speed
        if abs(self.node.pos[5]) > 3.0:
            print(f"Emergency: vz={self.node.pos[5]} out of [-3, 3]. Forcing hover.")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
            for _ in range(5):
                self.node.traj_pub.publish(msg)
                rclpy.spin_once(self.node, timeout_sec=0.1)
            obs = self._get_obs()
            return obs, -10.0, True, False, {"reason": "vertical_speed_exceeded", "vz": self.node.pos[5]}

        # Check if close to target and initiate landing
        goal_dist = np.sqrt((self.node.pos[0] - self.goal[0])**2 + (self.node.pos[1] - self.goal[1])**2)
        altitude_diff = abs(self.node.pos[2] - self.goal[2])
        if goal_dist < self.landing_threshold and altitude_diff < self.altitude_threshold and not self.has_landed:
            print(f"Close to target: XY dist={goal_dist}, Z diff={altitude_diff}. Initiating landing.")
            self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.has_landed = True
            # Wait for landing to complete (check arming state)
            start_time = self.node.get_clock().now()
            timeout = 30.0
            while (self.node.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
                self.node.publish_offboard_control_heartbeat()
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if self.node.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    print("Drone has landed and disarmed")
                    break
            else:
                print("Landing timed out, assuming landed")
            obs = self._get_obs()
            return obs, 300.0, True, False, {"reason": "landed", "goal_dist": goal_dist, "altitude_diff": altitude_diff}

        # Apply action with corrective blending
        msg = TrajectorySetpoint()
        if goal_dist < self.landing_threshold:
            z_error = self.goal[2] - self.node.pos[2]
        else:
            z_error = self.takeoff_height - self.node.pos[2]
        vz = 0.5 * z_error
        vz = max(min(vz, 1.0), -1.0)

        # Blend corrective action with agent action
        noise = np.random.normal(0, 0.1, size=2)  # Exploration noise
        scaled_action = np.clip(action + noise, -1.0, 1.0) * 1.5
        blended_action = [
            (1.0 - corrective_factor) * scaled_action[0] + corrective_factor * corrective_action[0],
            (1.0 - corrective_factor) * scaled_action[1] + corrective_factor * corrective_action[1]
        ]
        msg.velocity = [float(blended_action[0]), float(blended_action[1]), float(vz)]
        msg.position = [float('nan'), float('nan'), self.takeoff_height]
        msg.yaw = float(self.get_yaw_to_target())
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        for _ in range(5):
            self.node.traj_pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.1)
        print(f"Applying action: velocity={msg.velocity}, position={msg.position}, yaw={msg.yaw}")

        # Compute observation, reward, and termination conditions
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_done()
        truncated = self.step_count >= self.max_steps

        print(f"Terminated: {terminated}, Truncated: {truncated}")
        info = {
            "position": self.node.pos[:3].tolist(),
            "velocity": self.node.pos[3:6].tolist(),
            "goal_dist": goal_dist,
            "altitude_diff": altitude_diff,
            "min_dist": min_dist
        }
        return obs, reward, bool(terminated), bool(truncated), info

    def _get_obs(self):
        lidar_min = min(self.node.lidar_data) if self.node.lidar_data else self.max_range
        lidar_avg = np.mean(self.node.lidar_data) if self.node.lidar_data else self.max_range
        dx, dy = self.goal[0] - self.node.pos[0], self.goal[1] - self.node.pos[1]
        goal_dist = np.sqrt(dx**2 + dy**2)
        # Compute goal angle using heading
        goal_angle = np.arctan2(dy, dx) - self.node.heading
        # Normalize angle to [-pi, pi]
        goal_angle = np.arctan2(np.sin(goal_angle), np.cos(goal_angle))
        print(f"Observation - Goal dist: {goal_dist}, Goal angle: {goal_angle}, Lidar min: {lidar_min}, Lidar avg: {lidar_avg}")
        return np.array([*self.node.pos, lidar_min, lidar_avg, goal_dist, goal_angle], dtype=np.float32)

    def _compute_reward(self, action):
        dx, dy, dz = self.goal[0] - self.node.pos[0], self.goal[1] - self.node.pos[1], self.goal[2] - self.node.pos[2]
        vx, vy, vz = self.node.pos[3], self.node.pos[4], self.node.pos[5]
        ax, ay = action
        min_dist = min(self.node.lidar_data) if self.node.lidar_data else self.max_range

        # Balanced shaping reward
        C = 1.0 if np.sqrt(dx**2 + dy**2) < self.landing_threshold and abs(dz) < self.altitude_threshold else 0.0
        shaping = (
            -5.0 * np.sqrt(dx**2 + dy**2 + dz**2)  # Reduced distance penalty
            -2.0 * np.sqrt(vx**2 + vy**2 + vz**2)  # Reduced velocity penalty
            -0.5 * np.sqrt(ax**2 + ay**2)         # Control effort penalty
            + 10.0 * C * (1.0 - abs(ax))          # Throttle bonus when landed
            + 10.0 * C * (1.0 - abs(ay))
        )

        # Compute reward as difference in shaping
        reward = shaping - self.prev_shaping if self.prev_shaping is not None else 0.0
        self.prev_shaping = shaping

        # Obstacle penalty/reward
        if min_dist < 1.0:
            reward -= 3.0 * (1.0 - min_dist)
        elif min_dist > 2.0:
            reward += 1.0

        # Exploration bonus
        reward += 0.1

        print(f"Reward components - Dist: {np.sqrt(dx**2 + dy**2 + dz**2)}, Vel: {np.sqrt(vx**2 + vy**2 + vz**2)}, Action: {np.sqrt(ax**2 + ay**2)}, Reward: {reward}")
        return reward

    def _check_done(self):
        goal_dist = np.sqrt((self.node.pos[0] - self.goal[0])**2 + (self.node.pos[1] - self.goal[1])**2)
        min_dist = min(self.node.lidar_data) if self.node.lidar_data else self.max_range
        altitude_diff = abs(self.node.pos[2] - self.goal[2])
        goal_condition = goal_dist < 0.5
        obstacle_condition = min_dist < 0.3
        altitude_condition = goal_dist < self.landing_threshold and altitude_diff > self.altitude_threshold
        landing_condition = self.has_landed
        print(f"Done conditions - Goal dist: {goal_dist}, Min dist: {min_dist}, Altitude diff: {altitude_diff}, Landed: {landing_condition}")
        return goal_condition or obstacle_condition or altitude_condition or landing_condition

    def get_yaw_to_target(self):
        dx = self.goal[0] - self.node.pos[0]
        dy = self.goal[1] - self.node.pos[1]
        yaw = np.arctan2(dy, dx)
        return yaw

    def close(self):
        """Clean up resources."""
        if self.has_taken_off and not self.has_landed:
            print("Environment closing, initiating landing...")
            self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            start_time = self.node.get_clock().now()
            timeout = 30.0
            while (self.node.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
                self.node.publish_offboard_control_heartbeat()
                rclpy.spin_once(self.node, timeout_sec=0.1)
                if self.node.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                    print("Drone has landed and disarmed during cleanup")
                    break
            else:
                print("Landing timed out during cleanup, disarming anyway")
                self.node.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
        self.node.destroy_node()