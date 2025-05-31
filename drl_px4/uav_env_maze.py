import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleLocalPosition, TrajectorySetpoint, VehicleCommand, OffboardControlMode, VehicleStatus
import sensor_msgs.msg

class UAVEnv(Node, gym.Env):
    def __init__(self):
        super().__init__('uav_env')
        # Action space: [vx, vy] for horizontal velocity control
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Observation space: [x, y, z, vx, vy, vz, lidar_front, lidar_left, lidar_right, goal_dist, goal_angle]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        # QoS profile for ROS2 communication
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
            '/world/walls/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan',
            self.lidar_callback,
            qos
        )
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos)

        # State variables
        self.pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [x, y, z, vx, vy, vz]
        self.lidar_data = []
        self.lidar_sectors = {'front': 0.0, 'left': 0.0, 'right': 0.0}  # Processed LIDAR data
        self.goal = np.array([12.0, 12.0, 0.0], dtype=np.float32)  # Fixed goal position
        self.max_range = 10.0  # Maximum LIDAR range
        self.vehicle_status = VehicleStatus()
        self.offboard_setpoint_counter = 0
        self.has_taken_off = False
        self.has_landed = False
        self.takeoff_steps = 0
        self.max_takeoff_steps = 200
        self.takeoff_height = -2.0  # Constant altitude of 2.5 meters
        self.xy_valid = False
        self.z_valid = False
        self.step_count = 0
        self.max_steps = 500  # Increased for maze solving
        self.prev_shaping = None
        self.prev_goal_dist = None  # Track previous goal distance for progress reward
        self.x_bounds = [-10.0, 10.0]
        self.y_bounds = [-10.0, 10.0]
        self.z_bounds = [-30.0, 0.0]
        self.landing_threshold = 0.5  # Distance to goal for landing
        self.altitude_threshold = 0.5
        self.collision_threshold = 0.5  # Distance to walls for collision
        self.boundary_margin = 1.5
        self.stuck_steps = 0  # Counter for detecting stuck situations
        self.stuck_threshold = 100  # Steps without progress to consider stuck
        self.min_progress = 0.1  # Minimum distance change to reset stuck counter

        # Timer for offboard control heartbeat (increased frequency)
        self.timer = self.create_timer(0.02, self.publish_offboard_control_heartbeat)

    def pos_callback(self, msg):
        self.pos = np.array([msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz], dtype=np.float32)
        self.xy_valid = msg.xy_valid
        self.z_valid = msg.z_valid
        print(f"Received position: {self.pos}, xy_valid={self.xy_valid}, z_valid={self.z_valid}")

    def lidar_callback(self, msg):
        self.lidar_data = [min(r, self.max_range) for r in msg.ranges]
        # Process LIDAR into sectors: front (0° ± 30°), left (90° ± 30°), right (-90° ± 30°)
        n_scans = len(self.lidar_data)
        angle_increment = 360.0 / n_scans
        front_indices = [i for i in range(n_scans) if abs(i * angle_increment) <= 30]
        left_indices = [i for i in range(n_scans) if 60 <= i * angle_increment <= 120]
        right_indices = [i for i in range(n_scans) if -120 <= i * angle_increment <= -60]
        self.lidar_sectors['front'] = min([self.lidar_data[i] for i in front_indices]) if front_indices else self.max_range
        self.lidar_sectors['left'] = min([self.lidar_data[i] for i in left_indices]) if left_indices else self.max_range
        self.lidar_sectors['right'] = min([self.lidar_data[i] for i in right_indices]) if right_indices else self.max_range
        print(f"LIDAR sectors - Front: {self.lidar_sectors['front']}, Left: {self.lidar_sectors['left']}, Right: {self.lidar_sectors['right']}")

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

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self.lidar_data = []
        self.lidar_sectors = {'front': 0.0, 'left': 0.0, 'right': 0.0}
        self.offboard_setpoint_counter = 0
        self.has_taken_off = False
        self.has_landed = False
        self.takeoff_steps = 0
        self.step_count = 0
        self.prev_shaping = None
        self.prev_goal_dist = None
        self.stuck_steps = 0
        self.xy_valid = False
        self.z_valid = False

        # Random starting position within a valid maze area
        safe_start = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            self.takeoff_height
        ], dtype=np.float32)

        # Initialize offboard mode
        for _ in range(10):
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.02)
            self.offboard_setpoint_counter += 1

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        print("Set OFFBOARD mode")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        print("Arming drone")

        # Wait for arming and offboard mode
        timeout = 90.0
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.02)
            print(f"Current nav_state: {self.vehicle_status.nav_state}, arming_state: {self.vehicle_status.arming_state}")
            if (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and 
                self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED):
                print("Drone is armed and in OFFBOARD mode")
                break
        else:
            print("Failed to arm or set OFFBOARD mode within timeout")
            raise RuntimeError("Failed to arm or set OFFBOARD mode within timeout")

        # Wait for valid position estimate
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.02)
            print(f"Position validity - xy_valid: {self.xy_valid}, z_valid={self.z_valid}, pos: {self.pos}")
            if self.xy_valid and self.z_valid:
                print("Valid position estimate received")
                break
        else:
            print("Failed to get a valid position estimate within timeout")
            raise RuntimeError("Failed to get a valid position estimate within timeout")

        # Move to safe starting position if out of bounds
        if (self.pos[0] < self.x_bounds[0] or self.pos[0] > self.x_bounds[1] or
            self.pos[1] < self.y_bounds[0] or self.pos[1] > self.y_bounds[1]):
            print(f"Drone out of bounds at {self.pos[:3]}, moving to safe start {safe_start}")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.yaw = 0.0
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

            start_time = self.get_clock().now()
            while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
                dx = safe_start[0] - self.pos[0]
                dy = safe_start[1] - self.pos[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 0.5:
                    print("Reached safe starting position")
                    break
                vx = 0.5 * dx
                vy = 0.5 * dy
                vx = max(min(vx, 1.0), -1.0)
                vy = max(min(vy, 1.0), -1.0)
                z_error = self.takeoff_height - self.pos[2]
                vz = 0.5 * z_error
                vz = max(min(vz, 1.0), -1.0)
                msg.velocity = [float(vx), float(vy), float(vz)]
                for _ in range(5):
                    self.traj_pub.publish(msg)
                    rclpy.spin_once(self, timeout_sec=0.02)
                self.publish_offboard_control_heartbeat()
                rclpy.spin_once(self, timeout_sec=0.02)
                print(f"Moving to safe start - Pos: {self.pos[:3]}, Vel: [{vx}, {vy}, {vz}], Dist: {dist}")
            else:
                print("Failed to reach safe starting position within timeout")
                raise RuntimeError("Failed to reach safe starting position within timeout")

        # Adjust altitude to takeoff height
        if abs(self.pos[2] - self.takeoff_height) > 0.5:
            print(f"Adjusting altitude to {self.takeoff_height}, current z={self.pos[2]}")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.yaw = 0.0
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

            start_time = self.get_clock().now()
            while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
                z_error = self.takeoff_height - self.pos[2]
                vz = 0.5 * z_error
                vz = max(min(vz, 1.0), -1.0)
                msg.velocity = [0.0, 0.0, float(vz)]
                for _ in range(5):
                    self.traj_pub.publish(msg)
                    rclpy.spin_once(self, timeout_sec=0.02)
                self.publish_offboard_control_heartbeat()
                rclpy.spin_once(self, timeout_sec=0.02)
                altitude_diff = abs(self.pos[2] - self.takeoff_height)
                vel_norm = np.linalg.norm(self.pos[3:6])
                print(f"Altitude adjustment - Diff: {altitude_diff}, Velocity norm: {vel_norm}, vz: {vz}")
                if altitude_diff < 0.5 and vel_norm < 0.3:
                    print("Drone reached desired altitude")
                    break
                if altitude_diff < 1.0 and (self.get_clock().now() - start_time).nanoseconds / 1e9 > timeout / 2:
                    print("Close enough to desired altitude, proceeding")
                    break
            else:
                print("Failed to reach desired altitude within timeout")
                raise RuntimeError("Failed to reach desired altitude within timeout")

        # Check for obstacles at start
        if self.lidar_sectors['front'] < self.collision_threshold or \
           self.lidar_sectors['left'] < self.collision_threshold or \
           self.lidar_sectors['right'] < self.collision_threshold:
            print(f"Obstacles too close at start: Front={self.lidar_sectors['front']}, Left={self.lidar_sectors['left']}, Right={self.lidar_sectors['right']}. Commanding hover.")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(10):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.02)

        print(f"Starting episode from position: {self.pos[:3]}, Goal: {self.goal}")
        self.has_taken_off = True
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        print(f"Step count: {self.step_count}/{self.max_steps}")
        for _ in range(5):
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.02)

        # Check for collision
        if (self.lidar_sectors['front'] < self.collision_threshold or
            self.lidar_sectors['left'] < self.collision_threshold or
            self.lidar_sectors['right'] < self.collision_threshold):
            print(f"Collision detected: Front={self.lidar_sectors['front']}, Left={self.lidar_sectors['left']}, Right={self.lidar_sectors['right']}. Terminating episode.")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(10):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.02)
            obs = self._get_obs()
            return obs, -50.0, True, False, {}

        # Check boundaries
        if (self.pos[0] < self.x_bounds[0] or self.pos[0] > self.x_bounds[1] or
            self.pos[1] < self.y_bounds[0] or self.pos[1] > self.y_bounds[1] or
            self.pos[2] < self.z_bounds[0] or self.pos[2] > self.z_bounds[1]):
            print(f"Drone out of bounds: x={self.pos[0]}, y={self.pos[1]}, z={self.pos[2]}. Terminating episode.")
            obs = self._get_obs()
            return obs, -100.0, True, False, {}

        # Check if too far from goal
        goal_dist = np.sqrt((self.pos[0] - self.goal[0])**2 + (self.pos[1] - self.goal[1])**2)
        if goal_dist > 12.0:
            print(f"Drone too far from goal: distance={goal_dist} > 12m. Terminating episode.")
            obs = self._get_obs()
            return obs, -100.0, True, False, {}

        # Check for excessive vertical velocity
        if abs(self.pos[5]) > 3.0:
            print(f"Emergency: vz={self.pos[5]} out of [-3, 3]. Forcing hover.")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(10):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.02)
            obs = self._get_obs()
            return obs, -10.0, True, False, {}

        # Check if reached goal
        altitude_diff = abs(self.pos[2] - self.goal[2])
        if goal_dist < self.landing_threshold and altitude_diff < self.altitude_threshold and not self.has_landed:
            print(f"Reached goal: XY dist={goal_dist}, Z diff={altitude_diff}. Initiating landing.")
            self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.has_landed = True
            obs = self._get_obs()
            return obs, 500.0, True, False, {}  # Large reward for solving maze

        # Apply corrective action near boundaries
        corrective_action = None
        boundary_threshold = 0.5
        if self.pos[0] < self.x_bounds[0] + boundary_threshold:
            dist_to_boundary = self.x_bounds[0] + boundary_threshold - self.pos[0]
            corrective_action = [0.3 * dist_to_boundary / boundary_threshold, 0.0]
            print(f"Near left boundary x={self.pos[0]}, applying corrective action: {corrective_action}")
        elif self.pos[0] > self.x_bounds[1] - boundary_threshold:
            dist_to_boundary = self.pos[0] - (self.x_bounds[1] - boundary_threshold)
            corrective_action = [-0.3 * dist_to_boundary / boundary_threshold, 0.0]
            print(f"Near right boundary x={self.pos[0]}, applying corrective action: {corrective_action}")
        elif self.pos[1] < self.y_bounds[0] + boundary_threshold:
            dist_to_boundary = self.y_bounds[0] + boundary_threshold - self.pos[1]
            corrective_action = [0.0, 0.3 * dist_to_boundary / boundary_threshold]
            print(f"Near bottom boundary y={self.pos[1]}, applying corrective action: {corrective_action}")
        elif self.pos[1] > self.y_bounds[1] - boundary_threshold:
            dist_to_boundary = self.pos[1] - (self.y_bounds[1] - boundary_threshold)
            corrective_action = [0.0, -0.3 * dist_to_boundary / boundary_threshold]
            print(f"Near top boundary y={self.pos[1]}, applying corrective action: {corrective_action}")

        # Apply action
        msg = TrajectorySetpoint()
        z_error = self.takeoff_height - self.pos[2]  # Maintain constant altitude
        vz = 0.5 * z_error
        if corrective_action:
            msg.velocity = [corrective_action[0], corrective_action[1], float(vz)]
        else:
            noise = np.random.normal(0, 0.05, size=2)  # Reduced noise for smoother actions
            scaled_action = np.clip(action + noise, -1.0, 1.0) * 2.5  # Increased scaling for faster movement
            msg.velocity = [float(scaled_action[0]), float(scaled_action[1]), float(vz)]
        msg.position = [float('nan'), float('nan'), self.takeoff_height]
        msg.yaw = float(self.get_yaw_to_target())
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        for _ in range(10):  # Increased for continuous action
            self.traj_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)
        print(f"Applying action: velocity={msg.velocity}, position={msg.position}, yaw={msg.yaw}")

        # Get observation and reward
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_done()
        truncated = self.step_count >= self.max_steps

        print(f"Terminated: {terminated}, Truncated: {truncated}")
        return obs, reward, bool(terminated), bool(truncated), {}

    def _get_obs(self):
        dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
        goal_dist = np.sqrt(dx**2 + dy**2)
        goal_angle = np.arctan2(dy, dx) - np.arctan2(self.pos[4], self.pos[3])
        obs = np.array([
            *self.pos,
            self.lidar_sectors['front'],
            self.lidar_sectors['left'],
            self.lidar_sectors['right'],
            goal_dist,
            goal_angle
        ], dtype=np.float32)
        print(f"Observation - Goal dist: {goal_dist}, Goal angle: {goal_angle}, LIDAR: {self.lidar_sectors}")
        return obs

    def _compute_reward(self, action):
        dx, dy, dz = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1], self.goal[2] - self.pos[2]
        vx, vy, vz = self.pos[3], self.pos[4], self.pos[5]
        ax, ay = action
        goal_dist = np.sqrt(dx**2 + dy**2)

        # Shaping reward for maze solving
        shaping = (
            -20.0 * goal_dist  # Encourage moving toward goal
            -2.0 * np.sqrt(vx**2 + vy**2 + vz**2)  # Reduced penalty for velocity
            -0.5 * np.sqrt(ax**2 + ay**2)  # Penalize large control efforts
        )

        # Reward as difference in shaping
        reward = shaping - self.prev_shaping if self.prev_shaping is not None else 0.0
        self.prev_shaping = shaping

        # Progress reward: reward reducing distance to goal
        if self.prev_goal_dist is not None:
            dist_change = self.prev_goal_dist - goal_dist
            if dist_change > self.min_progress:
                reward += 20.0 * dist_change  # Increased bonus for progress
                self.stuck_steps = 0  # Reset stuck counter
            else:
                self.stuck_steps += 1  # Increment stuck counter
        self.prev_goal_dist = goal_dist

        # Obstacle avoidance penalty
        min_dist = min(self.lidar_sectors.values())
        if min_dist < 1.0:
            reward -= 10.0 * (1.0 - min_dist)  # Penalize being too close to walls
        elif min_dist > 2.0:
            reward += 5.0  # Bonus for keeping distance from walls

        # Boundary penalty
        boundary_penalty = 0.0
        boundary_threshold = 0.5
        if self.pos[0] < self.x_bounds[0] + boundary_threshold:
            boundary_penalty -= 0.5 * (self.x_bounds[0] + boundary_threshold - self.pos[0])
        elif self.pos[0] > self.x_bounds[1] - boundary_threshold:
            boundary_penalty -= 0.5 * (self.pos[0] - (self.x_bounds[1] - boundary_threshold))
        if self.pos[1] < self.y_bounds[0] + boundary_threshold:
            boundary_penalty -= 0.5 * (self.y_bounds[0] + boundary_threshold - self.pos[1])
        elif self.pos[1] > self.y_bounds[1] - boundary_threshold:
            boundary_penalty -= 0.5 * (self.pos[1] - (self.y_bounds[1] - boundary_threshold))
        reward += boundary_penalty

        # Exploration bonus
        reward += 0.5  # Increased to encourage continuous movement

        # Penalty for being stuck
        if self.stuck_steps >= self.stuck_threshold:
            print(f"Drone stuck for {self.stuck_steps} steps. Terminating episode.")
            reward -= 50.0
            self.stuck_steps = 0  # Reset counter

        print(f"Reward components - Goal dist: {goal_dist}, Vel: {np.sqrt(vx**2 + vy**2 + vz**2)}, Action: {np.sqrt(ax**2 + ay**2)}, Boundary penalty: {boundary_penalty}, Reward: {reward}")
        return reward

    def _check_done(self):
        goal_dist = np.sqrt((self.pos[0] - self.goal[0])**2 + (self.pos[1] - self.goal[1])**2)
        min_dist = min(self.lidar_sectors.values())
        altitude_diff = abs(self.pos[2] - self.goal[2])
        goal_condition = goal_dist < self.landing_threshold and altitude_diff < self.altitude_threshold
        collision_condition = min_dist < self.collision_threshold
        stuck_condition = self.stuck_steps >= self.stuck_threshold
        landing_condition = self.has_landed
        print(f"Done conditions - Goal dist: {goal_dist}, Min dist: {min_dist}, Altitude diff: {altitude_diff}, Stuck steps: {self.stuck_steps}, Landed: {landing_condition}")
        return goal_condition or collision_condition or stuck_condition or landing_condition

    def get_yaw_to_target(self):
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        return np.arctan2(dy, dx)