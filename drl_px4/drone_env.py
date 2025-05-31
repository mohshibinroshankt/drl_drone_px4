import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleLocalPosition, TrajectorySetpoint, VehicleCommand, OffboardControlMode, VehicleStatus
import sensor_msgs.msg

class DroneEnv(Node, gym.Env):
    def __init__(self):
        super().__init__('drone_env')
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # [vx, vy]
        # Observation: [x, y, z, vx, vy, vz, goal_dist, goal_angle, 36 LIDAR points]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8 + 36,), dtype=np.float32)
        
        qos = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.traj_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos)
        self.offboard_control_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos)

        self.pos_sub = self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.pos_callback, qos)
        self.lidar_sub = self.create_subscription(
            sensor_msgs.msg.LaserScan, 
            '/world/iris_maze/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan', 
            self.lidar_callback, 
            qos
        )
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos)
        
        self.pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.lidar_data = np.full(36, 8.0, dtype=np.float32)  # 36 points for 360 degrees
        self.max_range = 8.0  # Match LIDAR sensor's max range
        self.goal = np.array([7.0, 0.0, 0.0], dtype=np.float32)  # Target position (x, y, z)
        self.vehicle_status = VehicleStatus()
        self.offboard_setpoint_counter = 0
        self.has_taken_off = False
        self.takeoff_steps = 0
        self.max_takeoff_steps = 200
        self.takeoff_height = -3.0  # 3m above ground for navigation
        self.xy_valid = False
        self.z_valid = False
        self.step_count = 0
        self.max_steps = 200
        self.prev_goal_dist = None
        self.x_bounds = [-10.0, 10.0]
        self.y_bounds = [-10.0, 10.0]
        self.z_bounds = [-30.0, 0.0]
        self.is_landing = False
        self.has_landed = False

        self.timer = self.create_timer(0.05, self.publish_offboard_control_heartbeat)

    def pos_callback(self, msg):
        self.pos = np.array([msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz], dtype=np.float32)
        self.xy_valid = msg.xy_valid
        self.z_valid = msg.z_valid
        print(f"Received position: {self.pos}, xy_valid={self.xy_valid}, z_valid={self.z_valid}")

    def lidar_callback(self, msg):
        # Downsample 360-degree LIDAR to exactly 36 points (every 10 degrees)
        ranges = np.array([min(r, self.max_range) for r in msg.ranges], dtype=np.float32)
        print(f"LIDAR received: {len(ranges)} points, min={np.min(ranges):.2f}, max={np.max(ranges):.2f}")
        
        if len(ranges) >= 360:
            # Expected case: downsample to 36 points
            step = len(ranges) // 36  # Calculate step size to get exactly 36 points
            self.lidar_data = ranges[::step][:36]  # Take every step-th point, ensure 36 points
            print(f"LIDAR downsampling: step={step}, output shape={self.lidar_data.shape}")
        else:
            # Fallback: fill with max_range if insufficient points
            self.lidar_data = np.full(36, self.max_range, dtype=np.float32)
            print(f"Warning: Received {len(ranges)} LIDAR points, expected >= 360. Filling with max_range.")
        
        if self.lidar_data.shape != (36,):
            self.lidar_data = self.lidar_data[:36] if len(self.lidar_data) > 36 else np.pad(self.lidar_data, (0, 36 - len(self.lidar_data)), constant_values=self.max_range)
            print(f"Adjusted LIDAR data to shape {self.lidar_data.shape}")
        
        print(f"LIDAR data: min={np.min(self.lidar_data):.2f}, avg={np.mean(self.lidar_data):.2f}, shape={self.lidar_data.shape}")

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
        self.lidar_data = np.full(36, self.max_range, dtype=np.float32)
        self.offboard_setpoint_counter = 0
        self.has_taken_off = False
        self.takeoff_steps = 0
        self.step_count = 0
        self.prev_goal_dist = None
        self.xy_valid = False
        self.z_valid = False
        self.is_landing = False
        self.has_landed = False

        # Step 1: Set OFFBOARD mode and arm the drone
        for _ in range(10):
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.05)
            self.offboard_setpoint_counter += 1

        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        print("Set OFFBOARD mode")
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        print("Arming drone")

        timeout = 90.0
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.1)
            print(f"Current nav_state: {self.vehicle_status.nav_state}, arming_state: {self.vehicle_status.arming_state}")
            if (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and 
                self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED):
                print("Drone is armed and in OFFBOARD mode")
                break
        else:
            print("Failed to arm or set OFFBOARD mode within timeout")
            raise RuntimeError("Failed to arm or set OFFBOARD mode within timeout")

        # Step 2: Wait for valid position estimate
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout:
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.1)
            print(f"Position validity - xy_valid: {self.xy_valid}, z_valid={self.z_valid}, pos: {self.pos}")
            if self.xy_valid and self.z_valid:
                print("Valid position estimate received")
                break
        else:
            print("Failed to get a valid position estimate within timeout")
            raise RuntimeError("Failed to get a valid position estimate within timeout")

        # Step 3: Reposition to safe starting point if out of bounds
        safe_start = np.array([0.0, 0.0, self.takeoff_height], dtype=np.float32)
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

        # Step 4: Adjust to desired altitude if necessary
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

        # Step 5: Check for obstacles using LIDAR
        min_dist = np.min(self.lidar_data)
        if min_dist < 0.5:
            print(f"Obstacles too close: {min_dist}m, commanding hover")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(10):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.02)

        print(f"Starting episode from current position: {self.pos[:3]}")
        self.has_taken_off = True
        obs = self._get_obs()
        if obs.shape != (44,):
            raise ValueError(f"Observation shape {obs.shape} does not match expected (44,)")
        return obs, {}

    def step(self, action):
        self.step_count += 1
        print(f"Step count: {self.step_count}/{self.max_steps}")
        for _ in range(5):
            self.publish_offboard_control_heartbeat()
            rclpy.spin_once(self, timeout_sec=0.1)

        # Check for close obstacles
        min_dist = np.min(self.lidar_data)
        if min_dist < 0.5:
            print(f"Obstacle too close: {min_dist}m, forcing hover")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(5):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.1)
            obs = self._get_obs()
            if obs.shape != (44,):
                raise ValueError(f"Observation shape {obs.shape} does not match expected (44,)")
            return obs, -50.0, True, False, {}  # Heavy penalty for obstacle proximity

        # Check for boundary proximity and apply corrective action
        boundary_margin = 1.0
        corrective_action = None
        if self.pos[0] < self.x_bounds[0] + boundary_margin:
            corrective_action = [1.0, 0.0]
            print(f"Near left boundary x={self.pos[0]}, applying corrective action: {corrective_action}")
        elif self.pos[0] > self.x_bounds[1] - boundary_margin:
            corrective_action = [-1.0, 0.0]
            print(f"Near right boundary x={self.pos[0]}, applying corrective action: {corrective_action}")
        elif self.pos[1] < self.y_bounds[0] + boundary_margin:
            corrective_action = [0.0, 1.0]
            print(f"Near bottom boundary y={self.pos[1]}, applying corrective action: {corrective_action}")
        elif self.pos[1] > self.y_bounds[1] - boundary_margin:
            corrective_action = [0.0, -1.0]
            print(f"Near top boundary y={self.pos[1]}, applying corrective action: {corrective_action}")

        # Check out-of-bounds
        target_dist = np.sqrt((self.pos[0] - self.goal[0])**2 + (self.pos[1] - self.goal[1])**2 + (self.pos[2] - self.goal[2])**2)
        if (self.pos[0] < self.x_bounds[0] or self.pos[0] > self.x_bounds[1] or
            self.pos[1] < self.y_bounds[0] or self.pos[1] > self.y_bounds[1] or
            self.pos[2] < self.z_bounds[0] or self.pos[2] > self.z_bounds[1]):
            print(f"Drone out of bounds: x={self.pos[0]}, y={self.pos[1]}, z={self.pos[2]}, target_dist={target_dist}. Terminating episode.")
            obs = self._get_obs()
            if obs.shape != (44,):
                raise ValueError(f"Observation shape {obs.shape} does not match expected (44,)")
            return obs, -100.0, True, False, {}

        # Penalty for being too far from target
        if target_dist > 12.0:
            print(f"Drone too far from target: distance={target_dist} > 12m. Terminating episode.")
            obs = self._get_obs()
            if obs.shape != (44,):
                raise ValueError(f"Observation shape {obs.shape} does not match expected (44,)")
            return obs, -100.0, True, False, {}

        # Emergency velocity constraint
        if abs(self.pos[5]) > 3.0:
            print(f"Emergency: vz={self.pos[5]} out of [-3, 3]. Forcing hover.")
            msg = TrajectorySetpoint()
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.velocity = [0.0, 0.0, 0.0]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(5):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.1)
            obs = self._get_obs()
            if obs.shape != (44,):
                raise ValueError(f"Observation shape {obs.shape} does not match expected (44,)")
            return obs, -50.0, True, False, {}

        # Check if close to target in xy-plane to initiate landing
        xy_dist = np.sqrt((self.pos[0] - self.goal[0])**2 + (self.pos[1] - self.goal[1])**2)
        if xy_dist < 0.5 and not self.is_landing:
            print(f"Close to target in xy: {xy_dist}m, initiating landing")
            self.is_landing = True

        # Landing logic
        if self.is_landing:
            msg = TrajectorySetpoint()
            z_error = self.goal[2] - self.pos[2]  # Target z=0.0
            vz = 0.5 * z_error
            vz = max(min(vz, -0.5), -1.0)  # Limit descent speed
            msg.velocity = [0.0, 0.0, float(vz)]
            msg.position = [float('nan'), float('nan'), self.goal[2]]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(5):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.1)
            print(f"Landing - Pos: {self.pos[:3]}, Vel: [0, 0, {vz}], z_error: {z_error}")

            # Check if landed (z close to 0.0)
            if abs(self.pos[2] - self.goal[2]) < 0.1 and not self.has_landed:
                print("Drone has landed, disarming")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)  # Disarm
                self.has_landed = True
                obs = self._get_obs()
                if obs.shape != (44,):
                    raise ValueError(f"Observation shape {obs.shape} does not match expected (44,)")
                return obs, 500.0, True, False, {}  # Large reward for landing and disarming

        # Navigation logic (if not landing)
        else:
            msg = TrajectorySetpoint()
            z_error = self.takeoff_height - self.pos[2]
            vz = 0.5 * z_error
            if corrective_action:
                msg.velocity = [corrective_action[0], corrective_action[1], float(vz)]
            else:
                scaled_action = action * 2.0  # Scale action for faster movement
                msg.velocity = [float(scaled_action[0]), float(scaled_action[1]), float(vz)]
            msg.position = [float('nan'), float('nan'), self.takeoff_height]
            msg.yaw = float(self.get_yaw_to_target())
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            for _ in range(5):
                self.traj_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.1)
            print(f"Applying action: velocity={msg.velocity}, position={msg.position}, yaw={msg.yaw}")

        obs = self._get_obs()
        if obs.shape != (44,):
            raise ValueError(f"Observation shape {obs.shape} does not match expected (44,)")
        reward = self._compute_reward()
        terminated = self._check_done()
        truncated = self.step_count >= self.max_steps

        print(f"Terminated: {terminated}, Truncated: {truncated}")
        return obs, reward, bool(terminated), bool(truncated), {}

    def _get_obs(self):
        dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
        goal_dist = np.sqrt(dx**2 + dy**2 + (self.pos[2] - self.goal[2])**2)
        goal_angle = np.arctan2(dy, dx) - np.arctan2(self.pos[4], self.pos[3])
        # Ensure lidar_data is exactly 36 elements
        if self.lidar_data.shape != (36,):
            print(f"Warning: lidar_data shape {self.lidar_data.shape}, adjusting to (36,)")
            self.lidar_data = self.lidar_data[:36] if len(self.lidar_data) > 36 else np.pad(self.lidar_data, (0, 36 - len(self.lidar_data)), constant_values=self.max_range)
        obs = np.concatenate([self.pos, [goal_dist, goal_angle], self.lidar_data], axis=0)
        print(f"Observation - Goal dist: {goal_dist:.2f}, Goal angle: {goal_angle:.2f}, LIDAR min: {np.min(self.lidar_data):.2f}, shape={obs.shape}")
        return obs

    def _compute_reward(self):
        dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
        goal_dist = np.sqrt(dx**2 + dy**2 + (self.pos[2] - self.goal[2])**2)
        min_dist = np.min(self.lidar_data)

        # Base reward: penalize goal distance and reward speed
        reward = -0.5 * goal_dist
        speed = np.linalg.norm(self.pos[3:5])  # xy velocity magnitude
        reward += 2.0 * speed  # Encourage faster movement

        # Strong penalty for close obstacles
        if min_dist < 1.0:
            reward -= 50.0 * (1.0 - min_dist)  # Exponential penalty for closer obstacles
        elif min_dist > 2.0:
            reward += 5.0  # Bonus for safe distance

        # Bonus for reaching goal in xy-plane
        xy_dist = np.sqrt(dx**2 + dy**2)
        if xy_dist < 0.5:
            reward += 100.0

        # Bonus for being at correct altitude during landing
        if self.is_landing and abs(self.pos[2] - self.goal[2]) < 0.1:
            reward += 200.0

        # Progress reward
        if self.prev_goal_dist is not None:
            dist_diff = self.prev_goal_dist - goal_dist
            reward += 5.0 * dist_diff  # Larger reward for progress
        self.prev_goal_dist = goal_dist

        print(f"Reward components - Goal dist: {goal_dist:.2f}, Min dist: {min_dist:.2f}, Speed: {speed:.2f}, Reward: {reward:.2f}")
        return reward

    def _check_done(self):
        xy_dist = np.sqrt((self.pos[0] - self.goal[0])**2 + (self.pos[1] - self.goal[1])**2)
        min_dist = np.min(self.lidar_data)
        altitude_diff = abs(self.pos[2] - self.goal[2])
        goal_condition = xy_dist < 0.5 and altitude_diff < 0.1  # Includes landing
        obstacle_condition = min_dist < 0.5  # Tighter threshold for safety
        print(f"Done conditions - XY dist: {xy_dist:.2f}, Min dist: {min_dist:.2f}, Altitude diff: {altitude_diff:.2f}")
        return goal_condition or obstacle_condition or self.has_landed

    def get_yaw_to_target(self):
        dx = self.goal[0] - self.pos[0]
        dy = self.goal[1] - self.pos[1]
        return np.arctan2(dy, dx)