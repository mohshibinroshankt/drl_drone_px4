#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from drl_px4.uav_env_maze import UAVEnv
from drl_px4.sac_per_maze import SAC_PER
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize ROS2
        rclpy.init()
        logger.info("ROS2 initialized")

        # Create UAV environment
        env = UAVEnv()
        logger.info("UAVEnv initialized")

        # Load the trained SAC-PER model
        model_path = "sac_per_maze_checkpoint_20250509_120935_10000.zip"
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} not found")
            return 1
        model = SAC_PER.load(model_path)
        logger.info(f"Loaded SAC-PER model from {model_path}")

        # Reset environment
        obs, _ = env.reset()
        logger.info(f"Environment reset, starting episode at position: {obs.get('position', 'N/A')}")

        done = False
        step_count = 0
        max_steps = 500
        initial_steps_to_ignore_collision = 10  # Ignore collisions for the first 10 steps

        while not done:
            # Predict action using the model
            action, _ = model.predict(obs, deterministic=True)
            logger.debug(f"Step {step_count}: Action {action}")

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {step_count}: Goal dist: {obs['goal_dist']}, LIDAR: {obs['lidar']}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

            # Check termination conditions
            if step_count < initial_steps_to_ignore_collision:
                # Ignore collisions during initial steps to allow stabilization
                terminated = False
                logger.debug(f"Step {step_count}: Ignoring collision check")

            done = terminated or truncated
            step_count += 1

            if step_count >= max_steps:
                logger.info("Reached maximum steps, terminating episode")
                done = True

            if terminated:
                logger.info("Episode terminated due to collision or goal reached")
            if truncated:
                logger.info("Episode truncated")

        logger.info(f"Episode completed after {step_count} steps")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        return 1

    finally:
        # Clean up
        try:
            env.destroy_node()
            logger.info("UAVEnv node destroyed")
        except:
            pass
        rclpy.shutdown()
        logger.info("ROS2 shutdown")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())