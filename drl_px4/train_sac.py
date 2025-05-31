import rclpy
from drl_px4.drone_env import DroneEnv
from drl_px4.per_sac import SAC_PER
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def get_timestamp():
    """Return a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%SS")

def check_observation_shape(obs, expected_shape):
    """Check if the observation shape matches the expected shape."""
    if obs.shape != expected_shape:
        raise ValueError(f"Observation shape {obs.shape} does not match expected {expected_shape}")

def main():
    rclpy.init()
    writer = SummaryWriter(log_dir=f"./tb_logs/run_{get_timestamp()}")
    try:
        env = DroneEnv()
        print("Environment created")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Verify observation shape
        obs, _ = env.reset()
        check_observation_shape(obs, (44,))
        
        model = SAC_PER(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log="./tb_logs/",
            ent_coef="auto_0.1",
            learning_rate=0.0005,
            buffer_size=100000
        )
        
        total_timesteps = 100000
        checkpoint_interval = 10000
        episode_count = 0
        episode_rewards = []
        episode_distances = []
        episode_min_lidar = []
        episode_successes = 0
        global_step = 0

        try:
            for step in range(0, total_timesteps, checkpoint_interval):
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0
                episode_min_dist = float('inf')
                done = False
                truncated = False

                while not (done or truncated):
                    # Verify observation shape before prediction
                    check_observation_shape(obs, (44,))
                    
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_steps += 1
                    global_step += 1
                    episode_min_dist = min(episode_min_dist, np.min(env.lidar_data))

                    # Log step metrics
                    xy_dist = np.sqrt((env.pos[0] - env.goal[0])**2 + (env.pos[1] - env.goal[1])**2)
                    writer.add_scalar('Step/Reward', reward, global_step)
                    writer.add_scalar('Step/Distance_to_Goal', xy_dist, global_step)
                    writer.add_scalar('Step/Min_LIDAR_Distance', np.min(env.lidar_data), global_step)

                    if done or truncated:
                        episode_count += 1
                        episode_rewards.append(episode_reward)
                        episode_distances.append(xy_dist)
                        episode_min_lidar.append(episode_min_dist)
                        if env.has_landed and xy_dist < 0.5:
                            episode_successes += 1

                        # Log episode metrics
                        writer.add_scalar('Episode/Reward', episode_reward, episode_count)
                        writer.add_scalar('Episode/Distance_to_Goal', xy_dist, episode_count)
                        writer.add_scalar('Episode/Min_LIDAR_Distance', episode_min_dist, episode_count)
                        writer.add_scalar('Episode/Success_Rate', episode_successes / episode_count, episode_count)
                        print(f"Episode {episode_count}: Reward={episode_reward:.2f}, Steps={episode_steps}, "
                              f"Distance={xy_dist:.2f}, Min LIDAR={episode_min_dist:.2f}, Success={env.has_landed}")
                        break

                # Learn for checkpoint_interval timesteps
                model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
                timestamp = get_timestamp()
                model.save(f"sac_per_drone_checkpoint_{timestamp}_{step + checkpoint_interval}")
                print(f"Saved checkpoint at timestep {step + checkpoint_interval}: sac_per_drone_checkpoint_{timestamp}_{step + checkpoint_interval}.zip")

        except KeyboardInterrupt:
            timestamp = get_timestamp()
            print("Training interrupted by user")
            model.save(f"sac_per_drone_interrupted_{timestamp}")
            print(f"Saved model before exiting: sac_per_drone_interrupted_{timestamp}.zip")
            raise
        except RuntimeError as e:
            timestamp = get_timestamp()
            print(f"RuntimeError in environment: {e}")
            model.save(f"sac_per_drone_error_{timestamp}")
            print(f"Saved model before exiting due to error: sac_per_drone_error_{timestamp}.zip")
            raise
        except ValueError as e:
            timestamp = get_timestamp()
            print(f"ValueError: {e}")
            model.save(f"sac_per_drone_error_{timestamp}")
            print(f"Saved model before exiting due to error: sac_per_drone_error_{timestamp}.zip")
            raise
        except Exception as e:
            timestamp = get_timestamp()
            print(f"Unexpected error: {e}")
            model.save(f"sac_per_drone_error_{timestamp}")
            print(f"Saved model before exiting due to unexpected error: sac_per_drone_error_{timestamp}.zip")
            raise
        
        timestamp = get_timestamp()
        model.save(f"sac_per_drone_final_{timestamp}")
        print(f"Training completed, saved final model: sac_per_drone_final_{timestamp}.zip")
        
    finally:
        writer.close()
        if 'env' in locals():
            env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()