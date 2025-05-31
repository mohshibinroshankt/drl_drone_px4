import rclpy
from drl_px4.uav_env_maze import UAVEnv
from drl_px4.sac_per_maze import SAC_PER
from datetime import datetime

def get_timestamp():
    """Return a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    rclpy.init()
    try:
        env = UAVEnv()
        print("Environment created")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        model = SAC_PER(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log="./tb_logs/",
            ent_coef="auto_0.1",
            learning_rate=0.0003,  # Adjusted for maze solving
            buffer_size=200000,    # Increased for complex maze
        )
        
        total_timesteps = 500000  # Increased for maze solving
        checkpoint_interval = 10000
        try:
            for step in range(0, total_timesteps, checkpoint_interval):
                model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
                timestamp = get_timestamp()
                model.save(f"sac_per_maze_checkpoint_{timestamp}_{step + checkpoint_interval}")
                print(f"Saved checkpoint at timestep {step + checkpoint_interval}: sac_per_maze_checkpoint_{timestamp}_{step + checkpoint_interval}.zip")
        except KeyboardInterrupt:
            timestamp = get_timestamp()
            print("Training interrupted by user")
            model.save(f"sac_per_maze_interrupted_{timestamp}")
            print(f"Saved model before exiting: sac_per_maze_interrupted_{timestamp}.zip")
            raise
        except RuntimeError as e:
            timestamp = get_timestamp()
            print(f"RuntimeError in environment: {e}")
            model.save(f"sac_per_maze_error_{timestamp}")
            print(f"Saved model before exiting due to error: sac_per_maze_error_{timestamp}.zip")
            raise
        except Exception as e:
            timestamp = get_timestamp()
            print(f"Unexpected error: {e}")
            model.save(f"sac_per_maze_error_{timestamp}")
            print(f"Saved model before exiting due to unexpected error: sac_per_maze_error_{timestamp}.zip")
            raise
        
        timestamp = get_timestamp()
        model.save(f"sac_per_maze_final_{timestamp}")
        print(f"Training completed, saved final model: sac_per_maze_final_{timestamp}.zip")
        
    finally:
        if 'env' in locals():
            env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()