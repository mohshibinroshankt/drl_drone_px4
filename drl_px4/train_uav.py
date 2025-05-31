import rclpy
import torch
from drl_px4.uav_env import UAVEnv
from drl_px4.sac_per import SAC_PER
from datetime import datetime

def get_timestamp():
    """Return a timestamp string in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    rclpy.init()
    env = None
    try:
        try:
            env = UAVEnv()
            print("Environment created")
            print(f"Observation space: {env.observation_space}")
            print(f"Action space: {env.action_space}")
        except Exception as e:
            print(f"Failed to initialize environment: {e}")
            raise

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        timestamp = get_timestamp()
        tensorboard_run = f"sac_per_uav_{timestamp}"
        model = SAC_PER(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log="./tb_logs/",
            ent_coef="auto_0.1",
            learning_rate=0.0005,
            policy_kwargs=dict(net_arch=[128, 128]),
            buffer_size=1000000,
            alpha=0.6,
            beta=0.4,
            beta_increment=1e-3,  # Add this to match SAC_PER signature
        )

        total_timesteps = 500000
        checkpoint_interval = 10000
        try:
            for step in range(0, total_timesteps, checkpoint_interval):
                model.learn(
                    total_timesteps=checkpoint_interval,
                    reset_num_timesteps=False,
                    log_interval=1,
                    tb_log_name=tensorboard_run,
                )
                timestamp = get_timestamp()
                checkpoint_path = f"sac_per_uav_checkpoint_{timestamp}_{step + checkpoint_interval}"
                model.save(checkpoint_path)
                print(f"Saved checkpoint at timestep {step + checkpoint_interval}: {checkpoint_path}.zip")
        except KeyboardInterrupt:
            timestamp = get_timestamp()
            interrupted_path = f"sac_per_uav_interrupted_{timestamp}"
            print("Training interrupted by user")
            model.save(interrupted_path)
            print(f"Saved model before exiting: {interrupted_path}.zip")
            raise
        except RuntimeError as e:
            timestamp = get_timestamp()
            error_path = f"sac_per_uav_error_{timestamp}"
            print(f"RuntimeError in environment: {e}")
            model.save(error_path)
            print(f"Saved model before exiting due to error: {error_path}.zip")
            raise
        except Exception as e:
            timestamp = get_timestamp()
            error_path = f"sac_per_uav_error_{timestamp}"
            print(f"Unexpected error: {e}")
            model.save(error_path)
            print(f"Saved model before exiting due to unexpected error: {error_path}.zip")
            raise

        timestamp = get_timestamp()
        final_path = f"sac_per_uav_final_{timestamp}"
        model.save(final_path)
        print(f"Training completed, saved final model: {final_path}.zip")

    finally:
        if env is not None:
            env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()