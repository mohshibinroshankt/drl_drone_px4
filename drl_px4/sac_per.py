import numpy as np
import torch as th
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Dict, Tuple, Union, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-3,
        max_priority_clip: float = 1e3,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority_clip = max_priority_clip
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, *args, **kwargs) -> None:
        idx = self.pos
        super().add(*args, **kwargs)
        self.priorities[idx] = self.max_priority ** self.alpha

    def sample(self, batch_size: int, env=None) -> Tuple:
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        priorities = np.maximum(priorities, 1e-6)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(probs), batch_size, p=probs)

        weights = (self.buffer_size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = np.clip(weights, 1e-6, 1e6)

        samples = super()._get_samples(indices, env=env)
        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + 1e-6
        priorities = np.clip(priorities, 0, self.max_priority_clip)
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))

    def update_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment)
        logger.info(f"Updated beta: {self.beta}")

class SAC_PER(SAC):
    def __init__(
        self,
        *args,
        buffer_size: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-3,
        **kwargs
    ):
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        kwargs.pop("buffer_size", None)
        kwargs.pop("alpha", None)
        kwargs.pop("beta", None)
        kwargs.pop("beta_increment", None)

        super().__init__(*args, **kwargs)
        self._n_updates = 0
        logger.info("SAC_PER initialized")

    def _setup_model(self) -> None:
        super()._setup_model()
        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            alpha=self.alpha,
            beta=self.beta,
            beta_increment=self.beta_increment,
        )
        logger.info(f"Replay buffer type: {type(self.replay_buffer)}")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self._n_updates += gradient_steps
        self.replay_buffer.update_beta()

        gradient_steps_per_timestep = 4
        total_gradient_steps = gradient_steps * gradient_steps_per_timestep

        for step in range(total_gradient_steps):
            replay_data, indices, weights = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            weights = th.tensor(weights, device=self.device, dtype=th.float32)

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = self.critic_target(replay_data.next_observations, next_actions)
                next_q_values = th.cat(next_q_values, dim=1)  # Shape: (batch_size, 2)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)  # Shape: (batch_size, 1)

                # Ensure shapes are compatible
                next_log_prob = next_log_prob.view(-1, 1)  # Reshape to (batch_size, 1)

                # Handle ent_coef correctly
                ent_coef = self.ent_coef
                if isinstance(self.ent_coef, th.Tensor):
                    ent_coef = self.ent_coef.item()  # Convert to scalar if tensor
                ent_coef_tensor = th.tensor(ent_coef, device=self.device, dtype=th.float32)

                # Compute entropy-regularized Q-values
                entropy_term = ent_coef_tensor * next_log_prob  # Shape: (batch_size, 1)
                next_q_values = next_q_values - entropy_term  # Shape: (batch_size, 1)

                # Compute target Q-values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            current_q_values = th.cat(current_q_values, dim=1)
            current_q_values, _ = th.min(current_q_values, dim=1, keepdim=True)

            td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
            logger.info(f"Average TD-error: {np.mean(np.abs(td_errors))}")

            self.critic_optimizer.zero_grad()
            q_values = self.critic(replay_data.observations, replay_data.actions)
            q_values = th.cat(q_values, dim=1)
            q_values, _ = th.min(q_values, dim=1, keepdim=True)
            critic_loss = ((q_values - target_q_values) ** 2 * weights.unsqueeze(1)).mean()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actions, log_prob = self.actor.action_log_prob(replay_data.observations)
            q_values = self.critic(replay_data.observations, actions)
            q_values = th.cat(q_values, dim=1)
            q_values, _ = th.min(q_values, dim=1, keepdim=True)
            log_prob = log_prob.view(-1, 1)  # Reshape to (batch_size, 1)
            actor_loss = (ent_coef_tensor * log_prob - q_values).mean()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            if self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                _, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.view(-1, 1)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            self._update_target_networks()

    def _update_target_networks(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        logger.debug("Updated target networks")