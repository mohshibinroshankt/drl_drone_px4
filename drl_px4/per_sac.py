import numpy as np
import torch as th
import torch.optim as optim
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from typing import Dict, Tuple, Union, Optional

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
        weights /= weights.max()

        samples = super()._get_samples(indices, env=env)
        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.abs(priorities) + 1e-6
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))

    def update_beta(self):
        self.beta = min(1.0, self.beta + self.beta_increment)

class SAC_PER(SAC):
    def __init__(self, *args, buffer_size: int = 100000, **kwargs):
        if 'total_timesteps' in kwargs:
            self._total_timesteps = kwargs.pop('total_timesteps')
        else:
            self._total_timesteps = 1000000

        print("SAC_PER.__init__ called with args:", args)
        print("SAC_PER.__init__ called with kwargs:", kwargs)
        super().__init__(*args, **kwargs)
        print("After super().__init__")
        print(f"ent_coef after super().__init__: {self.ent_coef}, type: {type(self.ent_coef)}, shape: {self.ent_coef.shape if isinstance(self.ent_coef, th.Tensor) else 'N/A'}")
        if hasattr(self, 'log_ent_coef'):
            print(f"log_ent_coef: {self.log_ent_coef}, type: {type(self.log_ent_coef)}, shape: {self.log_ent_coef.shape if isinstance(self.log_ent_coef, th.Tensor) else 'N/A'}")
        print(f"policy: {hasattr(self, 'policy')}")
        print(f"policy_class: {self.policy_class if hasattr(self, 'policy_class') else 'Not set'}")
        print(f"critic: {hasattr(self, 'critic')}")
        print(f"actor: {hasattr(self, 'actor')}")
        print(f"critic_optimizer: {hasattr(self, 'critic_optimizer')}")
        print(f"actor_optimizer: {hasattr(self, 'actor_optimizer')}")
        print(f"ent_coef_optimizer: {hasattr(self, 'ent_coef_optimizer')}")
        if not hasattr(self, 'policy_class'):
            raise AttributeError("policy_class not set after super().__init__")

        self.buffer_size = buffer_size
        self._n_updates = 0
        print("SAC_PER initialized")

    def _setup_model(self) -> None:
        print("Calling SAC_PER._setup_model")
        super()._setup_model()
        print("After super()._setup_model()")
        print(f"policy: {hasattr(self, 'policy')}")
        print(f"policy type: {type(self.policy) if hasattr(self, 'policy') else 'Not set'}")
        print(f"critic: {hasattr(self, 'critic')}")
        print(f"actor: {hasattr(self, 'actor')}")
        print(f"critic_optimizer: {hasattr(self, 'critic_optimizer')}")
        print(f"actor_optimizer: {hasattr(self, 'actor_optimizer')}")
        print(f"ent_coef_optimizer: {hasattr(self, 'ent_coef_optimizer')}")

        if hasattr(self, 'critic'):
            critic_params = list(self.critic.parameters())
            print(f"Critic parameters: {len(critic_params)}")
        if hasattr(self, 'actor'):
            actor_params = list(self.actor.parameters())
            print(f"Actor parameters: {len(actor_params)}")

        if not hasattr(self, 'critic_optimizer') and hasattr(self, 'critic'):
            print("Manually initializing critic_optimizer")
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_schedule(1))
            print(f"critic_optimizer initialized: {self.critic_optimizer}")
        if not hasattr(self, 'actor_optimizer') and hasattr(self, 'actor'):
            print("Manually initializing actor_optimizer")
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_schedule(1))
            print(f"actor_optimizer initialized: {self.actor_optimizer}")

        if not hasattr(self, 'critic_optimizer'):
            raise AttributeError("critic_optimizer not initialized after manual setup")
        if not hasattr(self, 'actor_optimizer'):
            raise AttributeError("actor_optimizer not initialized after manual setup")

        print(f"ent_coef in _setup_model: {self.ent_coef}, type: {type(self.ent_coef)}, shape: {self.ent_coef.shape if isinstance(self.ent_coef, th.Tensor) else 'N/A'}")
        if hasattr(self, 'log_ent_coef'):
            print(f"log_ent_coef in _setup_model: {self.log_ent_coef}, type: {type(self.log_ent_coef)}, shape: {self.log_ent_coef.shape if isinstance(self.log_ent_coef, th.Tensor) else 'N/A'}")

        self.replay_buffer = PrioritizedReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            alpha=0.6,
            beta=0.4,
            beta_increment=1e-3,
        )
        print(f"Replay buffer type: {type(self.replay_buffer)}")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self._n_updates += gradient_steps
        self.replay_buffer.update_beta()

        gradient_steps_per_timestep = 4
        total_gradient_steps = gradient_steps * gradient_steps_per_timestep

        for _ in range(total_gradient_steps):
            replay_data, indices, weights = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            weights = th.tensor(weights, device=self.device, dtype=th.float32)

            with th.no_grad():
                next_q_values = self.critic_target(replay_data.next_observations, replay_data.actions)
                next_q_values = th.stack(next_q_values, dim=0)
                next_q_values = th.min(next_q_values, dim=0)[0]
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            current_q_values = th.stack(current_q_values, dim=0)
            current_q_values = th.min(current_q_values, dim=0)[0]

            td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
            td_errors = np.squeeze(td_errors)
            self.replay_buffer.update_priorities(indices, td_errors)
            print(f"Average TD-error: {np.mean(np.abs(td_errors))}")

            self.critic_optimizer.zero_grad()
            q_values = self.critic(replay_data.observations, replay_data.actions)
            q_values = th.stack(q_values, dim=0)
            q_values = th.min(q_values, dim=0)[0]
            critic_loss = ((q_values - target_q_values) ** 2 * weights).mean()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss = self._compute_actor_loss(replay_data.observations)
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss = self._compute_ent_coef_loss(replay_data.observations)
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

    def _compute_actor_loss(self, obs):
        actions, log_prob = self.actor.action_log_prob(obs)
        q_values = self.critic(obs, actions)
        q_values = th.stack(q_values, dim=0)
        q_values = th.min(q_values, dim=0)[0]
        print(f"obs shape: {obs.shape}")
        print(f"actions shape: {actions.shape}, type: {actions.dtype}")
        print(f"log_prob shape: {log_prob.shape}, type: {log_prob.dtype}")
        print(f"q_values shape: {q_values.shape}, type: {q_values.dtype}")
        print(f"ent_coef: {self.ent_coef}, type: {type(self.ent_coef)}, shape: {self.ent_coef.shape if isinstance(self.ent_coef, th.Tensor) else 'N/A'}")

        if isinstance(self.ent_coef, str) and "auto" in self.ent_coef:
            min_entropy = float(self.ent_coef.split("_")[1]) if "_" in self.ent_coef else 0.1
            ent_coef_value = th.exp(self.log_ent_coef).item() if hasattr(self, 'log_ent_coef') else min_entropy
            ent_coef_tensor = th.tensor(ent_coef_value, device=self.device, dtype=th.float32)
            print(f"Computed ent_coef_tensor from log_ent_coef: {ent_coef_value}, type: {type(ent_coef_value)}")
        else:
            ent_coef_value = self.ent_coef.item() if isinstance(self.ent_coef, th.Tensor) else self.ent_coef
            ent_coef_tensor = th.tensor(ent_coef_value, device=self.device, dtype=th.float32)
            print(f"ent_coef_tensor: {ent_coef_value}, type: {type(ent_coef_value)}")

        if len(log_prob.shape) > 1:
            log_prob = log_prob.squeeze()
            print(f"After squeeze, log_prob shape: {log_prob.shape}")
        if len(q_values.shape) > 1:
            q_values = q_values.squeeze()
            print(f"After squeeze, q_values shape: {q_values.shape}")

        entropy_term = ent_coef_tensor * log_prob
        print(f"entropy_term shape: {entropy_term.shape}, type: {entropy_term.dtype}")
        loss = -(q_values - entropy_term).mean()
        print(f"loss shape before mean: {loss.shape}, type: {loss.dtype}")
        return loss

    def _compute_ent_coef_loss(self, obs):
        _, log_prob = self.actor.action_log_prob(obs)
        if len(log_prob.shape) > 1:
            log_prob = log_prob.squeeze()
        return - (self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()