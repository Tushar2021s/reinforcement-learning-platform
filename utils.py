import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from collections import deque
import inspect
import math


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_agent_params(agent):
    param_dict = {}

    signature = inspect.signature(agent.__init__)

    for k, v in signature.parameters.items():

        if k not in ['self', 'env']:

            default = v.default

            if default is inspect.Parameter.empty:
                continue

            if isinstance(default, bool):
                param_dict[k] = float(default)

            elif isinstance(default, (int, float)):
                param_dict[k] = float(default)

            else:
                try:
                    param_dict[k] = float(default)
                except (ValueError, TypeError):
                    param_dict[k] = default

    return param_dict


# ============================================================
# SUM TREE
# ============================================================

class SumTree:

    write = 0

    def __init__(self, capacity):

        self.capacity = int(capacity)

        self.tree = np.zeros(
            2 * self.capacity - 1
        )

        self.data = np.zeros(
            self.capacity,
            dtype=object
        )

        self.write = 0

    def _propagate(self, idx, change):

        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(
                parent,
                change
            )

    def _retrieve(self, idx, s):

        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:

            return self._retrieve(
                left,
                s
            )

        else:

            return self._retrieve(
                right,
                s - self.tree[left]
            )

    def total(self):

        return self.tree[0]

    def add(self, p, data):

        idx = (
            self.write
            + self.capacity
            - 1
        )

        self.data[self.write] = data

        self.update(
            idx,
            p
        )

        self.write += 1

        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):

        change = p - self.tree[idx]

        self.tree[idx] = p

        self._propagate(
            idx,
            change
        )

    def get(self, s):

        idx = self._retrieve(
            0,
            s
        )

        data_idx = (
            idx
            - self.capacity
            + 1
        )

        return (
            idx,
            self.tree[idx],
            self.data[data_idx]
        )


# ============================================================
# BASIC REPLAY BUFFER
# ============================================================

class BasicBuffer:

    def __init__(self, max_size):

        self.max_size = int(max_size)

        self.buffer = deque(
            maxlen=self.max_size
        )

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            np.asarray(state, dtype=np.float32),
            action,
            np.array([reward], dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            float(done)
        )

        self.buffer.append(
            experience
        )

    def sample(self, batch_size):

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(
            self.buffer,
            batch_size
        )

        for experience in batch:

            (
                state,
                action,
                reward,
                next_state,
                done
            ) = experience

            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch
        )

    def sample_sequence(self, batch_size):

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        if len(self.buffer) < batch_size:
            raise ValueError(
                "Not enough samples in replay buffer."
            )

        max_start = len(self.buffer) - batch_size

        if max_start == 0:
            start = 0
        else:
            start = np.random.randint(
                0,
                max_start + 1
            )

        for i in range(
            start,
            start + batch_size
        ):

            (
                state,
                action,
                reward,
                next_state,
                done
            ) = self.buffer[i]

            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# PRIORITIZED EXPERIENCE REPLAY
# ============================================================

class PrioritizedBuffer:

    def __init__(
        self,
        max_size,
        alpha=0.6,
        beta=0.4
    ):

        self.sum_tree = SumTree(
            int(max_size)
        )

        self.alpha = alpha
        self.beta = beta
        self.current_length = 0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        if self.current_length == 0:
            priority = 1.0
        else:
            priority = max(
                self.sum_tree.tree[
                    -self.sum_tree.capacity:
                ].max(),
                1e-6
            )

        experience = (
            np.asarray(state, dtype=np.float32),
            action,
            np.array([reward], dtype=np.float32),
            np.asarray(next_state, dtype=np.float32),
            float(done)
        )

        self.sum_tree.add(
            priority,
            experience
        )

        self.current_length = min(
            self.current_length + 1,
            self.sum_tree.capacity
        )

    def sample(self, batch_size):

        batch_idx = []
        batch = []
        IS_weights = []

        total_priority = self.sum_tree.total()

        if total_priority <= 0:
            raise ValueError(
                "Total priority must be greater than zero."
            )

        segment = (
            total_priority
            / batch_size
        )

        for i in range(batch_size):

            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(
                a,
                b
            )

            idx, p, data = (
                self.sum_tree.get(s)
            )

            batch_idx.append(idx)
            batch.append(data)

            prob = p / total_priority

            IS_weight = (
                self.current_length
                * prob
            ) ** (-self.beta)

            IS_weights.append(
                IS_weight
            )

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for transition in batch:

            (
                state,
                action,
                reward,
                next_state,
                done
            ) = transition

            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                done_batch
            ),
            batch_idx,
            IS_weights
        )

    def update_priority(
        self,
        idx,
        td_error
    ):

        td_error = abs(
            float(td_error)
        ) + 1e-6

        priority = (
            td_error
            ** self.alpha
        )

        self.sum_tree.update(
            idx,
            priority
        )

    def __len__(self):

        return self.current_length


# ============================================================
# NOISY LINEAR
# ============================================================

class NoisyLinear(nn.Module):

    def __init__(
        self,
        num_in,
        num_out,
        is_training=True
    ):

        super(
            NoisyLinear,
            self
        ).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training

        self.mu_weight = nn.Parameter(
            torch.FloatTensor(
                num_out,
                num_in
            )
        )

        self.mu_bias = nn.Parameter(
            torch.FloatTensor(
                num_out
            )
        )

        self.sigma_weight = nn.Parameter(
            torch.FloatTensor(
                num_out,
                num_in
            )
        )

        self.sigma_bias = nn.Parameter(
            torch.FloatTensor(
                num_out
            )
        )

        self.register_buffer(
            "epsilon_weight",
            torch.FloatTensor(
                num_out,
                num_in
            )
        )

        self.register_buffer(
            "epsilon_bias",
            torch.FloatTensor(
                num_out
            )
        )

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):

        self.reset_noise()

        if self.is_training:

            weight = (
                self.mu_weight
                + self.sigma_weight
                * self.epsilon_weight
            )

            bias = (
                self.mu_bias
                + self.sigma_bias
                * self.epsilon_bias
            )

        else:

            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(
            x,
            weight,
            bias
        )

    def reset_parameters(self):

        std = math.sqrt(
            3 / self.num_in
        )

        self.mu_weight.data.uniform_(
            -std,
            std
        )

        self.mu_bias.data.uniform_(
            -std,
            std
        )

        self.sigma_weight.data.fill_(
            0.017
        )

        self.sigma_bias.data.fill_(
            0.017
        )

    def reset_noise(self):

        self.epsilon_weight.data.normal_()
        self.epsilon_bias.data.normal_()


# ============================================================
# FACTORIZED NOISY LINEAR
# ============================================================

class FactorizedNoisyLinear(nn.Module):

    def __init__(
        self,
        num_in,
        num_out,
        is_training=True
    ):

        super(
            FactorizedNoisyLinear,
            self
        ).__init__()

        self.num_in = num_in
        self.num_out = num_out
        self.is_training = is_training

        self.mu_weight = nn.Parameter(
            torch.FloatTensor(
                num_out,
                num_in
            )
        )

        self.mu_bias = nn.Parameter(
            torch.FloatTensor(
                num_out
            )
        )

        self.sigma_weight = nn.Parameter(
            torch.FloatTensor(
                num_out,
                num_in
            )
        )

        self.sigma_bias = nn.Parameter(
            torch.FloatTensor(
                num_out
            )
        )

        self.register_buffer(
            "epsilon_i",
            torch.FloatTensor(num_in)
        )

        self.register_buffer(
            "epsilon_j",
            torch.FloatTensor(num_out)
        )

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):

        self.reset_noise()

        if self.is_training:

            epsilon_weight = (
                self.epsilon_j.ger(
                    self.epsilon_i
                )
            )

            epsilon_bias = self.epsilon_j

            weight = (
                self.mu_weight
                + self.sigma_weight
                * epsilon_weight
            )

            bias = (
                self.mu_bias
                + self.sigma_bias
                * epsilon_bias
            )

        else:

            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(
            x,
            weight,
            bias
        )

    def reset_parameters(self):

        std = 1 / math.sqrt(
            self.num_in
        )

        self.mu_weight.data.uniform_(
            -std,
            std
        )

        self.mu_bias.data.uniform_(
            -std,
            std
        )

        self.sigma_weight.data.fill_(
            0.5 / math.sqrt(
                self.num_in
            )
        )

        self.sigma_bias.data.fill_(
            0.5 / math.sqrt(
                self.num_in
            )
        )

    def reset_noise(self):

        eps_i = torch.randn(
            self.num_in
        )

        eps_j = torch.randn(
            self.num_out
        )

        self.epsilon_i = (
            eps_i.sign()
            * eps_i.abs().sqrt()
        )

        self.epsilon_j = (
            eps_j.sign()
            * eps_j.abs().sqrt()
        )


# ============================================================
# ORNSTEIN-UHLENBECK NOISE
# ============================================================

class OUNoise:

    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000
    ):

        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

        self.action_dim = (
            action_space.shape[0]
        )

        self.low = action_space.low
        self.high = action_space.high

        self.reset()

    def reset(self):

        self.state = (
            np.ones(self.action_dim)
            * self.mu
        )

    def evolve_state(self):

        x = self.state

        dx = (
            self.theta
            * (self.mu - x)
            + self.sigma
            * np.random.randn(
                self.action_dim
            )
        )

        self.state = x + dx

        return self.state

    def get_action(
        self,
        action,
        t=0
    ):

        ou_state = (
            self.evolve_state()
        )

        self.sigma = (
            self.max_sigma
            - (
                self.max_sigma
                - self.min_sigma
            )
            * min(
                1.0,
                t / self.decay_period
            )
        )

        return np.clip(
            action + ou_state,
            self.low,
            self.high
        )


# ============================================================
# GYM COMPATIBILITY HELPERS
# ============================================================

def reset_env(env):

    """
    Works with both old Gym and modern Gym.
    """

    result = env.reset()

    if isinstance(result, tuple):
        state = result[0]
    else:
        state = result

    return state


def step_env(env, action):

    """
    Works with both old Gym and modern Gym.
    """

    result = env.step(action)

    if len(result) == 5:

        (
            next_state,
            reward,
            terminated,
            truncated,
            info
        ) = result

        done = (
            terminated
            or truncated
        )

    else:

        (
            next_state,
            reward,
            done,
            info
        ) = result

    return (
        next_state,
        reward,
        done,
        info
    )


def render_env(
    env,
    render_area=None
):

    """
    Render the environment.

    The environment should be created using:

        gym.make(
            env_name,
            render_mode="rgb_array"
        )
    """

    if render_area is None:
        return

    try:

        frame = env.render()

        if frame is not None:

            render_area.image(
                frame
            )

    except Exception as e:

        print(
            "Render warning:",
            e
        )


# ============================================================
# SCORE DISPLAY HELPER
# ============================================================

def update_score_chart(
    score_area,
    episode_rewards
):

    """
    Modern Streamlit replacement for the old
    score_area.add_rows() API.
    """

    if score_area is None:
        return

    if len(episode_rewards) == 0:
        return

    rolling_rewards = []

    for i in range(len(episode_rewards)):

        rolling_rewards.append(
            np.mean(
                episode_rewards[
                    max(0, i - 9): i + 1
                ]
            )
        )

    score_df = pd.DataFrame({
        "reward": episode_rewards,
        "rolling_reward": rolling_rewards
    })

    score_area.line_chart(
        score_df
    )


# ============================================================
# OFF-POLICY TRAINING
# DQN / DDPG / TD3
# ============================================================

def single_step_update_train(
    env,
    agent,
    max_episodes,
    max_steps,
    batch_size,
    render_area,
    score_area,
    progress_bar
):

    episode_rewards = []

    for episode in range(
        int(max_episodes)
    ):

        state = reset_env(env)

        episode_reward = 0

        for step in range(
            int(max_steps)
        ):

            render_env(
                env,
                render_area
            )

            action = agent.get_action(
                state
            )

            (
                next_state,
                reward,
                done,
                info
            ) = step_env(
                env,
                action
            )

            agent.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            episode_reward += reward

            if (
                len(agent.replay_buffer)
                >= int(batch_size)
            ):

                agent.update(
                    int(batch_size)
                )

            state = next_state

            if (
                done
                or step == int(max_steps) - 1
            ):

                episode_rewards.append(
                    episode_reward
                )

                print(
                    "Episode "
                    + str(episode + 1)
                    + ": "
                    + str(episode_reward)
                )

                # Modern Streamlit chart update
                update_score_chart(
                    score_area,
                    episode_rewards
                )

                if progress_bar is not None:

                    progress_bar.progress(
                        (episode + 1)
                        / int(max_episodes)
                    )

                break

    return (
        episode_rewards,
        agent
    )


# ============================================================
# ON-POLICY TRAINING
# A2C / PPO
# ============================================================

def episode_update_train(
    env,
    agent,
    max_episodes,
    max_steps,
    render_area,
    score_area,
    progress_bar
):

    episode_rewards = []

    for episode in range(
        int(max_episodes)
    ):

        state = reset_env(env)

        episode_reward = 0

        trajectory = []

        for step in range(
            int(max_steps)
        ):

            render_env(
                env,
                render_area
            )

            action = agent.get_action(
                state
            )

            (
                next_state,
                reward,
                done,
                info
            ) = step_env(
                env,
                action
            )

            trajectory.append(
                [
                    state,
                    action,
                    reward,
                    next_state,
                    done
                ]
            )

            episode_reward += reward

            state = next_state

            if (
                done
                or step == int(max_steps) - 1
            ):

                episode_rewards.append(
                    episode_reward
                )

                print(
                    "Episode "
                    + str(episode + 1)
                    + ": "
                    + str(episode_reward)
                )

                agent.update(
                    trajectory
                )

                # Modern Streamlit chart update
                update_score_chart(
                    score_area,
                    episode_rewards
                )

                if progress_bar is not None:

                    progress_bar.progress(
                        (episode + 1)
                        / int(max_episodes)
                    )

                break

    return (
        episode_rewards,
        agent
    )