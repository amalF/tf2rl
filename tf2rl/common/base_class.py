"""
Defines base classes for algorithms 
"""
from abc import ABC, abstractmethod
from typing import Dict, Callable
import tensorflow as tf
import numpy as np
import gym


class BasePG(ABC):
    def __init__(self, policy: Callable, env: Callable, critic_lr=1e-3, pi_lr=3e-4):
        self.policy = policy
        self.env = env
        self.action_space = self.env.action_space
        self.obs_space = self.env.observation_space
        self.critic_lr = critic_lr
        self.pi_lr = pi_lr
        self.critic_opt = tf.keras.optimizers.Adam(
            learning_rate=self.critic_lr)
        self.pi_opt = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)

    def set_random_seed(self, seed=None):
        """Set the seed for python random, numpy, tf, gym env"""
        if seed is None:
            return
        tf.random.set_seed(seed)
        np.random.seed(seed)
        # random.seed(self.seed)
        if self.env is not None:
            self.env.seed(seed)
            self.env.action_space.seed(seed)

        if self.action_space is not None:
            self.action_space.seed(seed)

    # @abstractmethod
    def compute_pi_loss(self, states: np.ndarray, *args, **kwargs):
        """
        Defines the policy loss
        """
        raise NotImplementedError

    def compute_critic_loss(self,
                            inputs: tf.Tensor,
                            targets: tf.Tensor) -> tf.Tensor:
        """Defines the critic loss"""
        vals = self.agent.critic(inputs)
        critic_loss = 0.5*tf.reduce_mean((targets - vals)**2)
        return critic_loss

    @tf.function
    def critic_train_step(self,
                          inputs: tf.Tensor,
                          targets: tf.Tensor) -> tf.Tensor:
        """
        Defines critic train step
        """
        with tf.GradientTape() as critic_tape:
            critic_loss = self.compute_critic_loss(inputs, targets)

        critic_grads = critic_tape.gradient(
            critic_loss, self.agent.critic_vars)
        self.critic_opt.apply_gradients(
            zip(critic_grads, self.agent.critic_vars))
        return critic_loss

    @tf.function
    def pi_train_step(self, *args, **kwargs) -> Dict[str, tf.Tensor]:
        """Defines policy train step"""

        with tf.GradientTape() as pi_tape:
            pi_info = self.compute_pi_loss(*args, **kwargs)
        pi_grads = pi_tape.gradient(pi_info['LossPi'], self.agent.policy_vars)
        self.pi_opt.apply_gradients(zip(pi_grads, self.agent.policy_vars))

        return pi_info

    # @abstractmethod
    def _train_step(self, *args, **kwargs):
        raise NotImplementedError

    # @abstractmethod
    def learn(self, timesteps):
        raise NotImplementedError


class BasePolicy(ABC):
    """
    Defines a base policy agent
    """

    def __init__(self,
                 obs_space: gym.spaces,
                 act_space: gym.spaces,
                 name="BasePolicy"):

        self.obs_space = obs_space
        self.act_space = act_space
        self.name = name

    @abstractmethod
    def step(self, obs, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def act(self, obs):
        raise NotImplementedError

    # @abstractmethod
    # def action_prob(self, obs):
    #    raise NotImplementedError
