"""
Implements agents for TD3 algorithms
"""

from typing import Tuple, Dict, List
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym


from tf2rl.common.models import MLPFeatureExtractor
from tf2rl.common.base_class import BasePolicy

tfd = tfp.distributions


class TD3Policy(BasePolicy):
    """
    Defines policy object for TD3-like actor critic agent using
    a  feedforward network
    """

    def __init__(self,
                 obs_space: gym.spaces,
                 act_space: gym.spaces,
                 layers: List[int] = [64, 64],
                 activation: str = "relu",
                 policy_kwargs=None,
                 name="DDPGPolicy"):

        super(TD3Policy, self).__init__(obs_space,
                                        act_space,
                                        name=name)

        self.policy = None
        self.q1_fn = None
        self.q2_fn = None
        self.layers = layers
        self.activation = activation
        self.policy_kwargs = policy_kwargs

    def setup(self, create_pi=True, create_qf=True):
        """Creates actor and critic models"""

        if create_pi:
            pi_net = MLPFeatureExtractor(self.layers,
                                         self.activation,
                                         name="pi_net")

            policy_layer = tf.keras.layers.Dense(self.act_space.shape[0],
                                                 activation="tanh",
                                                 name="pi_net/pi")
            self.policy = tf.keras.Sequential(pi_net.layers+[policy_layer])

        if create_qf:
            q1_net = MLPFeatureExtractor(self.layers,
                                         self.activation,
                                         name="q1_net")

            q1_out = tf.keras.layers.Dense(1,
                                           activation=None,
                                           name="q1_net/qval")
            self.q1_fn = tf.keras.Sequential(q1_net.layers+[q1_out])

            q2_net = MLPFeatureExtractor(self.layers,
                                         self.activation,
                                         name="q2_net")

            q2_out = tf.keras.layers.Dense(1,
                                           activation=None,
                                           name="q2_net/qval")
            self.q2_fn = tf.keras.Sequential(q2_net.layers+[q2_out])

        return self.policy, self.q1_fn, self.q2_fn

    @tf.function
    def step(self, obs: np.ndarray, noise_std=None) -> Tuple[tf.Tensor, Dict]:
        """
        Return a deterministic action bounded between -1 and 1
        """

        action = self.policy(obs)
        if noise_std is not None:
            noise = tf.random.normal(shape=action.shape, stddev=noise_std)
            action += noise

            # we need to clip again to -1 and 1 after applying the noise
            action = tf.clip_by_value(action, -1, 1)

        q1val = self.q1_fn(tf.concat([obs, action], axis=-1))
        q2val = self.q2_fn(tf.concat([obs, action], axis=-1))

        return action, q1val, q2val

    def act(self, state: np.ndarray, noise_std=None) -> tf.Tensor:
        """
        Return an action for a given state
        """
        return self.step(state, noise_std=noise_std)[0]

    @property
    def policy_vars(self) -> List[tf.Tensor]:
        """
        Returns actor trainable variables
        """
        return self.policy.trainable_variables

    @property
    def critic_vars(self) -> List[tf.Tensor]:
        """
        Returns critic trainable variables
        """
        return self.q1_fn.trainable_variables+self.q2_fn.trainable_variables
