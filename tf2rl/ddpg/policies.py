"""
Implements agents for DDPG algorithms
"""

from typing import Tuple, Dict, List
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym


from tf2rl.common.models import MLPFeatureExtractor
from tf2rl.common.base_class import BasePolicy

tfd = tfp.distributions


class DDPGPolicy(BasePolicy):
    """
    Defines policy object for DDPG-like actor critic agent using
    a  feedforward network
    """

    def __init__(self,
                 obs_space: gym.spaces,
                 act_space: gym.spaces,
                 layers: List[int] = [64, 64],
                 activation: str = "relu",
                 policy_kwargs=None,
                 name="DDPGPolicy"):

        super(DDPGPolicy, self).__init__(obs_space,
                                         act_space,
                                         name=name)

        self.policy = None
        self.q_fn = None
        self.layers = layers
        self.activation = activation
        self.policy_kwargs = policy_kwargs

        # self.setup()
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
            critic_net = MLPFeatureExtractor(self.layers,
                                             self.activation,
                                             name="q_net")

            q_layer = tf.keras.layers.Dense(1,
                                            activation=None,
                                            name="q_net/qval")
            self.q_fn = tf.keras.Sequential(critic_net.layers+[q_layer])

        return self.policy, self.q_fn

    @tf.function
    def step(self, obs: np.ndarray, noise_std=None) -> Tuple[tf.Tensor, Dict]:
        """
        Return a deterministic action bounded between -1 and 1
        """

        action = self.policy(obs)
        if noise_std is not None:
            noise = noise_std * tf.random.normal(shape=action.shape)
            action += noise

            # we need to clip again to -1 and 1 after applying the noise
            action = tf.clip_by_value(action, -1, 1)

        qval = self.q_fn(tf.concat([obs, action], axis=-1))

        return action, qval

    def act(self, state: np.ndarray, noise_std=None) -> tf.Tensor:
        """
        Return an action for a given state
        """
        return self.step(state, noise_std=noise_std)[0]

    @tf.function
    def value(self, obs, act):
        qval = self.q_fn(tf.concat([obs, act], axis=-1))
        return qval

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
        return self.q_fn.trainable_variables
