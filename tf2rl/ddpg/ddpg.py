"""
Implements DDPG algorithm

"""
import time
import os
from typing import List, Dict, NamedTuple, Callable
import tensorflow as tf
import numpy as np

import click
from prettytable import PrettyTable
from tf2rl.common.base_class import BasePG
from tf2rl.common.buffer import ReplayBuffer


class DDPG(BasePG):
    def __init__(self,
                 policy: Callable,
                 env: Callable,
                 seed=0,
                 pi_lr=1e-4,
                 q_lr=1e-3,
                 polyak_tau=0.005,
                 gamma=0.99,
                 buffer_size=1e6,
                 warm_up_steps=10000,
                 act_noise_std=0.1,
                 update_after=1000,
                 batch_size=100,
                 clip_norm=None,
                 log_dir_path="ddpg_results"):

        super(DDPG, self).__init__(policy, env, critic_lr=q_lr, pi_lr=pi_lr)

        self.polyak_tau = polyak_tau
        self.gamma = gamma
        self.clip_norm = clip_norm
        self.warm_up_steps = warm_up_steps
        self.act_noise_std = act_noise_std
        self.update_after = update_after
        self.batch_size = batch_size
        self.log_dir_path = log_dir_path

        self.set_random_seed(seed)
        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed=seed)

        self.policy_tf = self.policy(self.obs_space,
                                     self.action_space)

        # Initialize policy nd Q Networks
        self.pi_fn, self.q_fn = self.policy_tf.setup()

        # Initialize target Q network
        self.target_pi, self.target_q_fn = self.policy(self.obs_space,
                                                       self.action_space,
                                                       name="target").setup()
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir_path, 'summaries', 'train'))

        self.test_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir_path, 'summaries', 'test'))
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                                        pi=self.pi_fn,
                                        q_fn=self.q_fn,
                                        pi_opt=self.pi_opt,
                                        q_opt=self.critic_opt)

        ckpt_path = os.path.join(self.log_dir_path, 'checkpoints')
        self.manager = tf.train.CheckpointManager(
            self.ckpt, ckpt_path, max_to_keep=3)

    @tf.function
    def compute_losses(self,
                       obs: tf.Tensor,
                       rews: tf.Tensor,
                       acts: tf.Tensor,
                       next_obs: tf.Tensor,
                       terminals: tf.Tensor) -> Dict[str, tf.Tensor]:

        # For the policy loss we use deterministic action (EQ.6)
        pi_actions = self.pi_fn(obs)
        q_pi = self.q_fn(tf.concat([obs, pi_actions], axis=-1))
        # we want to pick the action that maximizes the Q function
        pi_loss = -tf.reduce_mean(q_pi)

        # Compute Q targets
        pi_targets = self.target_pi(next_obs)
        inputs = tf.concat([next_obs, pi_targets], axis=-1)
        q_pi_targets = self.target_q_fn(inputs)
        q_targets = tf.stop_gradient(rews +
                                     self.gamma * (1-terminals) * q_pi_targets)
        q_values = self.q_fn(tf.concat([obs, acts], axis=-1))

        q_loss = tf.reduce_mean(tf.square(q_targets-q_values))

        info = dict(LossPi=pi_loss, LossCritic=q_loss)

        return info

    def _train_step(self,
                    batch: List[NamedTuple],
                    *args, **kwargs) -> Dict[str, tf.Tensor]:
        """main train loop"""

        obs = tf.cast(batch.obs, tf.float32)
        acts = tf.cast(batch.acts, tf.float32)
        next_obs = tf.cast(batch.next_obs, tf.float32)
        rews = tf.cast(batch.rews, tf.float32)
        done = tf.cast(batch.done, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            loss_info = self.compute_losses(obs,
                                            rews, acts, next_obs, done)

        pi_grads = tape.gradient(
            loss_info["LossPi"], self.pi_fn.trainable_variables)
        q_grads = tape.gradient(
            loss_info["LossCritic"], self.q_fn.trainable_variables)
        if self.clip_norm:
            pi_grads = [tf.clip_by_global_norm(
                g, self.clip_norm) for g in pi_grads]

            q_grads = [tf.clip_by_global_norm(
                g, self.clip_norm) for g in q_grads]

        self.pi_opt.apply_gradients(
            zip(pi_grads, self.pi_fn.trainable_variables))
        self.critic_opt.apply_gradients(
            zip(q_grads, self.q_fn.trainable_variables))

        # update_targets
        self.update_targets()

        return loss_info

    @tf.function
    def update_targets(self):
        self.set_target_values(self.target_pi.trainable_variables,
                               self.pi_fn.trainable_variables,
                               self.polyak_tau)

        self.set_target_values(self.target_q_fn.trainable_variables,
                               self.q_fn.trainable_variables,
                               self.polyak_tau)

    @staticmethod
    def set_target_values(target_vars, source_vars, polyak_tau):
        for t_var, s_var in zip(target_vars, source_vars):
            t_var.assign((1-polyak_tau)*t_var+polyak_tau*s_var)

    def learn(self, max_episodes=5000):

        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        nrof_episodes = 0
        nrof_env_inter = 0
        total_ep_rets = []
        total_ep_len = []
        epoch_q_loss = []
        epoch_pi_loss = []
        epoch_q_values = []
        start_time = time.time()

        train_logger = PrettyTable(header=False)
        test_logger = PrettyTable(header=False)
        low, high = self.env.action_space.low, self.env.action_space.high
        while nrof_episodes < max_episodes:
            state = obs[None, :].astype(np.float32)

            if nrof_env_inter > self.warm_up_steps:
                action, qval = self.policy_tf.step(
                    state, noise_std=self.act_noise_std)
                epoch_q_values.append(qval)
                action = action.numpy()[0]
                # the action is between -1 and +1, we need to scaled back to env
                # action space bounds
                # this formula is taken from stable_baselines
                unscaled_action = low + (0.5*(action+1.0)*(high-low))
            else:
                unscaled_action = self.env.action_space.sample()
                # scale the action between -1 and 1
                action = 2.0*((unscaled_action-low)/(high-low)) - 1.0

            next_obs, rew, done, _ = self.env.step(unscaled_action)
            ep_ret += rew
            ep_len += 1

            self.replay_buffer.store((obs, action, next_obs, rew, done))

            obs = next_obs

            if done:
                total_ep_rets.append(ep_ret)
                total_ep_len.append(ep_len)
                nrof_episodes += 1
                obs, ep_ret, ep_len = self.env.reset(), 0, 0

            # Update networks
            if nrof_env_inter >= self.update_after:
                batch = self.replay_buffer.get(self.batch_size)
                opt_info = self._train_step(batch, step=nrof_env_inter)
                epoch_pi_loss.append(opt_info["LossPi"])
                epoch_q_loss.append(opt_info["LossCritic"])

            nrof_env_inter += 1
            self.ckpt.step.assign_add(1)
            if done and nrof_env_inter > self.update_after:

                mean_rets = np.mean(np.array(total_ep_rets))
                mean_len = np.mean(np.array(total_ep_len))
                mean_q_loss = np.mean(np.array(epoch_q_loss))
                mean_pi_loss = np.mean(np.array(epoch_pi_loss))

                total_ep_rets, total_ep_len, epoch_q_loss, epoch_pi_loss = [], [], [], []

                train_logger.add_row(
                    ["nrof_episodes ", nrof_episodes])
                train_logger.add_row(["totalEnvInter", nrof_env_inter])
                train_logger.add_row(
                    ["Total Time (s)", round(time.time() - start_time, 2)])

                train_logger.add_row(["AvgEpRet", mean_rets])
                train_logger.add_row(["AvgEpLen", mean_len])
                train_logger.add_row(["AvgPiLoss", mean_pi_loss])
                train_logger.add_row(["AvgQLoss", mean_q_loss])
                print(train_logger)
                with self.train_writer.as_default():

                    tf.summary.scalar(
                        "AvgPiLoss", mean_pi_loss, step=self.ckpt.step)
                    tf.summary.scalar(
                        "AvgQLoss", mean_q_loss, step=self.ckpt.step)

                    tf.summary.scalar("Episode_avg_return",
                                      mean_rets, step=self.ckpt.step)
                    tf.summary.scalar("Episode_avg_length",
                                      mean_len, step=self.ckpt.step)
                train_logger.clear_rows()
                print("Saving checkpoint ....")
                self.manager.save()

            if nrof_env_inter % 5000 == 0:
                print("Testing policy for %d test episodes", 10)

                avgTestEpRet, avgTestEpLen = run_policy(
                    self.env, self.policy_tf, num_episodes=10, render=False)
                test_logger.add_row(["TestAvgEpRet", avgTestEpRet])
                test_logger.add_row(["TestAvgEpLen", avgTestEpLen])

                print(click.style("\n {}".format(
                    test_logger.get_string()), fg="green"))
                with self.test_writer.as_default():
                    tf.summary.scalar("Episode_avg_return",
                                      avgTestEpRet, step=self.ckpt.step)
                    tf.summary.scalar("Episode_avg_length",
                                      avgTestEpLen, step=self.ckpt.step)
                test_logger.clear_rows()


def run_policy(env, agent, max_ep_len=None, num_episodes=100, render=True, verbose=False):

    totalEpRet = 0
    totalEpLen = 0

    assert env is not None, \
        "Environment not found! :( \n\n "

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        low, high = env.action_space.low, env.action_space.high
        a = agent.act(o[None, :].astype(np.float32), noise_std=None)
        action = a.numpy()[0]
        unscaled_action = low + (0.5*(action+1.0)*(high-low))
        o, r, d, _ = env.step(unscaled_action)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            if verbose:
                click.secho("Episode {} \t EpRet {} \t EpLen {}".format(
                    n, ep_ret, ep_len), fg="blue")

            totalEpRet += ep_ret
            totalEpLen += ep_len
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    return totalEpRet/num_episodes, totalEpLen/num_episodes
