"""
Implements PPO algorithm

"""
import os
from typing import List, Dict, NamedTuple, Callable
import logging
import tensorflow as tf
import numpy as np
import time
import click
from prettytable import PrettyTable
from tf2rl.common.base_class import BasePG
from tf2rl.common.buffer import ReplayBuffer


class TD3(BasePG):
    def __init__(self,
                 policy: Callable,
                 env: Callable,
                 seed=0,
                 pi_lr=3e-4,
                 q_lr=3e-4,
                 polyak_tau=0.005,
                 gamma=0.99,
                 buffer_size=1e6,
                 warm_up_steps=10000,
                 act_noise_std=0.1,
                 update_after=100,
                 batch_size=100,
                 policy_delay=2,
                 target_policy_noise=0.2,
                 target_noise_clip=0.5,
                 clip_norm=None,
                 log_dir_path="results"):

        super(TD3, self).__init__(policy, env, critic_lr=q_lr, pi_lr=pi_lr)

        self.polyak_tau = polyak_tau
        self.gamma = gamma
        self.clip_norm = clip_norm
        self.warm_up_steps = warm_up_steps
        self.act_noise_std = act_noise_std
        self.update_after = update_after
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.log_dir_path = log_dir_path

        self.set_random_seed(seed)
        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed=seed)

        self.policy_tf = self.policy(self.obs_space,
                                     self.action_space)

        # Initialize policy nd Q Networks
        self.pi_fn, self.q1_fn, self.q2_fn = self.policy_tf.setup()

        # Initialize target Q network
        self.target_pi, self.target_q1_fn, self.target_q2_fn = self.policy(self.obs_space,
                                                                           self.action_space,
                                                                           name="target").setup()
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir_path, 'summaries', 'train'))

        self.test_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir_path, 'summaries', 'test'))
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                                        pi=self.pi_fn,
                                        q1_fn=self.q1_fn,
                                        q2_fn=self.q2_fn,
                                        t_pi=self.target_pi,
                                        t_q1_fn=self.target_q1_fn,
                                        t_q2_fn=self.target_q2_fn,
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

        # Compute Q targets
        # Get target next deterministic actions (between -1 nd 1)
        target_pi_outs = self.target_pi(next_obs)
        # Add clipped noise to the target actions
        target_noise = tf.random.normal(
            shape=target_pi_outs.shape, stddev=self.target_policy_noise)
        target_noise = tf.clip_by_value(
            target_noise, -self.target_noise_clip, self.target_noise_clip)
        # Add noise to the target actions
        target_actions = target_pi_outs+target_noise
        # Clip the noisy target actions remain between -1 and 1
        target_actions = tf.clip_by_value(target_actions, -1, 1)

        inputs = tf.concat([next_obs, target_actions], axis=-1)
        q1_target_vals = self.target_q1_fn(inputs)
        q2_target_vals = self.target_q2_fn(inputs)
        q_pi_targets = tf.minimum(q1_target_vals, q2_target_vals)
        # Q function targets
        q_targets = tf.stop_gradient(rews +
                                     self.gamma * (1-terminals) * q_pi_targets)

        q1_values = self.q1_fn(tf.concat([obs, acts], axis=-1))
        q2_values = self.q2_fn(tf.concat([obs, acts], axis=-1))
        q1_loss = tf.reduce_mean(tf.square(q_targets-q1_values))
        q2_loss = tf.reduce_mean(tf.square(q_targets-q2_values))

        critic_loss = q1_loss+q2_loss

        # For the policy loss we use deterministic action
        pi_actions = self.pi_fn(obs)
        q1_pi = self.q1_fn(tf.concat([obs, pi_actions], axis=-1))
        # we want to pick the action that maximizes the Q function
        pi_loss = -tf.reduce_mean(q1_pi)

        info = dict(LossPi=pi_loss, LossCritic=critic_loss)

        return info

    def _train_step(self,
                    batch: List[NamedTuple],
                    update_policy: True) -> Dict[str, tf.Tensor]:
        """main train loop"""

        obs = tf.cast(batch.obs, tf.float32)
        acts = tf.cast(batch.acts, tf.float32)
        next_obs = tf.cast(batch.next_obs, tf.float32)
        rews = tf.cast(batch.rews, tf.float32)
        done = tf.cast(batch.done, tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            loss_info = self.compute_losses(obs,
                                            rews, acts, next_obs, done)

        q_grads = tape.gradient(
            loss_info["LossCritic"], self.policy_tf.critic_vars)
        self.critic_opt.apply_gradients(
            zip(q_grads, self.policy_tf.critic_vars))

        if update_policy:
            pi_grads = tape.gradient(
                loss_info["LossPi"], self.pi_fn.trainable_variables)

            self.pi_opt.apply_gradients(
                zip(pi_grads, self.pi_fn.trainable_variables))

            # update_targets
            self.update_targets()

        return loss_info

    @tf.function
    def update_targets(self):
        assert len(self.target_params) > 0
        assert len(self.source_params) > 0
        self.set_target_values(self.target_params,
                               self.source_params,
                               self.polyak_tau)

    @property
    def target_params(self):
        target_params = self.target_pi.trainable_variables + \
            self.target_q1_fn.trainable_variables+self.target_q2_fn.trainable_variables
        return target_params

    @property
    def source_params(self):
        source_prams = self.pi_fn.trainable_variables + \
            self.q1_fn.trainable_variables+self.q2_fn.trainable_variables
        return source_prams

    @staticmethod
    def set_target_values(target_vars, source_vars, polyak_tau):
        for t_var, s_var in zip(target_vars, source_vars):
            t_var.assign((1-polyak_tau)*t_var+polyak_tau*s_var)

    def learn(self, total_timesteps=100000):

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
        for step in range(total_timesteps):
            state = obs[None, :].astype(np.float32)

            if step > self.update_after:
                action, q1val, q2val = self.policy_tf.step(
                    state, noise_std=self.act_noise_std)
                epoch_q_values.append(min(q1val.numpy(), q2val.numpy()))
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
            if step > self.update_after:
                # for grad_step in range(100):
                batch = self.replay_buffer.get(self.batch_size)
                # (step+grad_step) % self.policy_delay == 0
                update_policy = step % self.policy_delay == 0
                opt_info = self._train_step(batch, update_policy)
                epoch_pi_loss.append(opt_info["LossPi"])
                epoch_q_loss.append(opt_info["LossCritic"])

            self.ckpt.step.assign_add(1)
            if done and step > self.update_after:

                mean_rets = np.mean(np.array(total_ep_rets))
                mean_len = np.mean(np.array(total_ep_len))
                mean_q_loss = np.mean(np.array(epoch_q_loss))
                mean_pi_loss = np.mean(np.array(epoch_pi_loss))

                total_ep_rets, total_ep_len, epoch_q_loss, epoch_pi_loss = [], [], [], []

                train_logger.add_row(
                    ["nrof_episodes ", nrof_episodes])
                train_logger.add_row(["totalEnvInter", step])
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

            if step % 5000 == 0:
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
