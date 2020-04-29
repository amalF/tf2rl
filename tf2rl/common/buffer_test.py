import random
import tensorflow as tf
import numpy as np

from tfrl.buffers.buffer import ReplayBuffer

class ReplayBufferTest(tf.test.TestCase):
    def test_store_one_sample_q_learning(self):
        obs_dim = 4
        act_dim = 2
        obs = np.random.uniform(size=obs_dim)
        next_obs = np.random.uniform(size=obs_dim)
        act = random.randint(0, act_dim)
        reward = random.uniform(0, 1)
        done = False

        samples_def = ["obs", "act", "next_obs", "rewds", "done"]
        rb = ReplayBuffer(capacity=10,
                          samples_def=samples_def)
        rb.store((obs, act, next_obs, reward, done))

        self.assertEqual(rb.size, 1)
        self.assertEqual(len(rb.data), 1)
        self.assertAllEqual(rb.data[rb.size-1][0], obs)
        self.assertAllEqual(rb.data[rb.size-1][1], act)
        self.assertAllEqual(rb.data[rb.size-1][2], next_obs)
        self.assertEqual(rb.data[rb.size-1][3], reward)
        self.assertEqual(rb.data[rb.size-1][4], done)

    def test_store_batch_sample_q_learning(self):
        obs_dim = 4
        act_dim = 2
        batch_size = 10
        obs = np.random.uniform(size=(batch_size, obs_dim))
        next_obs = np.random.uniform(size=(batch_size, obs_dim))
        acts = np.random.randint(0, act_dim, size=(batch_size,))
        rewards = np.random.uniform(0, 1, size=(batch_size,))
        done = np.random.randint(0, 1, size=(batch_size,)).astype(bool)

        samples = [x for x in zip(obs, acts, next_obs, rewards, done)]
        samples_def = ["obs", "act", "next_obs", "rewds", "done"]
        rb = ReplayBuffer(capacity=10,
                          samples_def=samples_def)
        rb.store(samples)

        self.assertEqual(rb.size, len(samples))
        self.assertEqual(len(rb.data), len(samples))
        self.assertAllEqual(rb.data[rb.size-1][0], samples[-1][0])
        self.assertAllEqual(rb.data[rb.size-1][1], samples[-1][1])
        self.assertAllEqual(rb.data[rb.size-1][2], samples[-1][2])
        self.assertEqual(rb.data[rb.size-1][3], samples[-1][3])
        self.assertEqual(rb.data[rb.size-1][4], samples[-1][4])

    def test_store_batch_more_than_free_space(self):
        obs_dim = 4
        act_dim = 2
        batch_size = 5
        samples_def = ["obs", "act", "next_obs", "rewds", "done"]
        rb = ReplayBuffer(capacity=8,
                          samples_def=samples_def)
        obs = np.random.uniform(size=(batch_size, obs_dim))
        next_obs = np.random.uniform(size=(batch_size, obs_dim))
        acts = np.random.randint(0, act_dim, size=(batch_size,))
        rewards = np.random.uniform(0, 1, size=(batch_size,))
        done = np.random.randint(0, 1, size=(batch_size,)).astype(bool)

        samples = [x for x in zip(obs, acts, next_obs, rewards, done)]
        
        for _ in range(2):
            rb.store(samples)

        self.assertEqual(rb.size, 8)
        self.assertEqual(rb.removed_count,2)

    def test_get_batch(self):
        obs_dim = 4
        act_dim = 2
        batch_size = 5
        samples_def = ["obs", "act", "next_obs", "rewds", "done"]
        rb = ReplayBuffer(capacity=10,
                          samples_def=samples_def)
        obs = np.random.uniform(size=(batch_size, obs_dim))
        next_obs = np.random.uniform(size=(batch_size, obs_dim))
        acts = np.random.randint(0, act_dim, size=(batch_size,1))
        rewards = np.random.uniform(0, 1, size=(batch_size,1))
        done = np.random.randint(0, 1, size=(batch_size,1)).astype(bool)
                                                                        
        samples = [x for x in zip(obs, acts, next_obs, rewards, done)]
        rb.store(samples)
        batch = rb.get(batch_size)
        self.assertEqual(len(batch),batch_size)
        self.assertAllEqual(batch.obs.shape, obs.shape)
        self.assertAllEqual(batch.next_obs.shape, next_obs.shape)
        self.assertAllEqual(batch.act.shape, acts.shape)
        self.assertAllEqual(batch.rewds.shape, rewards.shape)
        self.assertAllEqual(batch.done.shape, done.shape)

if __name__ == "__main__":
    tf.test.main()
