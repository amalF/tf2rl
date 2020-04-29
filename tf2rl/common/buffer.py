"""
Implements replay buffers
"""
import logging
from collections import namedtuple
from typing import NamedTuple, Tuple, List, Any
import secrets
import numpy as np

DEFAULT_FIELDS = ["obs", "acts", "next_obs", "rews", "done"]


class ReplayBuffer:
    def __init__(self,
                 capacity: int = None,
                 seed: int = 1234,
                 samples_def: List[str] = DEFAULT_FIELDS):

        self.logger = logging.getLogger('tfrl.replayBuffer')
        if capacity is None:
            self.logger.warning("Capacity of the buffer cannot be None")
            raise ValueError

        if samples_def is None:
            self.logger.warning(
                "Please specify the definition of the buffer outputs")
            raise ValueError

        self.samples_def = samples_def
        self.sample = namedtuple("sample", samples_def)
        self.size = 0
        self.capacity = capacity
        self.data = []
        self.seed = seed
        self.removed_count = 0
        self.secure_random = secrets.SystemRandom(self.seed)
        self.traj_start_idx = 0

    def store(self, samples: Tuple[Any]):
        """
        store samples into the buffer
        """

        if len(samples) > self.capacity:
            self.logger.warning("Samples size is more than buffer capacity")
            raise RuntimeError

        if not isinstance(samples, list):
            samples = [samples]

        if len(samples[0]) != len(self.sample._fields):
            self.logger.warning(
                "the sample does not respect the sample def %s", self.sample._fields)
            raise RuntimeError

        nrof_samples = len(samples)
        if self.size+nrof_samples >= self.capacity:
            # pop nrof_samples first elements from the buffer
            pop_count = self.size+nrof_samples-self.capacity
            self.data = self.data[pop_count:]
            self.removed_count += pop_count
            self.size = self.capacity
        else:
            self.size = self.size+nrof_samples

        self.data.extend(samples)

    def get(self, batch_size: int = None) -> List[NamedTuple]:
        """Returns a batch of samples"""
        if batch_size is None:
            self.logger.warning(
                "Batch size is None ! get method will return all the buffer")
            self.size = 0
            self.clear_buffer()
            return self._to_sample(self.data)

        data = self.secure_random.sample(self.data, min(batch_size, self.size))

        return self._to_sample(data)

    def _to_sample(self, data: List[Tuple[np.ndarray]]) -> List[NamedTuple]:
        batch = [np.vstack(x) for x in zip(*data)]
        return self.sample(**dict(zip(self.samples_def, batch)))

    def clear_buffer(self):
        self.data = []
