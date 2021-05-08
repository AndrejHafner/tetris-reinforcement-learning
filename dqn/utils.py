from collections import namedtuple, deque
import random

Experience = namedtuple("Experience", ["state", "action", "next_state", "reward"])

class ReplayMemory(object):

    def __init__(self, max_size):
        self.memory = deque([], maxlen=max_size)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)