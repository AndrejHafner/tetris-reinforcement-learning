from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])

class ReplayMemory(object):

    def __init__(self, max_size):
        self.memory = deque([], maxlen=max_size)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)