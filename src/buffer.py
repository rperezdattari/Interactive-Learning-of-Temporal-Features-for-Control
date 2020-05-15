import random

"""Code inspired in: https://github.com/fhennecker/deepdoom/blob/master/src/memory.py"""


class Buffer:
    def __init__(self, min_size, max_size):
        self.buffer = []
        self.min_size, self.max_size = min_size, max_size

    def full(self):
        return len(self.buffer) >= self.max_size

    def initialized(self):
        return len(self.buffer) >= self.min_size

    def add(self, step):
        if self.full():
            self.buffer.pop(0)
        self.buffer.append(step)

    def sample(self, batch_size):
        return [random.choice(self.buffer) for _ in range(batch_size)]
