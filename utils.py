from collections import deque
import random
import numpy as np
import torch

def replay_loader(datas, batch_size):
    datalen = datas[0].shape[0]
    idxs = np.arange(datalen)
    np.random.shuffle(idxs)
    for i in range(datalen//batch_size):
        yield [data[idxs[i*batch_size:(i+1)*batch_size]] for data in datas]

class Replay:
    def __init__(self, cfg):
        self._size = 0
        self._capacity = cfg.capacity
        self._memory = [deque([], maxlen=self._capacity) for _ in range(4)]

    def __len__(self):
        return self._size

    def push(self, data):
        assert len(data) == len(self._memory)
        for v, arr in zip(data, self._memory):
            arr.append(v)
        self._size = min(self._size+1, self._capacity)

    def get(self):
        return [torch.stack([torch.Tensor(_) for _ in list(ats)]) for ats in self._memory]

    def clear(self):
        self._memory = [deque([], maxlen=self._capacity) for _ in range(4)]
        self._size = 0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
