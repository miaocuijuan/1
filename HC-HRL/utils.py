"""Utilities"""
import os
import torch
import numpy as np


def save_model(model, path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), path)


def topsis_rank(matrix, weights=None, cost_flags=None):
    m, n = matrix.shape
    if weights is None:
        weights = np.ones(n)
    weights = weights / (np.sum(weights) + 1e-12)
    if cost_flags is None:
        cost_flags = np.zeros(n, dtype=bool)
    norm = np.linalg.norm(matrix, axis=0)
    norm_matrix = matrix / (norm + 1e-12)
    weighted_matrix = norm_matrix * weights
    ideal, nadir = np.empty(n), np.empty(n)
    for j in range(n):
        if cost_flags[j]:
            ideal[j] = np.min(weighted_matrix[:, j])
            nadir[j] = np.max(weighted_matrix[:, j])
        else:
            ideal[j] = np.max(weighted_matrix[:, j])
            nadir[j] = np.min(weighted_matrix[:, j])
    d_ideal = np.linalg.norm(weighted_matrix - ideal, axis=1)
    d_nadir = np.linalg.norm(weighted_matrix - nadir, axis=1)
    scores = d_nadir / (d_ideal + d_nadir + 1e-12)
    return int(np.argmax(scores))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, **kwargs):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)
