"""Prioritized replay"""
import random
import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return float(self.tree[1])

    def add(self, p, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(1, s)
        return (idx, self.tree[idx], self.data[idx - self.capacity])


class PrioritizedReplay:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=200000):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = 1e-6
        self.max_priority = 1.0

    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))

    def push(self, transition):
        p = max(self.max_priority, self.eps) ** self.alpha
        self.tree.add(p, transition)

    def sample(self, batch_size):
        total = max(self.tree.total(), 1.0)
        batch, idxs, priorities = [], [], []
        segment = total / batch_size
        self.frame += 1
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(max(s, 1e-6))
            if data is None:
                j = random.randrange(self.tree.n_entries)
                data = self.tree.data[j]
                idx = j + self.tree.capacity
                p = max(self.tree.tree[idx], self.eps)
            batch.append(data)
            idxs.append(idx)
            priorities.append(max(p, self.eps))
        probs = np.array(priorities) / (total + self.eps)
        weights = (self.tree.n_entries * np.clip(probs, self.eps, 1.0)) ** (-self.beta_by_frame())
        weights = weights / (weights.max() + self.eps)
        return idxs, batch, weights.astype(np.float32)

    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            p = float(abs(p) + self.eps)
            self.max_priority = max(self.max_priority, p)
            self.tree.update(idx, p ** self.alpha)
