"""HC-HRL for Task Offloading"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Dirichlet
import numpy as np
import time
import os

from env import OffloadingEnv
from config import EnvConfig


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class TaskFeatureEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=np.sqrt(2))

    def forward(self, task_features):
        return self.encoder(task_features)


class ClusterTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_clusters=3, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.cluster_tokens = nn.Parameter(torch.randn(num_clusters, hidden_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, task_embeddings):
        batch_size, num_tasks, _ = task_embeddings.shape
        cluster_tokens = self.cluster_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        combined = torch.cat([cluster_tokens, task_embeddings], dim=1)
        seq_len = combined.shape[1]
        combined = combined + self.pos_embedding[:, :seq_len, :]
        output = self.transformer(combined)
        output = self.norm(output)
        cluster_out = output[:, :self.num_clusters, :]
        task_out = output[:, self.num_clusters:, :]
        return cluster_out, task_out


class SoftAssignment(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, task_embeddings, cluster_embeddings):
        scores = torch.bmm(task_embeddings, cluster_embeddings.transpose(1, 2))
        Y = F.softmax(scores / self.temperature, dim=-1)
        return Y


class FeatureFusion(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        orthogonal_init(self.fusion_proj[0], gain=np.sqrt(2))

    def forward(self, task_embeddings, cluster_embeddings, soft_assignment):
        cluster_info = torch.bmm(soft_assignment, cluster_embeddings)
        combined = torch.cat([task_embeddings, cluster_info], dim=-1)
        fused = self.fusion_proj(combined)
        return fused


class HighLevelActor(nn.Module):
    def __init__(self, hidden_dim=128, action_dim=3):
        super().__init__()
        self.action_dim = action_dim
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)

    def forward(self, cluster_embeddings, soft_assignment, temperature=1.0, hard=False):
        batch_size, num_clusters, _ = cluster_embeddings.shape
        logits = self.policy_net(cluster_embeddings)
        cluster_actions = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
        task_priors = torch.bmm(soft_assignment, cluster_actions)
        return cluster_actions, task_priors, logits


class LowLevelActor(nn.Module):
    def __init__(self, hidden_dim=128, action_dim=3):
        super().__init__()
        self.action_dim = action_dim
        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        for layer in self.alpha_net:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=0.01)

    def forward(self, fused_features, task_priors=None):
        if task_priors is not None:
            features = fused_features + F.linear(task_priors,
                torch.eye(self.action_dim, fused_features.shape[-1], device=fused_features.device).T * 0.1)
        else:
            features = fused_features
        alpha_raw = self.alpha_net(features)
        alpha = F.softplus(alpha_raw) + 1.0
        return alpha

    def sample(self, alpha, deterministic=False):
        if deterministic:
            return alpha / alpha.sum(dim=-1, keepdim=True)
        dist = Dirichlet(alpha)
        return dist.sample()

    def log_prob(self, alpha, action):
        action_clamped = torch.clamp(action, 1e-6, 1.0 - 1e-6)
        action_clamped = action_clamped / action_clamped.sum(dim=-1, keepdim=True)
        dist = Dirichlet(alpha)
        return dist.log_prob(action_clamped)

    def entropy(self, alpha):
        dist = Dirichlet(alpha)
        return dist.entropy()


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_dim=256):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for layer in self.q_net:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=1.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q_net(x)


class CEMModule:
    def __init__(self, num_samples=10, elite_ratio=0.3, noise_std=0.1):
        self.num_samples = num_samples
        self.elite_ratio = elite_ratio
        self.noise_std = noise_std
        self.num_elite = max(1, int(num_samples * elite_ratio))

    def generate_candidates(self, rho_init, q_network, state):
        batch_size = rho_init.shape[0]
        device = rho_init.device
        noise = torch.randn(batch_size, self.num_samples, 3, device=device) * self.noise_std
        perturbed = rho_init.unsqueeze(1) + noise
        perturbed = F.softmax(perturbed, dim=-1)
        state_expanded = state.unsqueeze(1).expand(-1, self.num_samples, -1)
        state_flat = state_expanded.reshape(-1, state.shape[-1])
        action_flat = perturbed.reshape(-1, 3)
        with torch.no_grad():
            q_values = q_network(state_flat, action_flat).reshape(batch_size, self.num_samples)
        _, elite_indices = torch.topk(q_values, self.num_elite, dim=1)
        elite_samples = torch.gather(perturbed, 1, elite_indices.unsqueeze(-1).expand(-1, -1, 3))
        balanced = torch.ones(batch_size, 1, 3, device=device) / 3
        candidates = torch.cat([rho_init.unsqueeze(1), elite_samples, balanced], dim=1)
        num_candidates = candidates.shape[1]
        state_expanded = state.unsqueeze(1).expand(-1, num_candidates, -1)
        state_flat = state_expanded.reshape(-1, state.shape[-1])
        candidates_flat = candidates.reshape(-1, 3)
        with torch.no_grad():
            all_q_values = q_network(state_flat, candidates_flat).reshape(batch_size, num_candidates)
        return candidates, all_q_values


class TOPSISSelector:
    def __init__(self, weights=None):
        self.weights = weights or [0.35, 0.35, 0.15, 0.15]
        self.is_benefit = [False, True, False, False]

    def select(self, candidates, task_features):
        batch_size, num_candidates, _ = candidates.shape
        candidates_np = candidates.cpu().numpy()
        if task_features.dim() == 3:
            task_features = task_features.mean(dim=1)
        task_np = task_features.cpu().numpy()
        selected_list = []
        selected_idx_list = []
        for b in range(batch_size):
            size, cycles, ddl = task_np[b]
            criteria_matrix = []
            for c in range(num_candidates):
                cand = candidates_np[b, c]
                local_ratio, rsu_ratio, v2v_ratio = cand
                local_delay = (cycles * local_ratio) / 80 if local_ratio > 0.01 else 0
                rsu_trans = (size * rsu_ratio * 8) / 25 if rsu_ratio > 0.01 else 0
                rsu_comp = (cycles * rsu_ratio) / 600 if rsu_ratio > 0.01 else 0
                v2v_trans = (size * v2v_ratio * 8) / 200 if v2v_ratio > 0.01 else 0
                v2v_comp = (cycles * v2v_ratio) / 80 if v2v_ratio > 0.01 else 0
                total_delay = max(local_delay, rsu_trans + rsu_comp, v2v_trans + v2v_comp)
                if total_delay < ddl * 0.8:
                    success_prob = 1.0
                elif total_delay < ddl:
                    success_prob = 0.9 - 0.4 * (total_delay - ddl * 0.8) / (ddl * 0.2 + 1e-6)
                else:
                    success_prob = max(0.1, 0.5 - (total_delay - ddl) / (ddl + 1e-6))
                energy = local_ratio * 1.0 + rsu_ratio * 0.2 + v2v_ratio * 0.4
                uniform = np.array([1/3, 1/3, 1/3])
                kl_div = np.sum(cand * np.log((cand + 1e-8) / uniform))
                criteria_matrix.append([total_delay, success_prob, energy, kl_div])
            criteria_matrix = np.array(criteria_matrix)
            best_idx = self._topsis_rank(criteria_matrix)
            selected_list.append(candidates_np[b, best_idx])
            selected_idx_list.append(best_idx)
        selected = torch.tensor(np.array(selected_list), dtype=torch.float32, device=candidates.device)
        selected_idx = torch.tensor(selected_idx_list, dtype=torch.long, device=candidates.device)
        return selected, selected_idx

    def _topsis_rank(self, matrix):
        norm = np.sqrt((matrix ** 2).sum(axis=0))
        norm[norm == 0] = 1
        normalized = matrix / norm
        weighted = normalized * self.weights
        ideal = np.zeros(len(self.weights))
        anti_ideal = np.zeros(len(self.weights))
        for j in range(len(self.weights)):
            if self.is_benefit[j]:
                ideal[j] = weighted[:, j].max()
                anti_ideal[j] = weighted[:, j].min()
            else:
                ideal[j] = weighted[:, j].min()
                anti_ideal[j] = weighted[:, j].max()
        d_plus = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
        d_minus = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))
        closeness = d_minus / (d_plus + d_minus + 1e-8)
        return np.argmax(closeness)


class HCHRL(nn.Module):
    def __init__(self, num_tasks=5, hidden_dim=128, num_clusters=3, num_transformer_layers=2, action_dim=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.action_dim = action_dim
        self.task_encoder = TaskFeatureEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.cluster_transformer = ClusterTransformerEncoder(
            hidden_dim=hidden_dim, num_clusters=num_clusters,
            num_layers=num_transformer_layers, num_heads=4
        )
        self.soft_assignment = SoftAssignment(temperature=1.0)
        self.feature_fusion = FeatureFusion(hidden_dim=hidden_dim)
        self.high_level_actor = HighLevelActor(hidden_dim=hidden_dim, action_dim=action_dim)
        self.low_level_actor = LowLevelActor(hidden_dim=hidden_dim, action_dim=action_dim)

    def forward(self, task_features, temperature=1.0, deterministic=False):
        batch_size = task_features.shape[0]
        task_embeddings = self.task_encoder(task_features)
        cluster_out, task_out = self.cluster_transformer(task_embeddings)
        Y = self.soft_assignment(task_out, cluster_out)
        fused_features = self.feature_fusion(task_embeddings, cluster_out, Y)
        cluster_actions, task_priors, hi_logits = self.high_level_actor(
            cluster_out, Y, temperature=temperature, hard=not self.training
        )
        alpha = self.low_level_actor(fused_features, task_priors)
        rho_init = self.low_level_actor.sample(alpha, deterministic=deterministic)
        return {
            'task_embeddings': task_embeddings, 'cluster_embeddings': cluster_out,
            'task_out': task_out, 'soft_assignment': Y, 'fused_features': fused_features,
            'cluster_actions': cluster_actions, 'task_priors': task_priors,
            'hi_logits': hi_logits, 'alpha': alpha, 'rho_init': rho_init
        }


class HCHRLCritic(nn.Module):
    def __init__(self, num_tasks=5, hidden_dim=128, num_clusters=3):
        super().__init__()
        self.task_encoder = TaskFeatureEncoder(input_dim=3, hidden_dim=hidden_dim)
        self.cluster_transformer = ClusterTransformerEncoder(
            hidden_dim=hidden_dim, num_clusters=num_clusters, num_layers=1, num_heads=4
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * (num_clusters + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                orthogonal_init(layer, gain=1.0)

    def forward(self, task_features):
        task_embeddings = self.task_encoder(task_features)
        cluster_out, task_out = self.cluster_transformer(task_embeddings)
        cluster_pooled = cluster_out.mean(dim=1)
        task_pooled = task_out.mean(dim=1)
        combined = torch.cat([cluster_pooled, cluster_out.view(cluster_out.shape[0], -1)], dim=-1)
        if combined.shape[-1] != self.value_head[0].in_features:
            combined = torch.cat([cluster_pooled, task_pooled], dim=-1)
            combined = F.pad(combined, (0, self.value_head[0].in_features - combined.shape[-1]))
        return self.value_head(combined)


class HCHRLAgent:
    def __init__(self, num_tasks=5, hidden_dim=128, num_clusters=3, lr_actor=1e-4, lr_critic=3e-4, lr_q=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, entropy_coef=0.01, distill_coef=0.1,
                 hi_coef=0.5, max_grad_norm=0.5, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.distill_coef = distill_coef
        self.hi_coef = hi_coef
        self.max_grad_norm = max_grad_norm
        self.num_tasks = num_tasks
        self.actor = HCHRL(num_tasks, hidden_dim, num_clusters).to(device)
        self.critic = HCHRLCritic(num_tasks, hidden_dim, num_clusters).to(device)
        self.q_network = QNetwork(num_tasks * 3, action_dim=3, hidden_dim=hidden_dim).to(device)
        self.cem = CEMModule(num_samples=10, elite_ratio=0.3, noise_std=0.1)
        self.topsis = TOPSISSelector()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr_q)
        self.buffer = []

    def select_action(self, task_features, deterministic=False, use_cem_topsis=True):
        if task_features.dim() == 2:
            task_features = task_features.unsqueeze(0)
        task_features = task_features.to(self.device)
        self.actor.eval()
        with torch.no_grad():
            outputs = self.actor(task_features, deterministic=deterministic)
            value = self.critic(task_features)
        self.actor.train()
        rho_init = outputs['rho_init'].mean(dim=1)
        if use_cem_topsis and not deterministic:
            state = task_features.view(task_features.shape[0], -1)
            candidates, q_values = self.cem.generate_candidates(rho_init, self.q_network, state)
            selected, selected_idx = self.topsis.select(candidates, task_features)
            action = selected
        else:
            action = rho_init
        return {'action': action, 'rho_init': rho_init, 'alpha': outputs['alpha'], 'value': value,
                'hi_logits': outputs['hi_logits'], 'soft_assignment': outputs['soft_assignment']}

    def store_transition(self, state, action, reward, next_state, done, info):
        self.buffer.append({'state': state, 'action': action, 'reward': reward,
                           'next_state': next_state, 'done': done, 'info': info})

    def update(self, batch_size=32, update_epochs=10):
        if len(self.buffer) < batch_size:
            return {}
        states = torch.stack([torch.FloatTensor(t['state']) for t in self.buffer]).to(self.device)
        actions = torch.stack([torch.FloatTensor(t['action']) for t in self.buffer]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in self.buffer]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in self.buffer]).to(self.device)
        if states.dim() == 2:
            states = states.view(-1, self.num_tasks, 3)
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = torch.cat([values[1:], torch.zeros(1, device=self.device)])
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_loss = 0
        for _ in range(update_epochs):
            outputs = self.actor(states)
            current_values = self.critic(states).squeeze(-1)
            alpha = outputs['alpha']
            rho_init = outputs['rho_init'].mean(dim=1)
            log_probs = self.actor.low_level_actor.log_prob(alpha.mean(dim=1), actions)
            ratio = torch.exp(log_probs - log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            lo_loss = -torch.min(surr1, surr2).mean()
            distill_loss = F.mse_loss(rho_init, actions)
            hi_logits = outputs['hi_logits']
            hi_probs = F.softmax(hi_logits, dim=-1)
            hi_entropy = -(hi_probs * torch.log(hi_probs + 1e-8)).sum(dim=-1).mean()
            hi_loss = -self.entropy_coef * hi_entropy
            value_loss = F.mse_loss(current_values, returns)
            entropy = self.actor.low_level_actor.entropy(alpha.mean(dim=1)).mean()
            loss = lo_loss + self.distill_coef * distill_loss + self.hi_coef * hi_loss + 0.5 * value_loss - self.entropy_coef * entropy
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            total_loss += loss.item()
        state_flat = states.view(-1, self.num_tasks * 3)
        q_pred = self.q_network(state_flat, actions).squeeze(-1)
        q_target = returns
        q_loss = F.mse_loss(q_pred, q_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.buffer = []
        return {'loss': total_loss / update_epochs, 'value_loss': value_loss.item(),
                'distill_loss': distill_loss.item(), 'entropy': entropy.item()}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                    'q_network': self.q_network.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.q_network.load_state_dict(checkpoint['q_network'])


class MultiStepEnv:
    def __init__(self, max_steps=10, tasks_per_step=5, device='cpu'):
        self.cfg = EnvConfig()
        self.cfg.tasks_per_episode = tasks_per_step
        self.base_env = OffloadingEnv(device=device)
        self.base_env.cfg = self.cfg
        self.max_steps = max_steps
        self.tasks_per_step = tasks_per_step
        self.device = torch.device(device)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.base_env.reset()
        return self._get_state()

    def _get_state(self):
        task_feat = self.base_env.task_feat
        task_norm = task_feat.clone()
        task_norm[:, 0] = task_feat[:, 0] / 1.5
        task_norm[:, 1] = task_feat[:, 1] / 300
        task_norm[:, 2] = task_feat[:, 2] / 1.0
        return task_norm.to(self.device), task_feat.to(self.device)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device)
        if action.dim() == 1:
            action = action.unsqueeze(0).expand(self.tasks_per_step, -1)
        _, reward, _, info = self.base_env.step(action)
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        if not done:
            self.base_env.reset()
        next_state = self._get_state()
        return next_state, float(reward), done, info


def train_hchrl(num_episodes=300, max_steps=10, tasks_per_step=5, eval_interval=50, save_path='outputs', device='cpu'):
    env = MultiStepEnv(max_steps=max_steps, tasks_per_step=tasks_per_step, device=device)
    agent = HCHRLAgent(num_tasks=tasks_per_step, hidden_dim=128, num_clusters=3,
                       lr_actor=1e-4, lr_critic=3e-4, lr_q=3e-4, gamma=0.99,
                       entropy_coef=0.02, distill_coef=0.1, device=device)
    history = {'reward': [], 'success_rate': [], 'total_delay': []}
    best_reward = float('-inf')
    start_time = time.time()
    for episode in range(1, num_episodes + 1):
        state, task_feat = env.reset()
        episode_reward = 0
        episode_success = 0
        episode_delay = 0
        episode_steps = 0
        for step in range(max_steps):
            with torch.no_grad():
                output = agent.select_action(state, deterministic=False, use_cem_topsis=True)
            action = output['action'].squeeze(0).cpu().numpy()
            next_state_tuple, reward, done, info = env.step(action)
            next_state, next_task_feat = next_state_tuple
            agent.store_transition(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done, output)
            episode_reward += reward
            episode_success += info.get('success_rate', 0)
            episode_delay += info.get('total_delay', 0)
            episode_steps += 1
            state = next_state
            task_feat = next_task_feat
            if done:
                break
        metrics = agent.update(batch_size=32, update_epochs=5)
        success_rate = episode_success / episode_steps
        avg_delay = episode_delay / (episode_steps * tasks_per_step)
        history['reward'].append(episode_reward)
        history['success_rate'].append(success_rate)
        history['total_delay'].append(avg_delay)
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs(save_path, exist_ok=True)
            agent.save(f'{save_path}/model_hchrl_best.pth')
        if episode % eval_interval == 0:
            elapsed = time.time() - start_time
            loss = metrics.get('loss', 0)
            print(f"[Ep {episode:04d}] R={episode_reward:.2f} SR={success_rate*100:.1f}% T={elapsed:.1f}s")
    agent.save(f'{save_path}/model_hchrl_final.pth')
    np.savez(f'{save_path}/training_history_hchrl.npz', reward=np.array(history['reward']),
             success_rate=np.array(history['success_rate']), total_delay=np.array(history['total_delay']))
    return agent, history


def evaluate_hchrl(agent, num_episodes=50, tasks_per_step=5, max_steps=10, use_cem_topsis=True):
    env = MultiStepEnv(max_steps=max_steps, tasks_per_step=tasks_per_step, device=agent.device)
    returns, success_rates, delays = [], [], []
    for _ in range(num_episodes):
        state, task_feat = env.reset()
        ep_reward, ep_success, ep_delay, ep_steps = 0, 0, 0, 0
        for _ in range(max_steps):
            with torch.no_grad():
                output = agent.select_action(state, deterministic=True, use_cem_topsis=use_cem_topsis)
            action = output['action'].squeeze(0).cpu().numpy()
            next_state_tuple, reward, done, info = env.step(action)
            next_state, next_task_feat = next_state_tuple
            ep_reward += reward
            ep_success += info.get('success_rate', 0)
            ep_delay += info.get('total_delay', 0)
            ep_steps += 1
            state = next_state
            if done:
                break
        returns.append(ep_reward)
        success_rates.append(ep_success / ep_steps)
        delays.append(ep_delay / (ep_steps * tasks_per_step))
    return {'return_mean': np.mean(returns), 'return_std': np.std(returns),
            'success_rate': np.mean(success_rates), 'avg_delay': np.mean(delays)}


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent, history = train_hchrl(num_episodes=300, max_steps=10, tasks_per_step=5, eval_interval=50, device=device)
    results = evaluate_hchrl(agent, num_episodes=50, use_cem_topsis=True)
    print(f"Final: SR={results['success_rate']*100:.1f}% R={results['return_mean']:.2f}")
