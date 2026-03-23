"""VEC Task Offloading Environment"""
import random
import heapq
from typing import Dict, Tuple, List
import numpy as np
import torch
from torch.nn.functional import normalize
from config import EnvConfig

DIFF = {"BW_RSU_SHRINK": 0.70, "BW_V2V_SHRINK": 0.92, "SETUP_RSU": 0.020, "SETUP_V2V": 0.006,
        "CTX_VEH": 0.0002, "CTX_RSU": 0.003, "MERGE_OVERHEAD": 0.004}


class OffloadingEnv:
    def __init__(self, device: str = "cpu", **env_kwargs):
        self.device = torch.device(device)
        self.cfg = EnvConfig()
        for k, v in env_kwargs.items():
            setattr(self.cfg, k, v)
        self.task_feat = None
        self.veh_feat = None
        self.rsu_feat = None
        self.vehicle_rsu_map = None
        self.task_origin = None
        self.ep = 0

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
        cfg, dev = self.cfg, self.device
        self.ep += 1
        veh_compute = torch.empty(cfg.num_vehicles, 1, device=dev).uniform_(cfg.veh_compute_min, cfg.veh_compute_max)
        veh_queue = torch.zeros(cfg.num_vehicles, 1, device=dev)
        default_rsu = torch.randint(0, cfg.num_rsus, (cfg.num_vehicles, 1), device=dev)
        self.vehicle_rsu_map = default_rsu.squeeze(-1).clone()
        self.veh_feat = torch.cat([veh_compute, veh_queue, default_rsu.float()], dim=-1)
        rsu_cap = torch.empty(cfg.num_rsus, 1, device=dev).uniform_(cfg.rsu_compute_min, cfg.rsu_compute_max)
        rsu_queue = torch.zeros(cfg.num_rsus, 1, device=dev)
        zero_pad2 = torch.zeros(cfg.num_rsus, 1, device=dev)
        self.rsu_feat = torch.cat([rsu_cap, rsu_queue, zero_pad2], dim=-1)
        T = cfg.tasks_per_episode
        n_type = len(cfg.task_types)
        weights = np.array([tp.get("weight", 1.0 / n_type) for tp in cfg.task_types])
        weights = weights / weights.sum()
        tid = torch.tensor(np.random.choice(n_type, size=T, p=weights), device=dev, dtype=torch.long)
        size_mb = torch.empty(T, device=dev)
        cycles = torch.empty(T, device=dev)
        ddl = torch.empty(T, device=dev)
        origin = torch.randint(0, cfg.num_vehicles, (T,), device=dev)
        for i in range(T):
            tp = cfg.task_types[tid[i].item()]
            smin, smax = tp["size"]
            size = random.uniform(smin, smax)
            flop_cfg = tp["flop"]
            flop_per_mb = random.uniform(flop_cfg[0], flop_cfg[1]) if isinstance(flop_cfg, (list, tuple)) else float(flop_cfg)
            base_cycles = size * flop_per_mb
            noise = 1.0 + random.uniform(-cfg.cycle_noise, cfg.cycle_noise)
            size_mb[i] = size
            cycles[i] = base_cycles * noise
            ddl_cfg = tp["ddl"]
            base_ddl = random.uniform(ddl_cfg[0], ddl_cfg[1]) if isinstance(ddl_cfg, (list, tuple)) else float(ddl_cfg)
            ddl_low = base_ddl * (1.0 - cfg.deadline_jitter)
            ddl_high = base_ddl * (1.0 + cfg.deadline_jitter)
            ddl[i] = random.uniform(ddl_low, ddl_high)
        self.task_origin = origin.clone()
        self.task_feat = torch.stack([size_mb, cycles, ddl], dim=-1)
        x_dict = {"task": normalize(self.task_feat.clone(), dim=0), "vehicle": normalize(self.veh_feat.clone(), dim=0),
                  "rsu": normalize(self.rsu_feat.clone(), dim=0)}
        edge_index_dict = self._build_edge_index_dict()
        return x_dict, edge_index_dict

    def _build_edge_index_dict(self) -> dict:
        cfg, dev = self.cfg, self.device
        T, V, R = cfg.tasks_per_episode, cfg.num_vehicles, cfg.num_rsus
        same_ori_src, same_ori_dst, knn_src, knn_dst = [], [], [], []
        belongs_src, belongs_dst, owns_src, owns_dst = [], [], [], []
        share_src, share_dst, uplink_src, uplink_dst = [], [], [], []
        downlink_src, downlink_dst = [], []
        sizes = self.task_feat[:, 0].cpu().numpy()
        bins = {}
        for t in range(T):
            v = int(self.task_origin[t].item())
            r = int(self.vehicle_rsu_map[v].item())
            belongs_src.append(t); belongs_dst.append(v)
            owns_src.append(v); owns_dst.append(t)
            uplink_src.append(v); uplink_dst.append(r)
            downlink_src.append(r); downlink_dst.append(v)
            b = int(min(3, max(0, sizes[t] // max(1.0, sizes.mean()/2.0))))
            bins.setdefault(b, []).append(t)
        origin_to_tasks = {}
        for t in range(T):
            v = int(self.task_origin[t].item())
            origin_to_tasks.setdefault(v, []).append(t)
        for v, ts in origin_to_tasks.items():
            for i in range(len(ts)):
                for j in range(len(ts)):
                    if i != j:
                        same_ori_src.append(ts[i]); same_ori_dst.append(ts[j])
        rsu_to_vs = {}
        for v in range(V):
            r = int(self.vehicle_rsu_map[v].item())
            rsu_to_vs.setdefault(r, []).append(v)
        for r, vs in rsu_to_vs.items():
            for i in range(len(vs)):
                for j in range(len(vs)):
                    if i != j:
                        share_src.append(vs[i]); share_dst.append(vs[j])
        for _, ts in bins.items():
            for i in range(len(ts)):
                for j in range(len(ts)):
                    if i != j:
                        knn_src.append(ts[i]); knn_dst.append(ts[j])
        def pair_to_tensor(src_list, dst_list):
            if len(src_list) == 0:
                return torch.empty(2, 0, dtype=torch.long, device=dev)
            return torch.stack([torch.tensor(src_list, dtype=torch.long, device=dev),
                                torch.tensor(dst_list, dtype=torch.long, device=dev)], dim=0)
        return {("task","same_origin","task"): pair_to_tensor(same_ori_src, same_ori_dst),
                ("task","knn_sim","task"): pair_to_tensor(knn_src, knn_dst),
                ("task","belongs","vehicle"): pair_to_tensor(belongs_src, belongs_dst),
                ("vehicle","owns","task"): pair_to_tensor(owns_src, owns_dst),
                ("vehicle","share_rsu","vehicle"): pair_to_tensor(share_src, share_dst),
                ("vehicle","uplink","rsu"): pair_to_tensor(uplink_src, uplink_dst),
                ("rsu","downlink","vehicle"): pair_to_tensor(downlink_src, downlink_dst)}

    def _bw_eff_v2v(self, u: int, v: int) -> float:
        return float(self.cfg.bw_v2v) * DIFF["BW_V2V_SHRINK"]

    def _bw_eff_v2i(self) -> float:
        return float(self.cfg.bw_v2i) * DIFF["BW_RSU_SHRINK"]

    def _predict_ttl_v2v(self, u: int, v: int) -> float:
        return float(getattr(self.cfg, "v2v_ttl_base", 0.70))

    def step(self, rho: torch.Tensor):
        cfg, dev = self.cfg, self.device
        T = cfg.tasks_per_episode
        V, R = cfg.num_vehicles, cfg.num_rsus
        node_queues = [[] for _ in range(V + R)]
        link_finish = torch.zeros(V + R, device=dev)
        finish_times: List[float] = []
        total_delay, worst_delay, total_price, success, timeout = 0.0, 0.0, 0.0, 0, 0
        for i in range(T):
            size_i = float(self.task_feat[i, 0].item())
            cyc_i = float(self.task_feat[i, 1].item())
            dl_i = float(self.task_feat[i, 2].item())
            vid = int(self.task_origin[i].item())
            rsu_id = int(self.vehicle_rsu_map[vid].item())
            rsu_idx = V + rsu_id
            rho_l, rho_r, rho_v = float(rho[i,0].item()), float(rho[i,1].item()), float(rho[i,2].item())
            lat_l = lat_r = lat_v = 0.0
            if rho_l > 0:
                c_l = cyc_i * rho_l
                heap = node_queues[vid]
                start = 0.0 if len(heap) < cfg.max_parallel else heapq.heappop(heap)
                pow_v = float(self.veh_feat[vid, 0].item())
                fin_l = start + c_l / max(1e-6, pow_v) + DIFF["CTX_VEH"]
                heapq.heappush(heap, fin_l)
                lat_l = fin_l
            if rho_r > 0:
                c_r = cyc_i * rho_r
                bits = size_i * rho_r * 8e6
                t_tx = DIFF["SETUP_RSU"] + bits / (self._bw_eff_v2i() * 1e6)
                heap_r = node_queues[rsu_idx]
                start_r = t_tx if len(heap_r) < cfg.max_parallel else max(t_tx, heapq.heappop(heap_r))
                pow_r = float(self.rsu_feat[rsu_id, 0].item())
                fin_r = start_r + c_r / max(1e-6, pow_r) + DIFF["CTX_RSU"]
                heapq.heappush(heap_r, fin_r)
                lat_r = fin_r
            if rho_v > 0:
                c_v = cyc_i * rho_v
                bits_v = size_i * rho_v * 8e6
                available_neighbors = []
                for vv in range(V):
                    if vv == vid:
                        continue
                    ttl = self._predict_ttl_v2v(vid, vv)
                    pow_vv = float(self.veh_feat[vv, 0].item())
                    heap_vv = node_queues[vv]
                    wait_time = heapq.nsmallest(1, heap_vv)[0] if len(heap_vv) >= cfg.max_parallel and heap_vv else 0
                    t_txv = DIFF["SETUP_V2V"] + bits_v / (self._bw_eff_v2v(vid, vv) * 1e6)
                    if t_txv < ttl:
                        available_neighbors.append((vv, pow_vv, ttl, wait_time))
                if len(available_neighbors) > 0:
                    available_neighbors.sort(key=lambda x: x[3])
                    dynamic_v2v_limit = max(2, V // 2)
                    max_v2v_parallel = min(len(available_neighbors), dynamic_v2v_limit)
                    selected = available_neighbors[:max_v2v_parallel]
                    c_per_neighbor = c_v / len(selected)
                    max_finish_time = 0
                    for vv, pow_vv, ttl, wait_time in selected:
                        heap_vv = node_queues[vv]
                        start_vv = wait_time if len(heap_vv) < cfg.max_parallel else heapq.heappop(heap_vv)
                        t_txv = DIFF["SETUP_V2V"] + bits_v / len(selected) / (self._bw_eff_v2v(vid, vv) * 1e6)
                        t_cmp_vv = c_per_neighbor / max(1e-6, pow_vv)
                        fin_vv = max(start_vv, t_txv) + t_cmp_vv
                        heapq.heappush(heap_vv, fin_vv)
                        max_finish_time = max(max_finish_time, fin_vv)
                    lat_v = max_finish_time + DIFF["MERGE_OVERHEAD"] * len(selected)
                else:
                    lat_v = dl_i * 1.5
            t_off = max(lat_l, lat_r, lat_v)
            paths_used = int(rho_l > 0.0) + int(rho_r > 0.0) + int(rho_v > 0.0)
            if paths_used >= 2:
                t_off += DIFF["MERGE_OVERHEAD"]
            finish_times.append(t_off)
            total_delay += t_off
            worst_delay = max(worst_delay, t_off)
            if t_off <= dl_i:
                success += 1
            else:
                timeout += 1
            price_i = cfg.price_rsu * rho_r + cfg.price_veh * rho_v
            total_price += price_i
        avg_delay = total_delay / max(1, T)
        succ_rate = success / max(1, T)
        avg_timeout = timeout / max(1, T)
        rho_np = rho.cpu().numpy() if hasattr(rho, 'cpu') else rho.numpy()
        local_usage = (rho_np[:, 0] > 0.1).mean()
        rsu_usage = (rho_np[:, 1] > 0.1).mean()
        v2v_usage = (rho_np[:, 2] > 0.1).mean()
        usage_count = int(local_usage > 0.05) + int(rsu_usage > 0.05) + int(v2v_usage > 0.05)
        alpha, beta = 20.0, 80.0
        R = alpha * succ_rate - beta * avg_delay
        done = True
        info = {"avg_delay": avg_delay, "total_delay": total_delay, "worst_delay": worst_delay,
                "success_rate": succ_rate, "total_price": total_price, "avg_timeout": avg_timeout,
                "finish_times": finish_times, "local_ratio": local_usage, "rsu_ratio": rsu_usage,
                "v2v_ratio": v2v_usage, "strategy_count": usage_count}
        x_dict = {"task": normalize(self.task_feat.clone(), dim=0), "vehicle": normalize(self.veh_feat.clone(), dim=0),
                  "rsu": normalize(self.rsu_feat.clone(), dim=0)}
        edge_index_dict = self._build_edge_index_dict()
        return (x_dict, edge_index_dict), R, done, info
