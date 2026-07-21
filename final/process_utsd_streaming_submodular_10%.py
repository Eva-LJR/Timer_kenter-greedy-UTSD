#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTSD 流式亚模最大化筛选（算法 B，可直接运行）

安装依赖：
    python -m pip install numpy tqdm datasets==3.3.2 dtaidistance

默认运行（脚本放在 Timer_kenter-greedy-UTSD/final/ 下时）：
    python process_utsd_streaming_submodular.py

断点恢复：
    python process_utsd_streaming_submodular.py --resume

小规模试跑：
    python process_utsd_streaming_submodular.py \
        --max-samples 1000 --anchor-size 32 --downsample-len 128 \
        --output-root ./output/algorithmB_smoke

设计说明：
1. 文档给出的集合函数是设施选址函数（facility location），但伪代码中的
   1-exp(-d_min^2/gamma) 并不是该集合函数的严格边际增益。
2. 为了让文档中的 Sieve-Streaming 近似保证与代码一致，本实现固定一个通过
   蓄水池采样获得的参考锚点集 A，并优化以下代理设施选址目标：

       f_A(S) = sum_{a in A} max_{s in S} exp(-DTW(a,s)^p / gamma)

   它是单调非负亚模函数；代码计算的是它的精确边际增益。
3. 数据只顺序扫描，不把整个 UTSD 加载到内存。筛选结束后再次扫描 Arrow，
   将被选中的原始 target 按 OpenLTM 的 UTSD_Npy 格式保存为 (length, 1) 的 .npy。
4. DTW 使用 dtaidistance 的 C/OpenMP 实现；按 batch 一次计算“当前样本 × 锚点”
   的交叉距离矩阵，避免 Python 内层循环，并启用约束窗口与内部剪枝。
5. 若得分最高的流式过滤器不足 K 个样本，则对其余候选执行懒惰贪心：
   每一步按当前覆盖下的精确设施选址边际增益选择样本，直至严格补满 K。
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import pickle
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - 给缺依赖时提供清晰报错
    raise SystemExit("缺少 tqdm，请执行：python -m pip install tqdm") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "dataset_cache" / "UTSD-1G"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "algorithmB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Sieve-Streaming + DTW 设施选址目标筛选 UTSD"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="包含 utsd-train.arrow 的目录；会递归查找 .arrow 文件",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="算法B输出根目录",
    )
    parser.add_argument(
        "--selection-ratio",
        type=float,
        default=0.1,
        help="每个标签的子集上限比例，默认 0.1",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Sieve 估计阈值的几何间隔，默认 0.1",
    )
    parser.add_argument(
        "--anchor-size",
        type=int,
        default=128,
        help="每个标签的设施选址参考锚点数，越大越准确但越慢",
    )
    parser.add_argument(
        "--downsample-len",
        type=int,
        default=500,
        help="仅用于DTW评分的统一下采样长度；原始序列保存时不下采样",
    )
    parser.add_argument(
        "--window-ratio",
        type=float,
        default=0.1,
        help="DTW Sakoe-Chiba 窗口比例 R/L，文档默认 0.1",
    )
    parser.add_argument(
        "--kernel-power",
        type=int,
        choices=(1, 2),
        default=2,
        help="相似度核使用 DTW^p；文档伪代码对应 p=2",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="相似度核带宽；<=0 时按每个标签锚点距离中位数自动估计",
    )
    parser.add_argument(
        "--gamma-pairs",
        type=int,
        default=1024,
        help="自动估计 gamma 时每个标签最多使用的锚点距离对数",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=64,
        help="Arrow流式读取和交叉DTW计算的batch大小",
    )
    parser.add_argument(
        "--max-chunk-len",
        type=int,
        default=250000,
        help="输出单个 .npy 文件的最大时间点数，与K-Center版本一致",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help="只处理指定标签，逗号分隔；空字符串表示全部标签",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="固定随机种子"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="每处理多少个batch保存一次断点；0表示关闭",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 output-root/checkpoint.pkl 恢复筛选阶段",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出数据目录；默认遇到旧结果会停止",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="仅处理前N个样本，用于小规模试跑；0表示完整数据集",
    )
    parser.add_argument(
        "--no-dtw-parallel",
        action="store_true",
        help="关闭 dtaidistance 的 OpenMP 并行",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.selection_ratio <= 1:
        raise ValueError("selection-ratio 必须在 (0, 1] 内")
    if not 0 < args.epsilon <= 1:
        raise ValueError("epsilon 必须在 (0, 1] 内")
    if args.anchor_size <= 0:
        raise ValueError("anchor-size 必须大于0")
    if args.downsample_len < 4:
        raise ValueError("downsample-len 至少为4")
    if not 0 < args.window_ratio <= 1:
        raise ValueError("window-ratio 必须在 (0, 1] 内")
    if args.stream_batch_size <= 0:
        raise ValueError("stream-batch-size 必须大于0")
    if args.max_chunk_len <= 0:
        raise ValueError("max-chunk-len 必须大于0")
    if args.max_samples < 0:
        raise ValueError("max-samples 不能为负数")


def import_datasets():
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "缺少 datasets，请执行：python -m pip install datasets==3.3.2"
        ) from exc
    return Dataset


def import_dtw():
    try:
        from dtaidistance import dtw
    except ImportError as exc:
        raise SystemExit(
            "缺少 dtaidistance，请执行：python -m pip install dtaidistance"
        ) from exc
    return dtw


def find_arrow_file(root_dir: Path) -> Path:
    """沿用算法A的本地Arrow逻辑，但避免误选lock或临时文件。"""
    root_dir = root_dir.expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{root_dir}")

    files = sorted(p for p in root_dir.rglob("*.arrow") if p.is_file())
    if not files:
        raise FileNotFoundError(f"在 {root_dir} 下没有找到 .arrow 文件")

    preferred = [p for p in files if p.name == "utsd-train.arrow"]
    if preferred:
        return preferred[0]
    if len(files) > 1:
        print(f"⚠️ 找到多个Arrow文件，按排序选择：{files[0]}")
    return files[0]


def extract_label(item_id: str) -> str:
    """与K-Center代码一致：item_id第一个下划线前为标签。"""
    return str(item_id).split("_", 1)[0]


def parse_label_filter(value: str) -> Optional[set[str]]:
    labels = {x.strip() for x in value.split(",") if x.strip()}
    return labels or None


def clean_and_downsample(values: Sequence[float], target_len: int) -> np.ndarray:
    """清洗、线性下采样、逐序列z-normalize，返回C连续float64数组。"""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(target_len, dtype=np.float64)

    finite = np.isfinite(x)
    if not finite.all():
        if not finite.any():
            x = np.zeros_like(x)
        else:
            idx = np.arange(x.size)
            x = np.interp(idx, idx[finite], x[finite])

    if x.size == 1:
        y = np.full(target_len, x[0], dtype=np.float64)
    elif x.size == target_len:
        y = x.copy()
    else:
        old_grid = np.linspace(0.0, 1.0, num=x.size)
        new_grid = np.linspace(0.0, 1.0, num=target_len)
        y = np.interp(new_grid, old_grid, x)

    mean = float(y.mean())
    std = float(y.std())
    if std > 1e-12:
        y = (y - mean) / std
    else:
        y = y - mean
    return np.ascontiguousarray(y, dtype=np.float64)


def iter_dataset_batches(dataset, limit: int, batch_size: int):
    """按索引顺序批量读取Arrow，避免一次性构造全部Python对象。"""
    for start in range(0, limit, batch_size):
        end = min(start + batch_size, limit)
        batch = dataset[start:end]
        yield start, end, batch


@dataclass
class AnchorPool:
    sequences: List[np.ndarray]
    source_indices: List[int]
    seen: int = 0

    def consider(
        self,
        sequence: np.ndarray,
        source_index: int,
        capacity: int,
        rng: random.Random,
    ) -> None:
        """标准蓄水池采样：仅保存固定数量参考序列。"""
        self.seen += 1
        if len(self.sequences) < capacity:
            self.sequences.append(sequence)
            self.source_indices.append(source_index)
            return
        replace_at = rng.randrange(self.seen)
        if replace_at < capacity:
            self.sequences[replace_at] = sequence
            self.source_indices[replace_at] = source_index


def first_pass_build_anchors(
    dataset,
    limit: int,
    batch_size: int,
    anchor_size: int,
    downsample_len: int,
    label_filter: Optional[set[str]],
    seed: int,
) -> Tuple[Dict[str, int], Dict[str, AnchorPool]]:
    """算法B准备步骤：流式统计标签数量，并为每个标签构造固定锚点集。"""
    counts: Dict[str, int] = defaultdict(int)
    pools: Dict[str, AnchorPool] = {}
    rngs: Dict[str, random.Random] = {}

    bar = tqdm(total=limit, desc="Pass 1/3: counts + anchors", unit="sample")
    for start, end, batch in iter_dataset_batches(dataset, limit, batch_size):
        item_ids = batch["item_id"]
        targets = batch["target"]
        for offset, (item_id, target) in enumerate(zip(item_ids, targets)):
            label = extract_label(item_id)
            if label_filter is not None and label not in label_filter:
                continue
            global_index = start + offset
            counts[label] += 1
            if label not in pools:
                pools[label] = AnchorPool([], [])
                # Python hash每次进程可能变化，改用稳定的字符和构造标签种子。
                label_seed = seed + sum((i + 1) * ord(c) for i, c in enumerate(label))
                rngs[label] = random.Random(label_seed)
            seq = clean_and_downsample(target, downsample_len)
            pools[label].consider(seq, global_index, anchor_size, rngs[label])
        bar.update(end - start)
    bar.close()

    if not counts:
        raise RuntimeError("没有找到符合条件的UTSD样本，请检查labels和数据文件")
    return dict(counts), pools


class DtwBatchEngine:
    """使用C/OpenMP实现批量约束DTW；只计算候选样本与锚点的交叉块。"""

    def __init__(
        self,
        downsample_len: int,
        window_ratio: float,
        parallel: bool,
    ) -> None:
        self.dtw = import_dtw()
        self.length = downsample_len
        self.window = max(1, int(round(downsample_len * window_ratio)))
        self.parallel = parallel

    def cross_distances(
        self, candidates: np.ndarray, anchors: np.ndarray
    ) -> np.ndarray:
        if candidates.ndim != 2 or anchors.ndim != 2:
            raise ValueError("candidates和anchors必须是二维矩阵")
        n_candidates = candidates.shape[0]
        n_anchors = anchors.shape[0]
        if n_candidates == 0 or n_anchors == 0:
            return np.empty((n_candidates, n_anchors), dtype=np.float64)

        series = np.ascontiguousarray(
            np.vstack((candidates, anchors)), dtype=np.float64
        )
        block = ((0, n_candidates), (n_candidates, n_candidates + n_anchors))
        matrix = self.dtw.distance_matrix_fast(
            series,
            window=self.window,
            use_pruning=True,
            block=block,
            compact=False,
            parallel=self.parallel,
            only_triu=True,
        )
        matrix = np.asarray(matrix, dtype=np.float64)
        result = matrix[:n_candidates, n_candidates : n_candidates + n_anchors]
        if result.shape != (n_candidates, n_anchors):
            raise RuntimeError(
                f"DTW交叉矩阵形状异常：{result.shape}，预期 {(n_candidates, n_anchors)}"
            )
        if not np.isfinite(result).all():
            bad = int((~np.isfinite(result)).sum())
            raise RuntimeError(f"DTW结果中存在 {bad} 个非有限值")
        return result

    def anchor_pair_distances(self, anchors: np.ndarray) -> np.ndarray:
        n = anchors.shape[0]
        if n < 2:
            return np.empty(0, dtype=np.float64)
        matrix = self.dtw.distance_matrix_fast(
            np.ascontiguousarray(anchors, dtype=np.float64),
            window=self.window,
            use_pruning=True,
            compact=False,
            parallel=self.parallel,
            only_triu=True,
        )
        matrix = np.asarray(matrix, dtype=np.float64)
        return matrix[np.triu_indices(n, k=1)]


def estimate_gamma(
    anchor_matrix: np.ndarray,
    engine: DtwBatchEngine,
    kernel_power: int,
    max_pairs: int,
    seed: int,
) -> float:
    distances = engine.anchor_pair_distances(anchor_matrix)
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size == 0:
        return 1.0
    if max_pairs > 0 and distances.size > max_pairs:
        rng = np.random.default_rng(seed)
        distances = rng.choice(distances, size=max_pairs, replace=False)
    powered = np.power(distances, kernel_power)
    gamma = float(np.median(powered))
    if not math.isfinite(gamma) or gamma <= 1e-12:
        gamma = float(np.mean(powered)) if powered.size else 1.0
    return max(gamma, 1e-12)


def build_thresholds(lower: float, upper: float, epsilon: float) -> np.ndarray:
    """对应算法B第1步：T={(1+epsilon)^i | lower<=v<=upper}。"""
    if upper < lower:
        upper = lower
    base = 1.0 + epsilon
    exponent = math.ceil(math.log(lower, base)) if lower > 0 else 0
    values: List[float] = []
    value = base**exponent
    while value <= upper * (1.0 + 1e-12):
        values.append(float(value))
        value *= base
    if not values:
        values = [float(lower)]
    return np.asarray(values, dtype=np.float64)


class SieveFilterBank:
    """一个标签对应的一组Sieve过滤器；过滤器维度使用NumPy同时更新。"""

    def __init__(
        self,
        label: str,
        budget: int,
        anchor_count: int,
        epsilon: float,
    ) -> None:
        self.label = label
        self.budget = budget
        # f_A(S)最大不超过anchor_count，因此v>anchor_count的过滤器必然无用。
        self.thresholds = build_thresholds(1.0, float(anchor_count), epsilon)
        n_filters = self.thresholds.size
        self.coverage = np.zeros((n_filters, anchor_count), dtype=np.float32)
        self.scores = np.zeros(n_filters, dtype=np.float64)
        self.sizes = np.zeros(n_filters, dtype=np.int64)
        self.selected_indices: List[List[int]] = [[] for _ in range(n_filters)]
        self.accepted_gains: List[List[float]] = [[] for _ in range(n_filters)]
        # 补齐阶段需要重算“当前覆盖下”的边际增益。每个样本仅保存一个
        # anchor_count维相似度向量，不保存原始序列或全量两两距离矩阵。
        self.candidate_indices: List[int] = []
        self.candidate_similarities: List[np.ndarray] = []
        self.singleton_gains: List[float] = []

    def update(self, similarity: np.ndarray, global_index: int) -> None:
        """对应算法B第4~12步：计算边际增益、动态阈值并执行准入。"""
        sim = np.asarray(similarity, dtype=np.float32)
        self.candidate_indices.append(int(global_index))
        self.candidate_similarities.append(sim.copy())
        self.singleton_gains.append(float(sim.sum(dtype=np.float64)))
        # 精确设施选址边际增益：sum(max(0, sim(x,a)-current_best(a)))
        gains = np.maximum(sim[None, :] - self.coverage, 0.0).sum(
            axis=1, dtype=np.float64
        )
        remaining = self.budget - self.sizes
        active = remaining > 0
        required = np.full_like(self.scores, np.inf)
        required[active] = (
            self.thresholds[active] / 2.0 - self.scores[active]
        ) / remaining[active]

        # 零增益元素不会改善目标，跳过不影响目标值和近似结果。
        accepted = np.flatnonzero(active & (gains >= required) & (gains > 1e-12))
        for filter_index in accepted.tolist():
            self.coverage[filter_index] = np.maximum(
                self.coverage[filter_index], sim
            )
            gain = float(gains[filter_index])
            self.scores[filter_index] += gain
            self.sizes[filter_index] += 1
            self.selected_indices[filter_index].append(int(global_index))
            self.accepted_gains[filter_index].append(gain)

    def _fill_filter_to_budget(self, filter_index: int) -> Dict[str, object]:
        """从未选候选中按当前真实边际增益补齐到K（懒惰贪心）。

        堆中的缓存收益是边际增益上界。设施选址函数满足收益递减，因此
        只需重算堆顶；当其重算值不小于其余候选上界时，它就是当前具有
        最大真实边际增益的候选。
        """
        selected_indices = list(self.selected_indices[filter_index])
        accepted_gains = list(self.accepted_gains[filter_index])
        stream_selected_count = len(selected_indices)
        selected_set = set(selected_indices)
        coverage = self.coverage[filter_index].copy()
        score = float(self.scores[filter_index])

        # (-收益上界, global_index, candidate_position)，索引使并列可复现。
        heap = [
            (-gain, index, position)
            for position, (index, gain) in enumerate(
                zip(self.candidate_indices, self.singleton_gains)
            )
            if index not in selected_set
        ]
        heapq.heapify(heap)

        while len(selected_indices) < self.budget:
            while heap and heap[0][1] in selected_set:
                heapq.heappop(heap)
            if not heap:
                raise RuntimeError(
                    f"标签{self.label}无法补齐到K={self.budget}：候选样本不足"
                )

            _, global_index, position = heapq.heappop(heap)
            sim = self.candidate_similarities[position]
            exact_gain = float(
                np.maximum(sim - coverage, 0.0).sum(dtype=np.float64)
            )

            while heap and heap[0][1] in selected_set:
                heapq.heappop(heap)
            next_upper = -heap[0][0] if heap else -math.inf
            tolerance = 1e-12 * max(1.0, abs(exact_gain))

            if exact_gain + tolerance >= next_upper:
                selected_set.add(global_index)
                selected_indices.append(int(global_index))
                accepted_gains.append(exact_gain)
                coverage = np.maximum(coverage, sim)
                score += exact_gain
            else:
                heapq.heappush(heap, (-exact_gain, global_index, position))

        return {
            "selected_indices": selected_indices,
            "accepted_gains": accepted_gains,
            "score": score,
            "stream_selected_count": stream_selected_count,
            "backfilled_count": len(selected_indices) - stream_selected_count,
        }

    def best_result(self) -> Dict[str, object]:
        # 先最大化流式目标值；同分时优先样本更多、阈值更小的过滤器。
        candidates = list(range(self.thresholds.size))
        best = max(
            candidates,
            key=lambda i: (self.scores[i], self.sizes[i], -self.thresholds[i]),
        )
        filled = self._fill_filter_to_budget(best)
        return {
            "label": self.label,
            "filter_index": int(best),
            "threshold_v": float(self.thresholds[best]),
            "score": float(filled["score"]),
            "normalized_score": float(
                filled["score"] / self.coverage.shape[1]
            ),
            "budget": int(self.budget),
            "selected_indices": filled["selected_indices"],
            "accepted_gains": filled["accepted_gains"],
            "stream_selected_count": int(filled["stream_selected_count"]),
            "backfilled_count": int(filled["backfilled_count"]),
            "backfill_method": "lazy_greedy_exact_marginal_gain",
            "filter_count": int(self.thresholds.size),
        }


def similarities_from_distances(
    distances: np.ndarray, gamma: float, kernel_power: int
) -> np.ndarray:
    powered = np.power(distances, kernel_power)
    return np.exp(-powered / gamma).astype(np.float32, copy=False)


def checkpoint_signature(
    dataset_fingerprint: str,
    limit: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "dataset_fingerprint": dataset_fingerprint,
        "limit": limit,
        "selection_ratio": args.selection_ratio,
        "epsilon": args.epsilon,
        "anchor_size": args.anchor_size,
        "downsample_len": args.downsample_len,
        "window_ratio": args.window_ratio,
        "kernel_power": args.kernel_power,
        "gamma": args.gamma,
        "seed": args.seed,
        "algorithm_version": "sieve_then_lazy_greedy_fill_v2",
        "labels": args.labels,
    }


def atomic_pickle_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_checkpoint(
    path: Path,
    signature: Mapping[str, object],
    next_start: int,
    counts: Mapping[str, int],
    anchors: Mapping[str, np.ndarray],
    anchor_indices: Mapping[str, Sequence[int]],
    gammas: Mapping[str, float],
    banks: Mapping[str, SieveFilterBank],
) -> None:
    atomic_pickle_dump(
        {
            "signature": dict(signature),
            "next_start": int(next_start),
            "counts": dict(counts),
            "anchors": dict(anchors),
            "anchor_indices": {k: list(v) for k, v in anchor_indices.items()},
            "gammas": dict(gammas),
            "banks": dict(banks),
        },
        path,
    )


def load_checkpoint(path: Path, expected_signature: Mapping[str, object]):
    if not path.exists():
        raise FileNotFoundError(f"没有找到断点文件：{path}")
    with path.open("rb") as handle:
        state = pickle.load(handle)
    if state.get("signature") != dict(expected_signature):
        raise RuntimeError(
            "断点参数与当前命令不一致。请使用原参数恢复，或不加--resume重新运行。"
        )
    return state


def second_pass_select(
    dataset,
    limit: int,
    args: argparse.Namespace,
    counts: Mapping[str, int],
    anchors: Mapping[str, np.ndarray],
    gammas: Mapping[str, float],
    banks: Mapping[str, SieveFilterBank],
    engine: DtwBatchEngine,
    start_at: int,
    checkpoint_path: Path,
    signature: Mapping[str, object],
    anchor_indices: Mapping[str, Sequence[int]],
) -> None:
    """算法B第2~14步：按Arrow原顺序扫描并更新所有标签过滤器。"""
    if start_at >= limit:
        return
    total = limit - start_at
    bar = tqdm(total=total, desc="Pass 2/3: Sieve-Streaming", unit="sample")
    batch_counter = 0

    for start in range(start_at, limit, args.stream_batch_size):
        end = min(start + args.stream_batch_size, limit)
        batch = dataset[start:end]
        grouped_positions: Dict[str, List[int]] = defaultdict(list)
        prepared: Dict[int, np.ndarray] = {}

        for offset, (item_id, target) in enumerate(
            zip(batch["item_id"], batch["target"])
        ):
            label = extract_label(item_id)
            if label not in banks:
                continue
            pos = start + offset
            grouped_positions[label].append(pos)
            prepared[pos] = clean_and_downsample(target, args.downsample_len)

        similarity_by_position: Dict[int, np.ndarray] = {}
        for label, positions in grouped_positions.items():
            candidate_matrix = np.stack([prepared[p] for p in positions], axis=0)
            distances = engine.cross_distances(candidate_matrix, anchors[label])
            similarities = similarities_from_distances(
                distances, gammas[label], args.kernel_power
            )
            for row, position in enumerate(positions):
                similarity_by_position[position] = similarities[row]

        # 严格按照数据流原顺序执行准入，不能按标签重排。
        for position in range(start, end):
            if position in similarity_by_position:
                label = extract_label(batch["item_id"][position - start])
                banks[label].update(similarity_by_position[position], position)

        bar.update(end - start)
        batch_counter += 1
        if (
            args.checkpoint_every > 0
            and batch_counter % args.checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoint_path,
                signature,
                end,
                counts,
                anchors,
                anchor_indices,
                gammas,
                banks,
            )
            bar.set_postfix_str(f"checkpoint={end}/{limit}")
    bar.close()


class NpyChunkWriter:
    """与K-Center保存逻辑一致：拼接原始序列并保存为(length,1)。"""

    def __init__(self, root: Path, max_chunk_len: int) -> None:
        self.root = root
        self.max_chunk_len = max_chunk_len
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.buffer_lengths: Dict[str, int] = defaultdict(int)
        self.file_indices: Dict[str, int] = defaultdict(int)
        self.total_lengths: Dict[str, int] = defaultdict(int)

    def add(self, label: str, target: Sequence[float]) -> None:
        arr = np.asarray(target, dtype=np.float32).reshape(-1, 1)
        if arr.size == 0:
            return
        self.buffers[label].append(arr)
        self.buffer_lengths[label] += len(arr)
        self.total_lengths[label] += len(arr)
        if self.buffer_lengths[label] >= self.max_chunk_len:
            self._flush_full_chunks(label)

    def _save_chunk(self, label: str, chunk: np.ndarray) -> None:
        label_dir = self.root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        index = self.file_indices[label]
        np.save(label_dir / f"{index}.npy", chunk.astype(np.float32, copy=False))
        self.file_indices[label] += 1

    def _flush_full_chunks(self, label: str) -> None:
        if not self.buffers[label]:
            return
        merged = np.concatenate(self.buffers[label], axis=0)
        start = 0
        while len(merged) - start >= self.max_chunk_len:
            end = start + self.max_chunk_len
            self._save_chunk(label, merged[start:end])
            start = end
        remainder = merged[start:]
        self.buffers[label] = [remainder] if len(remainder) else []
        self.buffer_lengths[label] = len(remainder)

    def finish(self) -> None:
        for label in list(self.buffers):
            if self.buffers[label]:
                merged = np.concatenate(self.buffers[label], axis=0)
                if len(merged):
                    self._save_chunk(label, merged)
            self.buffers[label] = []
            self.buffer_lengths[label] = 0


def prepare_dataset_output(dataset_dir: Path, overwrite: bool) -> None:
    if dataset_dir.exists() and any(dataset_dir.rglob("*.npy")):
        if not overwrite:
            raise FileExistsError(
                f"输出数据目录已有NPY文件：{dataset_dir}\n"
                "如确认覆盖，请重新运行并添加 --overwrite"
            )
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)


def third_pass_save(
    dataset,
    limit: int,
    batch_size: int,
    dataset_dir: Path,
    max_chunk_len: int,
    results: Mapping[str, Mapping[str, object]],
) -> Tuple[List[Tuple[int, str, str, float]], NpyChunkWriter]:
    """算法B第15步之后：再次流式扫描，只保存最终最优过滤器选中的原始数据。"""
    index_to_info: Dict[int, Tuple[str, float]] = {}
    for label, result in results.items():
        indices = result["selected_indices"]
        gains = result["accepted_gains"]
        for index, gain in zip(indices, gains):
            index_to_info[int(index)] = (label, float(gain))

    writer = NpyChunkWriter(dataset_dir, max_chunk_len)
    records: List[Tuple[int, str, str, float]] = []
    bar = tqdm(total=limit, desc="Pass 3/3: save selected NPY", unit="sample")
    for start, end, batch in iter_dataset_batches(dataset, limit, batch_size):
        for offset, (item_id, target) in enumerate(
            zip(batch["item_id"], batch["target"])
        ):
            index = start + offset
            info = index_to_info.get(index)
            if info is None:
                continue
            label, gain = info
            writer.add(label, target)
            records.append((index, str(item_id), label, gain))
        bar.update(end - start)
    bar.close()
    writer.finish()
    return records, writer


def write_metadata(
    output_root: Path,
    records: Sequence[Tuple[int, str, str, float]],
    counts: Mapping[str, int],
    results: Mapping[str, Mapping[str, object]],
    gammas: Mapping[str, float],
    anchor_indices: Mapping[str, Sequence[int]],
    writer: NpyChunkWriter,
    args: argparse.Namespace,
    arrow_path: Path,
    elapsed_seconds: float,
) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    indices = np.asarray([r[0] for r in records], dtype=np.int64)
    weights = np.asarray([r[3] for r in records], dtype=np.float64)
    np.save(metadata_dir / "selected_indices.npy", indices)
    np.save(metadata_dir / "selected_weights.npy", weights)

    with (metadata_dir / "selected_ids.txt").open("w", encoding="utf-8") as f:
        f.write("global_index\titem_id\tlabel\tmarginal_gain\n")
        for index, item_id, label, gain in records:
            f.write(f"{index}\t{item_id}\t{label}\t{gain:.10g}\n")

    anchors_json = {k: [int(x) for x in v] for k, v in anchor_indices.items()}
    with (metadata_dir / "anchor_indices.json").open("w", encoding="utf-8") as f:
        json.dump(anchors_json, f, ensure_ascii=False, indent=2)

    label_summaries: Dict[str, Dict[str, object]] = {}
    for label in sorted(results):
        result = results[label]
        label_summaries[label] = {
            "original_count": int(counts[label]),
            "budget": int(result["budget"]),
            "selected_count": len(result["selected_indices"]),
            "selected_ratio": len(result["selected_indices"]) / counts[label],
            "stream_selected_count": int(result["stream_selected_count"]),
            "backfilled_count": int(result["backfilled_count"]),
            "backfill_method": str(result["backfill_method"]),
            "facility_score": float(result["score"]),
            "normalized_facility_score": float(result["normalized_score"]),
            "chosen_threshold_v": float(result["threshold_v"]),
            "filter_count": int(result["filter_count"]),
            "gamma": float(gammas[label]),
            "npy_files": int(writer.file_indices[label]),
            "timepoints": int(writer.total_lengths[label]),
        }

    summary = {
        "algorithm": (
            "Sieve-Streaming with anchor-based DTW facility location, "
            "then lazy-greedy marginal-gain fill to K"
        ),
        "arrow_path": str(arrow_path),
        "dataset_output": str(output_root / "dataset"),
        "seed": args.seed,
        "selection_ratio_target": args.selection_ratio,
        "epsilon": args.epsilon,
        "anchor_size": args.anchor_size,
        "downsample_len": args.downsample_len,
        "dtw_window_ratio": args.window_ratio,
        "kernel_power": args.kernel_power,
        "gamma_mode": "fixed" if args.gamma > 0 else "auto_median",
        "max_chunk_len": args.max_chunk_len,
        "processed_samples": int(sum(counts.values())),
        "selected_samples": int(len(records)),
        "elapsed_seconds": float(elapsed_seconds),
        "labels": label_summaries,
    }

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "UTSD 流式亚模最大化筛选摘要",
        "=" * 72,
        f"输入Arrow: {arrow_path}",
        f"训练数据目录: {output_root / 'dataset'}",
        f"总处理样本: {sum(counts.values())}",
        f"总选中样本: {len(records)}",
        f"总耗时: {elapsed_seconds / 3600:.2f} 小时",
        "",
        "各标签统计：",
    ]
    for label, info in label_summaries.items():
        lines.append(
            f"- {label}: {info['original_count']} -> {info['selected_count']} "
            f"({100 * info['selected_ratio']:.2f}%), K={info['budget']}, "
            f"stream={info['stream_selected_count']}, "
            f"filled={info['backfilled_count']}, "
            f"score={info['facility_score']:.6f}, gamma={info['gamma']:.6g}, "
            f"npy={info['npy_files']}"
        )
    text = "\n".join(lines) + "\n"
    (output_root / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    started_at = time.time()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_root / "checkpoint.pkl"
    dataset_dir = output_root / "dataset"

    arrow_path = find_arrow_file(args.data_root)
    print(f"📂 Arrow: {arrow_path}")
    Dataset = import_datasets()
    dataset = Dataset.from_file(str(arrow_path))
    required_columns = {"item_id", "target"}
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise KeyError(f"Arrow缺少必要字段：{sorted(missing)}")

    limit = len(dataset)
    if args.max_samples > 0:
        limit = min(limit, args.max_samples)
    print(f"📊 处理样本数: {limit}/{len(dataset)}")
    print(f"📋 列名: {dataset.column_names}")

    fingerprint = str(getattr(dataset, "_fingerprint", arrow_path.stat().st_mtime_ns))
    signature = checkpoint_signature(fingerprint, limit, args)
    engine = DtwBatchEngine(
        args.downsample_len,
        args.window_ratio,
        parallel=not args.no_dtw_parallel,
    )
    print(
        f"⚙️ DTW: C/OpenMP={not args.no_dtw_parallel}, "
        f"length={args.downsample_len}, window={engine.window}"
    )

    if args.resume:
        state = load_checkpoint(checkpoint_path, signature)
        counts = state["counts"]
        anchors = state["anchors"]
        anchor_indices = state["anchor_indices"]
        gammas = state["gammas"]
        banks = state["banks"]
        start_at = int(state["next_start"])
        print(f"♻️ 从断点恢复: {start_at}/{limit}")
    else:
        label_filter = parse_label_filter(args.labels)
        counts, pools = first_pass_build_anchors(
            dataset,
            limit,
            args.stream_batch_size,
            args.anchor_size,
            args.downsample_len,
            label_filter,
            args.seed,
        )
        anchors = {
            label: np.ascontiguousarray(np.stack(pool.sequences), dtype=np.float64)
            for label, pool in pools.items()
        }
        anchor_indices = {
            label: list(pool.source_indices) for label, pool in pools.items()
        }
        gammas: Dict[str, float] = {}
        banks: Dict[str, SieveFilterBank] = {}

        print("\n🔧 初始化各标签gamma和Sieve过滤器")
        for label in sorted(counts):
            gamma = (
                float(args.gamma)
                if args.gamma > 0
                else estimate_gamma(
                    anchors[label],
                    engine,
                    args.kernel_power,
                    args.gamma_pairs,
                    args.seed + sum(ord(c) for c in label),
                )
            )
            gammas[label] = gamma
            budget = max(1, int(math.floor(counts[label] * args.selection_ratio)))
            banks[label] = SieveFilterBank(
                label, budget, anchors[label].shape[0], args.epsilon
            )
            print(
                f"  {label}: N={counts[label]}, K={budget}, "
                f"anchors={anchors[label].shape[0]}, gamma={gamma:.6g}, "
                f"filters={banks[label].thresholds.size}"
            )
        start_at = 0

    second_pass_select(
        dataset,
        limit,
        args,
        counts,
        anchors,
        gammas,
        banks,
        engine,
        start_at,
        checkpoint_path,
        signature,
        anchor_indices,
    )

    # 对应算法B第15步：选择最优过滤器；不足K时按真实边际增益补齐。
    results: Dict[str, Dict[str, object]] = {}
    for label in tqdm(
        sorted(banks), desc="Post: select best + fill to K", unit="label"
    ):
        results[label] = banks[label].best_result()
    print("\n🏆 最优过滤器")
    for label in sorted(results):
        result = results[label]
        print(
            f"  {label}: selected={len(result['selected_indices'])}/"
            f"{result['budget']}, score={result['score']:.6f}, "
            f"stream={result['stream_selected_count']}, "
            f"filled={result['backfilled_count']}, "
            f"v={result['threshold_v']:.6g}"
        )

    prepare_dataset_output(dataset_dir, args.overwrite)
    records, writer = third_pass_save(
        dataset,
        limit,
        args.stream_batch_size,
        dataset_dir,
        args.max_chunk_len,
        results,
    )
    elapsed = time.time() - started_at
    write_metadata(
        output_root,
        records,
        counts,
        results,
        gammas,
        anchor_indices,
        writer,
        args,
        arrow_path,
        elapsed,
    )

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print("✅ 算法B筛选完成")
    print(f"✅ OpenLTM --root_path 应设置为: {dataset_dir}")


if __name__ == "__main__":
    main()
