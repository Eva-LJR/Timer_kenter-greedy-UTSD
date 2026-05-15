import os
import random
from collections import defaultdict

import numpy as np
from datasets import Dataset
from tqdm import tqdm

# ==================== Config ====================
OUTPUT_ROOT = "UTSD-1G-0_1_random"
SELECTION_RATIO = 0.1
MAX_CHUNK_LEN = 250000
MIN_FILTER_SIZE = 1000   # 小于该值的标签保留全部样本
SEED = 42

os.makedirs(OUTPUT_ROOT, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ==================== 数据加载（复用原代码）====================
def find_arrow_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".arrow"):
                return os.path.join(dirpath, f)
    raise FileNotFoundError(f"No .arrow file found in {root_dir}")

def load_ds():
    cache_root = "/home/ljx/timer/timer_kcenter_utsd/new/thuml___utsd"
    arrow_path = find_arrow_file(cache_root)
    print(f"📂 加载本地 Arrow 文件: {arrow_path}")
    return Dataset.from_file(arrow_path)

# ==================== 随机选择（每个标签独立）====================
def random_select_indices(n_total, ratio, min_filter_size, seed):
    """返回选中的索引列表（保持原有顺序）"""
    if n_total < min_filter_size:
        # 全部保留
        return list(range(n_total))
    k = max(1, int(round(n_total * ratio)))
    # 随机抽样，不重复，保持原顺序（抽样后排序即可）
    indices = sorted(random.sample(range(n_total), k))
    return indices

# ==================== 保存（与原代码完全一致）====================
def save_selected(groups, selected_indices_by_label):
    print("\nSaving selected data...")
    save_items = list(selected_indices_by_label.items())
    for label, indices in tqdm(save_items, desc="Saving", unit="label"):
        print(f"\nSaving label: {label}")

        selected_sequences = [groups[label][i] for i in indices]
        long_sequence = np.concatenate(selected_sequences, axis=0)
        total_len = len(long_sequence)

        print(f"   Concatenated length: {total_len}")

        label_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(label_dir, exist_ok=True)

        chunk_idx = 0
        start = 0
        while start < total_len:
            end = min(start + MAX_CHUNK_LEN, total_len)
            chunk = long_sequence[start:end].reshape(-1, 1)

            save_path = os.path.join(label_dir, f"{chunk_idx}.npy")
            np.save(save_path, chunk)
            print(f"   Saved {chunk_idx}.npy: shape {chunk.shape}")

            chunk_idx += 1
            start = end

        print(f"   Done: {chunk_idx} files")

def print_summary(groups, selected_indices_by_label):
    print("\n" + "=" * 50)
    print("Random selection summary")
    print("=" * 50)

    total_original = 0
    total_selected = 0

    for label, sequences in groups.items():
        selected_count = len(selected_indices_by_label.get(label, []))
        original_count = len(sequences)
        total_original += original_count
        total_selected += selected_count

        ratio = (selected_count / original_count * 100) if original_count > 0 else 0
        print(f"{label:15s}: {original_count:6d} -> {selected_count:6d} ({ratio:.1f}%)")

    print("-" * 50)
    total_ratio = (total_selected / total_original * 100) if total_original > 0 else 0
    print(f"{'Total':15s}: {total_original:6d} -> {total_selected:6d} ({total_ratio:.1f}%)")
    print("=" * 50)
    print(f"\nOutput dir: {OUTPUT_ROOT}")

def validate_output():
    print("\nValidating output format...")
    for label in os.listdir(OUTPUT_ROOT):
        label_path = os.path.join(OUTPUT_ROOT, label)
        if not os.path.isdir(label_path):
            continue

        files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
        if not files:
            continue

        sample = np.load(os.path.join(label_path, files[0]))
        print(f"{label}: {len(files)} files, first shape {sample.shape}")

# ==================== 主流程 ====================
def main():
    ds = load_ds()
    print(f"Total samples: {len(ds)}")
    print(f"Columns: {ds.column_names}")

    print("\nGrouping by label...")
    groups = defaultdict(list)
    for sample in tqdm(ds, desc="Grouping", unit="sample"):
        label = sample["item_id"].split("_")[0]
        target = np.array(sample["target"], dtype=np.float32)
        groups[label].append(target)

    print(f"Labels: {list(groups.keys())}")
    print(f"Counts: {[(k, len(v)) for k, v in groups.items()]}")

    # 随机选择每个标签下的索引
    selected_indices_by_label = {}
    for label, sequences in groups.items():
        n = len(sequences)
        indices = random_select_indices(n, SELECTION_RATIO, MIN_FILTER_SIZE, SEED)
        selected_indices_by_label[label] = indices
        print(f"\nLabel {label}: {n} -> {len(indices)} selected")

    save_selected(groups, selected_indices_by_label)
    print_summary(groups, selected_indices_by_label)
    validate_output()

if __name__ == "__main__":
    main()