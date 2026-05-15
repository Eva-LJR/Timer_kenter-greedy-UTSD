import os
from collections import defaultdict

import numpy as np
from datasets import Dataset
from tqdm import tqdm

# ==================== Config ====================
OUTPUT_ROOT = "UTSD-1G-0_1_full"
MAX_CHUNK_LEN = 250000

os.makedirs(OUTPUT_ROOT, exist_ok=True)

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

# ==================== 保存全部数据（与原代码分块逻辑一致）====================
def save_all(groups):
    print("\nSaving all data...")
    for label, sequences in tqdm(groups.items(), desc="Saving", unit="label"):
        print(f"\nSaving label: {label}")

        # 全部序列拼接
        long_sequence = np.concatenate(sequences, axis=0)
        total_len = len(long_sequence)

        print(f"   Total samples: {len(sequences)}")
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

def print_summary(groups):
    print("\n" + "=" * 50)
    print("Full data summary")
    print("=" * 50)

    total_samples = 0
    total_length = 0

    for label, sequences in groups.items():
        sample_count = len(sequences)
        total_length_label = sum(len(seq) for seq in sequences)
        total_samples += sample_count
        total_length += total_length_label
        print(f"{label:15s}: {sample_count:6d} samples, total length {total_length_label:10d}")

    print("-" * 50)
    print(f"{'Total':15s}: {total_samples:6d} samples, total length {total_length:10d}")
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

    save_all(groups)
    print_summary(groups)
    validate_output()

if __name__ == "__main__":
    main()