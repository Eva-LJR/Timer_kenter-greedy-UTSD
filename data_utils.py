import os
import numpy as np
from config import OUTPUT_ROOT, MAX_CHUNK_LEN

def save_by_label(label, sequences):
    save_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(save_dir, exist_ok=True)

    long_seq = np.concatenate(sequences, axis=0)
    total_len = len(long_seq)
    idx = 0
    start = 0

    while start < total_len:
        end = start + MAX_CHUNK_LEN
        chunk = long_seq[start:end]
        np.save(os.path.join(save_dir, f"{idx}.npy"), chunk)
        idx += 1
        start = end

    print(f"✅ {label} 保存完成，共 {idx} 个文件")

def group_sequences_by_label(data_list, label_list):
    groups = {}
    for seq, lab in zip(data_list, label_list):
        if lab not in groups:
            groups[lab] = []
        groups[lab].append(seq)
    return groups