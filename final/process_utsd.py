import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

import twed_kcenter

# ==================== Config ====================
OUTPUT_ROOT = "UTSD-1G-0_1"
# LOCAL_ARROW_PATH = os.path.join(OUTPUT_ROOT, "utsd-train.arrow")
SELECTION_RATIO = 0.1
MAX_CHUNK_LEN = 250000
DOWNSAMPLE_LEN = 500
TWED_LAMBDA = 0.1
TWED_NU = 0.001
TWED_P = 2
SEED = 42

USE_CUDA = True
PAIR_BATCH_SIZE = 4096
CUDA_PROGRESS_EVERY_BATCHES = 50
CUDA_LABEL_WORKERS = int(os.getenv("CUDA_LABEL_WORKERS", "4"))  # default 4
MIN_FILTER_SIZE = 1000  # labels smaller than this keep all samples (no filtering)

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ==================== 数据加载（本地 Arrow）====================
def find_arrow_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".arrow"):
                return os.path.join(dirpath, f)
    raise FileNotFoundError(f"No .arrow file found in {root_dir}")

def load_ds():
    cache_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset_cache",
        "UTSD-1G",
    )
    arrow_path = find_arrow_file(cache_root)
    print(f"📂 加载本地 Arrow 文件: {arrow_path}")
    ds = Dataset.from_file(arrow_path)
    return ds


def _select_one_label_task(args):
    (
        label,
        seqs_list,
        use_cuda,
        ratio,
        lambda_,
        nu,
        p,
        max_len,
        pair_batch_size,
        progress_every_batches,
        seed,
    ) = args

    t0 = time.time()
    backend = "cpu"

    if use_cuda and hasattr(twed_kcenter, "kcenter_greedy_simple_cuda"):
        try:
            selected_idx = twed_kcenter.kcenter_greedy_simple_cuda(
                seqs_list,
                ratio,
                lambda_,
                nu,
                p,
                max_len,
                pair_batch_size,
                progress_every_batches,
                seed,
            )
            backend = "cuda"
            return label, selected_idx, backend, time.time() - t0, ""
        except Exception as e:
            err = f"CUDA failed, fallback to CPU: {e}"
        else:
            err = ""
    else:
        err = ""

    selected_idx = twed_kcenter.kcenter_greedy_simple(
        seqs_list,
        ratio,
        lambda_,
        nu,
        p,
        max_len,
        seed,
    )
    return label, selected_idx, backend, time.time() - t0, err


# def load_ds():
#     if os.path.exists(LOCAL_ARROW_PATH):
#         print(f"Loading local Arrow dataset: {LOCAL_ARROW_PATH}")
#         return Dataset.from_file(LOCAL_ARROW_PATH)
#     print("Loading HuggingFace dataset...")
#     return load_dataset("thuml/UTSD", "UTSD-1G", split="train")

def load_ds():
    cache_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset_cache",
        "UTSD-1G",
    )
    arrow_path = find_arrow_file(cache_root)
    print(f"📂 加载本地 Arrow 文件: {arrow_path}")
    return Dataset.from_file(arrow_path)

def run_selection(groups):
    selected_indices_by_label = {}

    eligible = []
    for label, sequences in groups.items():
        if len(sequences) < MIN_FILTER_SIZE:
            print(f"\nProcessing label: {label}")
            print(f"   Original count: {len(sequences)}")
            print(f"   Keep all (count < {MIN_FILTER_SIZE}), skip filtering")
            selected_indices_by_label[label] = list(range(len(sequences)))
            continue
        seqs_list = [seq.tolist() for seq in sequences]
        eligible.append((label, sequences, seqs_list))

    if not eligible:
        return selected_indices_by_label

    use_parallel = USE_CUDA and CUDA_LABEL_WORKERS > 1 and len(eligible) > 1

    if use_parallel:
        print(f"\nRunning parallel label selection with {CUDA_LABEL_WORKERS} workers...")
        tasks = []
        for label, sequences, seqs_list in eligible:
            num_pairs = len(sequences) * (len(sequences) - 1) // 2
            print(f"   [{label}] count={len(sequences)}, estimated pairs={num_pairs}")
            tasks.append((
                label,
                seqs_list,
                USE_CUDA,
                SELECTION_RATIO,
                TWED_LAMBDA,
                TWED_NU,
                TWED_P,
                DOWNSAMPLE_LEN,
                PAIR_BATCH_SIZE,
                CUDA_PROGRESS_EVERY_BATCHES,
                SEED,
            ))

        with ProcessPoolExecutor(max_workers=CUDA_LABEL_WORKERS) as ex:
            futures = [ex.submit(_select_one_label_task, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Selecting labels", unit="label"):
                label, selected_idx, backend, dt, err = fut.result()
                selected_indices_by_label[label] = selected_idx
                print(f"\nProcessing label: {label}")
                if err:
                    print(f"   {err}")
                print(f"   Backend: {backend}")
                print(f"   Selected count: {len(selected_idx)}")
                print(f"   Selected idx preview: {selected_idx[:5]}")
                print(f"   Selection time: {dt:.2f}s")
    else:
        print("\nRunning K-Center + TWED selection...")
        label_items = [(l, s, sl) for (l, s, sl) in eligible]
        for label, sequences, seqs_list in tqdm(label_items, desc="Selecting", unit="label"):
            print(f"\nProcessing label: {label}")
            print(f"   Original count: {len(sequences)}")
            num_pairs = len(sequences) * (len(sequences) - 1) // 2
            print(f"   Estimated TWED pairs: {num_pairs}")

            t0 = time.time()
            label, selected_idx, backend, dt, err = _select_one_label_task((
                label,
                seqs_list,
                USE_CUDA,
                SELECTION_RATIO,
                TWED_LAMBDA,
                TWED_NU,
                TWED_P,
                DOWNSAMPLE_LEN,
                PAIR_BATCH_SIZE,
                CUDA_PROGRESS_EVERY_BATCHES,
                SEED,
            ))
            _ = t0
            selected_indices_by_label[label] = selected_idx
            if err:
                print(f"   {err}")
            print(f"   Backend: {backend}")
            print(f"   Selected count: {len(selected_idx)}")
            print(f"   Selected idx preview: {selected_idx[:5]}")
            print(f"   Selection time: {dt:.2f}s")

    return selected_indices_by_label


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
    print("Selection summary")
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


def main():
    ds = load_ds()
    print(f"Total samples: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"First sample item_id: {ds[0]['item_id']}")
    print(f"First sample target length: {len(ds[0]['target'])}")

    print("\nGrouping by label...")
    groups = defaultdict(list)
    for sample in tqdm(ds, desc="Grouping", unit="sample"):
        label = sample["item_id"].split("_")[0]
        target = np.array(sample["target"], dtype=np.float32)
        groups[label].append(target)

    print(f"Labels: {list(groups.keys())}")
    print(f"Counts: {[(k, len(v)) for k, v in groups.items()]}")

    selected_indices_by_label = run_selection(groups)
    save_selected(groups, selected_indices_by_label)
    print_summary(groups, selected_indices_by_label)
    validate_output()


if __name__ == "__main__":
    main()
