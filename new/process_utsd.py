import os
import numpy as np
from datasets import load_dataset, Dataset
from collections import defaultdict
from tqdm import tqdm
import twed_kcenter  # 导入 C++ 模块

# ==================== 配置 ====================
OUTPUT_ROOT = "UTSD-1G-0_1"
SELECTION_RATIO = 0.1  # 10%
MAX_CHUNK_LEN = 250000
DOWNSAMPLE_LEN = 500  # 下采样长度（用于 TWED 计算）
TWED_LAMBDA = 0.1
TWED_NU = 0.001
TWED_P = 2
SEED = 42

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.makedirs(OUTPUT_ROOT, exist_ok=True)



# ==================== 1. 加载数据 ====================
def find_arrow_file(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".arrow"):
                return os.path.join(dirpath, f)
    raise FileNotFoundError(f"No .arrow file found in {root_dir}")

print("📂 查找本地 Arrow 文件...")
cache_root = "/home/ljx/timer/timer_kcenter_utsd/new/thuml___utsd"
arrow_path = find_arrow_file(cache_root)
print(f"找到 Arrow 文件: {arrow_path}")

ds = Dataset.from_file(arrow_path)
print(f"总样本数: {len(ds)}")
print(f"列名: {ds.column_names}")

# 验证第一个样本
print(f"第一个样本 item_id: {ds[0]['item_id']}")
print(f"第一个样本 target 长度: {len(ds[0]['target'])}")
# print("📂 加载 HuggingFace 数据集...")
# # ds = load_dataset("thuml/UTSD", "UTSD-1G", split="train")
# ds = load_dataset("/home/ljx/timer/timer_kcenter_utsd/new/thuml___utsd/", "UTSD-1G", split="train")

# print(f"总样本数: {len(ds)}")

# ==================== 2. 按标签分组 ====================
print("\n📊 按标签分组...")
groups = defaultdict(list)

for idx, sample in enumerate(tqdm(ds, desc="分组")):
    # 提取标签（从 item_id 的第一个下划线前部分）
    label = sample['item_id'].split('_')[0]
    # 提取序列值
    target = np.array(sample['target'], dtype=np.float32)
    groups[label].append(target)

print(f"发现标签: {list(groups.keys())}")
print(f"各标签样本数: {[(k, len(v)) for k, v in groups.items()]}")

# ==================== 3. 每个标签独立筛选 ====================
print("\n🔍 开始 K-Center + TWED 筛选...")

selected_indices_by_label = {}

for label, sequences in groups.items():
    print(f"\n📍 处理标签: {label}")
    print(f"   原始样本数: {len(sequences)}")
    
    if len(sequences) < 5:
        print(f"   ⚠️ 样本太少，跳过")
        continue
    
    # 准备数据格式（C++ 模块需要 list of list）
    # 调用 C++ K-Center Greedy（内部使用 TWED）
    seqs_list = [seq.tolist() for seq in sequences]  # 已经是 list of list
    selected_idx = twed_kcenter.kcenter_greedy_simple(
        seqs_list,  # 直接传 list of list
        SELECTION_RATIO,
        TWED_LAMBDA,
        TWED_NU,
        TWED_P,
        DOWNSAMPLE_LEN,
        SEED
    )
    
    selected_indices_by_label[label] = selected_idx
    print(f"   筛选后样本数: {len(selected_idx)}")
    print(f"   选中索引示例: {selected_idx[:5]}")

# ==================== 4. 保存选中的数据 ====================
print("\n💾 保存筛选后的数据...")

for label, indices in selected_indices_by_label.items():
    print(f"\n📁 保存标签: {label}")
    
    # 获取选中的序列
    selected_sequences = [groups[label][i] for i in indices]
    
    # 拼接成一条长序列
    long_sequence = np.concatenate(selected_sequences, axis=0)
    total_len = len(long_sequence)
    
    print(f"   拼接后总长度: {total_len}")
    
    # 创建标签目录
    label_dir = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(label_dir, exist_ok=True)
    
    # 分块保存
    chunk_idx = 0
    start = 0
    
    while start < total_len:
        end = min(start + MAX_CHUNK_LEN, total_len)
        chunk = long_sequence[start:end]
        
        # 重塑为 (length, 1) 格式
        chunk = chunk.reshape(-1, 1)
        
        save_path = os.path.join(label_dir, f"{chunk_idx}.npy")
        np.save(save_path, chunk)
        
        print(f"   保存 {chunk_idx}.npy: shape {chunk.shape}")
        
        chunk_idx += 1
        start = end
    
    print(f"   ✅ 完成，共 {chunk_idx} 个文件")

# ==================== 5. 生成统计报告 ====================
print("\n" + "="*50)
print("📈 筛选统计报告")
print("="*50)

total_original = 0
total_selected = 0

for label, sequences in groups.items():
    selected_count = len(selected_indices_by_label.get(label, []))
    original_count = len(sequences)
    total_original += original_count
    total_selected += selected_count
    
    ratio = selected_count / original_count * 100 if original_count > 0 else 0
    print(f"{label:15s}: {original_count:6d} -> {selected_count:6d} ({ratio:.1f}%)")

print("-"*50)
print(f"{'总计':15s}: {total_original:6d} -> {total_selected:6d} ({total_selected/total_original*100:.1f}%)")
print("="*50)
print(f"\n✅ 所有处理完成！输出目录: {OUTPUT_ROOT}")

# ==================== 6. 验证输出格式 ====================
print("\n🔍 验证输出格式...")
for label in os.listdir(OUTPUT_ROOT):
    label_path = os.path.join(OUTPUT_ROOT, label)
    if os.path.isdir(label_path):
        files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
        if files:
            sample = np.load(os.path.join(label_path, files[0]))
            print(f"{label}: {len(files)} 个文件, 第一个文件 shape {sample.shape}")