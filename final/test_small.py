import numpy as np
from datasets import load_dataset
import twed_kcenter

# 加载小样本测试
print("加载测试数据...")
ds = load_dataset("thuml/UTSD", "UTSD-1G", split="train")

# 只取前 100 个样本测试
test_samples = ds[:100]

# 准备数据
test_sequences = []
test_labels = []

for i in range(100):
    label = test_samples['item_id'][i].split('_')[0]
    target = np.array(test_samples['target'][i], dtype=np.float32)
    test_sequences.append(target)
    test_labels.append(label)

print(f"测试样本数: {len(test_sequences)}")

# 测试 C++ 模块
print("\n测试 C++ TWED 距离...")
seq1 = test_sequences[0].tolist()
seq2 = test_sequences[1].tolist()
ts1 = list(range(len(seq1)))
ts2 = list(range(len(seq2)))

dist = twed_kcenter.twed_distance(
    seq1, ts1, len(seq1),
    seq2, ts2, len(seq2),
    0.1, 0.001, 2
)
print(f"TWED 距离: {dist}")

# 测试 K-Center
print("\n测试 K-Center Greedy...")
seqs_list = [seq.tolist() for seq in test_sequences[:50]]
selected = twed_kcenter.kcenter_greedy_simple(
    np.array(seqs_list, dtype=np.float32),
    ratio=0.2,  # 选 20%
    lambda_=0.1,
    nu=0.001,
    p=2,
    max_len=300,
    seed=42
)
print(f"选中索引: {selected}")
print(f"选中数量: {len(selected)}")

print("\n✅ 测试通过！")