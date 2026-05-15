import torch
import numpy as np
import twed_cuda

def compute_twed_matrix_gpu(sequences, timestamps, max_len=200, device='cuda:0'):
    """
    sequences: list of np.ndarray (variable length)
    timestamps: list of np.ndarray (same length as sequences)
    """
    n = len(sequences)
    # 下采样到固定长度
    down_seqs = []
    down_tss = []
    lens = []
    for seq, ts in zip(sequences, timestamps):
        L = len(seq)
        if L > max_len:
            idx = np.linspace(0, L-1, max_len).astype(int)
            seq = seq[idx]
            ts = ts[idx]
            L = max_len
        else:
            # 填充到 max_len（用最后一个值填充）
            pad_len = max_len - L
            seq = np.pad(seq, (0, pad_len), constant_values=seq[-1])
            ts = np.pad(ts, (0, pad_len), constant_values=ts[-1])
        down_seqs.append(seq)
        down_tss.append(ts)
        lens.append(L)
    
    # 转换为 tensor
    seqs_tensor = torch.from_numpy(np.stack(down_seqs, axis=0)).float().to(device)
    tss_tensor = torch.from_numpy(np.stack(down_tss, axis=0)).float().to(device)
    lens_tensor = torch.tensor(lens, dtype=torch.int).to(device)
    
    # 调用 CUDA 扩展
    dist_matrix = twed_cuda.twed_matrix(seqs_tensor, tss_tensor, lens_tensor, lambda_=0.1, nu=0.001, p=2)
    return dist_matrix.cpu().numpy()

# 使用示例
sequences = [np.random.rand(896) for _ in range(1000)]
timestamps = [np.arange(len(s)) for s in sequences]
dist = compute_twed_matrix_gpu(sequences, timestamps, max_len=200)
print(dist.shape)  # (1000, 1000)