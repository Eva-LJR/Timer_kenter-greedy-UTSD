import numpy as np
from tqdm import tqdm

# 直接使用 TWED 原始实现，不从 pyts 导入
def twed(sA, sB, tA, tB, nu=0.001, lambda_=0.1, p=2):
    m = len(sA)
    n = len(sB)
    dp = np.full((m+1, n+1), np.inf)
    dp[0,0] = 0.0

    for i in range(1, m+1):
        cost = np.abs(sA[i-1] - 0.0) ** p
        dp[i,0] = dp[i-1,0] + cost + nu * (tA[i-1] - tA[i-2]) if i>1 else cost

    for j in range(1, n+1):
        cost = np.abs(0.0 - sB[j-1]) ** p
        dp[0,j] = dp[0,j-1] + cost + nu * (tB[j-1] - tB[j-2]) if j>1 else cost

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = np.abs(sA[i-1] - sB[j-1]) ** p
            delta_t = np.abs(tA[i-1] - tB[j-1])
            case1 = dp[i-1,j-1] + cost + nu * delta_t + lambda_
            case2 = dp[i-1,j] + np.abs(sA[i-1] - 0.0)**p + nu*(tA[i-1] - (tA[i-2] if i>1 else tA[i-1]))
            case3 = dp[i,j-1] + np.abs(0.0 - sB[j-1])**p + nu*(tB[j-1] - (tB[j-2] if j>1 else tB[j-1]))
            dp[i,j] = min(case1, case2, case3)
    return dp[m,n] ** (1/p)

def compute_twed_matrix(sequences, lam=0.1, nu=0.001, p=2):
    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    timestamps = [np.arange(len(s)) for s in sequences]

    for i in tqdm(range(n), desc="TWED矩阵计算"):
        seq_i = sequences[i].reshape(-1)
        ts_i = timestamps[i]
        for j in range(i + 1, n):
            seq_j = sequences[j].reshape(-1)
            ts_j = timestamps[j]
            d = twed(seq_i, seq_j, ts_i, ts_j, nu=nu, lambda_=lam, p=p)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix

def kcenter_greedy_selection(sequences, ratio, lam=0.1, nu=0.001, p=2, seed=42):
    np.random.seed(seed)
    n = len(sequences)
    k = max(1, int(n * ratio))

    dist_matrix = compute_twed_matrix(sequences, lam, nu, p)
    selected = []
    unselected = list(range(n))

    first = np.random.choice(unselected)
    selected.append(first)
    unselected.remove(first)

    for _ in range(1, k):
        min_dists = dist_matrix[unselected][:, selected].min(axis=1)
        idx = min_dists.argmax()
        best = unselected[idx]
        selected.append(best)
        unselected.remove(best)

    return sorted(selected)