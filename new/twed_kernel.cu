#include <cuda_runtime.h>
#include <math.h>

// 计算一对序列的 TWED（串行，在单个线程中执行）
__device__ float twed_pair_device(
    const float* seq1, const float* ts1, int len1,
    const float* seq2, const float* ts2, int len2,
    float lambda_, float nu, int p
) {
    // 动态分配 DP 矩阵（使用局部数组，最大长度限制为 256）
    // 注意：如果 len1,len2 > 256，会栈溢出。实际使用时需确保下采样后长度 <= 256
    const int MAXL = 256;
    float dp[MAXL+1][MAXL+1];
    
    dp[0][0] = 0.0f;
    for (int i = 1; i <= len1; ++i) {
        float cost = powf(fabs(seq1[i-1]), p);
        float time_cost = (i > 1) ? nu * (ts1[i-1] - ts1[i-2]) : 0.0f;
        dp[i][0] = dp[i-1][0] + cost + time_cost;
    }
    for (int j = 1; j <= len2; ++j) {
        float cost = powf(fabs(seq2[j-1]), p);
        float time_cost = (j > 1) ? nu * (ts2[j-1] - ts2[j-2]) : 0.0f;
        dp[0][j] = dp[0][j-1] + cost + time_cost;
    }
    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            float match = dp[i-1][j-1]
                + powf(fabs(seq1[i-1] - seq2[j-1]), p)
                + nu * fabs(ts1[i-1] - ts2[j-1])
                + lambda_;
            float del = dp[i-1][j]
                + powf(fabs(seq1[i-1]), p)
                + ((i > 1) ? nu * (ts1[i-1] - ts1[i-2]) : 0.0f);
            float ins = dp[i][j-1]
                + powf(fabs(seq2[j-1]), p)
                + ((j > 1) ? nu * (ts2[j-1] - ts2[j-2]) : 0.0f);
            dp[i][j] = fminf(fminf(match, del), ins);
        }
    }
    return powf(dp[len1][len2], 1.0f / p);
}

__global__ void twed_matrix_kernel(
    const float* __restrict__ seqs,   // [n, max_len]  flatten
    const float* __restrict__ tss,    // [n, max_len]
    const int* __restrict__ lens,     // [n] 实际长度（未填充）
    float* __restrict__ dist,         // [n, n]
    int n,
    int max_len,
    float lambda_,
    float nu,
    int p
) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= n || j >= n || i >= j) return;  // 只计算上三角
    
    const float* seq_i = seqs + i * max_len;
    const float* ts_i  = tss  + i * max_len;
    const float* seq_j = seqs + j * max_len;
    const float* ts_j  = tss  + j * max_len;
    int len_i = lens[i];
    int len_j = lens[j];
    
    float d = twed_pair_device(seq_i, ts_i, len_i, seq_j, ts_j, len_j, lambda_, nu, p);
    dist[i * n + j] = d;
    dist[j * n + i] = d;
}