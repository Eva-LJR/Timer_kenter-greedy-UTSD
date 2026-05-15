// twed_kcenter_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

namespace py = pybind11;

// TWED 距离（与之前相同）
float twed_distance(
    const std::vector<float>& seq1, const std::vector<float>& ts1,
    const std::vector<float>& seq2, const std::vector<float>& ts2,
    float lambda_, float nu, int p
) {
    int len1 = seq1.size(), len2 = seq2.size();
    std::vector<std::vector<double>> dp(len1 + 1, std::vector<double>(len2 + 1, 1e20));
    dp[0][0] = 0.0;

    for (int i = 1; i <= len1; i++) {
        double cost = std::pow(std::abs(seq1[i-1]), p);
        double time_cost = (i > 1) ? nu * (ts1[i-1] - ts1[i-2]) : 0;
        dp[i][0] = dp[i-1][0] + cost + time_cost;
    }

    for (int j = 1; j <= len2; j++) {
        double cost = std::pow(std::abs(seq2[j-1]), p);
        double time_cost = (j > 1) ? nu * (ts2[j-1] - ts2[j-2]) : 0;
        dp[0][j] = dp[0][j-1] + cost + time_cost;
    }

    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            double match = dp[i-1][j-1]
                + std::pow(std::abs(seq1[i-1] - seq2[j-1]), p)
                + nu * std::abs(ts1[i-1] - ts2[j-1])
                + lambda_;

            double del = dp[i-1][j]
                + std::pow(std::abs(seq1[i-1]), p)
                + ((i > 1) ? nu * (ts1[i-1] - ts1[i-2]) : 0);

            double ins = dp[i][j-1]
                + std::pow(std::abs(seq2[j-1]), p)
                + ((j > 1) ? nu * (ts2[j-1] - ts2[j-2]) : 0);

            dp[i][j] = std::min({match, del, ins});
        }
    }

    return std::pow(dp[len1][len2], 1.0 / p);
}

// 下采样（返回 pair）
std::pair<std::vector<float>, std::vector<float>> downsample(
    const std::vector<float>& seq,
    const std::vector<float>& ts,
    int target_len
) {
    if ((int)seq.size() <= target_len) return {seq, ts};

    std::vector<float> down_seq, down_ts;
    float step = (float)(seq.size() - 1) / (target_len - 1);
    for (int i = 0; i < target_len; i++) {
        int idx = (int)(i * step);
        down_seq.push_back(seq[idx]);
        down_ts.push_back(ts[idx]);
    }
    return {down_seq, down_ts};
}

// 计算距离矩阵（接受 vector of vector）
py::array_t<float> compute_twed_matrix(
    const std::vector<std::vector<float>>& sequences,
    const std::vector<std::vector<float>>& timestamps,
    float lambda_, float nu, int p,
    int max_len, int batch_size = 100
) {
    int n = sequences.size();

    // 下采样
    std::vector<std::vector<float>> proc_seqs(n), proc_tss(n);
    for (int i = 0; i < n; i++) {
        auto [ds, dt] = downsample(sequences[i], timestamps[i], max_len);
        proc_seqs[i] = ds;
        proc_tss[i] = dt;
    }

    // 分配矩阵
    py::array_t<float> result({n, n});
    auto buf = result.request();
    float* dist_ptr = static_cast<float*>(buf.ptr);

    // 对角线为0
    for (int i = 0; i < n; i++) dist_ptr[i * n + i] = 0.0f;

    // 分批计算
    for (int i = 0; i < n; i += batch_size) {
        int i_end = std::min(i + batch_size, n);
        for (int j = i + 1; j < n; j++) {
            for (int ii = i; ii < i_end; ii++) {
                float d = twed_distance(
                    proc_seqs[ii], proc_tss[ii],
                    proc_seqs[j], proc_tss[j],
                    lambda_, nu, p
                );
                dist_ptr[ii * n + j] = d;
                dist_ptr[j * n + ii] = d;
            }
        }
    }
    return result;
}

// K-Center Greedy 主函数（接受 vector of vector）
std::vector<int> kcenter_greedy_twed(
    const std::vector<std::vector<float>>& sequences,
    const std::vector<std::vector<float>>& timestamps,
    float ratio,
    float lambda_ = 0.1,
    float nu = 0.001,
    int p = 2,
    int max_len = 500,
    unsigned int seed = 42
) {
    int n = sequences.size();
    int budget = std::max(1, (int)(n * ratio));

    py::array_t<float> dist_matrix = compute_twed_matrix(
        sequences, timestamps, lambda_, nu, p, max_len
    );
    auto buf = dist_matrix.request();
    float* dist_ptr = static_cast<float*>(buf.ptr);

    std::vector<bool> selected(n, false);
    std::vector<int> selected_indices;
    std::vector<float> min_dist(n, std::numeric_limits<float>::max());

    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n - 1);
    int first = dis(gen);
    selected_indices.push_back(first);
    selected[first] = true;

    for (int i = 0; i < n; i++) {
        if (!selected[i]) min_dist[i] = dist_ptr[i * n + first];
    }

    while ((int)selected_indices.size() < budget) {
        int best_idx = -1;
        float best_dist = -1.0f;
        for (int i = 0; i < n; i++) {
            if (!selected[i] && min_dist[i] > best_dist) {
                best_dist = min_dist[i];
                best_idx = i;
            }
        }
        if (best_idx == -1) break;

        selected_indices.push_back(best_idx);
        selected[best_idx] = true;

        for (int i = 0; i < n; i++) {
            if (!selected[i]) {
                float d = dist_ptr[i * n + best_idx];
                if (d < min_dist[i]) min_dist[i] = d;
            }
        }
    }
    return selected_indices;
}

// 简化版接口（自动生成时间戳）
std::vector<int> kcenter_greedy_simple(
    const std::vector<std::vector<float>>& sequences,  // 关键：直接用 vector
    float ratio,
    float lambda_ = 0.1,
    float nu = 0.001,
    int p = 2,
    int max_len = 500,
    unsigned int seed = 42
) {
    int n = sequences.size();
    std::vector<std::vector<float>> timestamps(n);
    for (int i = 0; i < n; i++) {
        int len = sequences[i].size();
        timestamps[i].resize(len);
        for (int j = 0; j < len; j++) timestamps[i][j] = (float)j;
    }
    return kcenter_greedy_twed(sequences, timestamps, ratio, lambda_, nu, p, max_len, seed);
}

PYBIND11_MODULE(twed_kcenter, m) {
    m.def("kcenter_greedy_simple", &kcenter_greedy_simple,
          "K-Center Greedy with TWED (auto timestamps)",
          py::arg("sequences"), py::arg("ratio"),
          py::arg("lambda_") = 0.1, py::arg("nu") = 0.001,
          py::arg("p") = 2, py::arg("max_len") = 500, py::arg("seed") = 42);
    m.def("twed_distance", &twed_distance, "Compute TWED distance");
}