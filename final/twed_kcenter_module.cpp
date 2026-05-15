// twed_kcenter_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <Python.h>

namespace py = pybind11;

inline void check_interrupt() {
    if (PyErr_CheckSignals() != 0) {
        throw py::error_already_set();
    }
}

#ifdef WITH_CUDA
bool twed_cuda_create_context(
    const std::vector<float>& flat_seqs,
    const std::vector<float>& flat_tss,
    const std::vector<int>& offsets,
    const std::vector<int>& lengths,
    int max_len,
    int pair_capacity,
    void** out_ctx,
    std::string& error_msg
);

bool twed_cuda_run_pairs(
    void* ctx,
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    float lambda_, float nu, int p,
    std::vector<float>& out_distances,
    std::string& error_msg
);

void twed_cuda_destroy_context(void* ctx);
#endif

float twed_distance(
    const std::vector<float>& seq1, const std::vector<float>& ts1,
    const std::vector<float>& seq2, const std::vector<float>& ts2,
    float lambda_, float nu, int p
) {
    int len1 = static_cast<int>(seq1.size());
    int len2 = static_cast<int>(seq2.size());
    auto abs_pow = [p](double x) -> double {
        double ax = std::abs(x);
        if (p == 1) return ax;
        if (p == 2) return ax * ax;
        return std::pow(ax, p);
    };

    std::vector<double> point_cost1(len1), point_cost2(len2);
    std::vector<double> time_step1(len1, 0.0), time_step2(len2, 0.0);
    for (int i = 0; i < len1; i++) {
        point_cost1[i] = abs_pow(seq1[i]);
        if (i > 0) time_step1[i] = static_cast<double>(nu) * (ts1[i] - ts1[i - 1]);
    }
    for (int j = 0; j < len2; j++) {
        point_cost2[j] = abs_pow(seq2[j]);
        if (j > 0) time_step2[j] = static_cast<double>(nu) * (ts2[j] - ts2[j - 1]);
    }

    std::vector<double> prev(len2 + 1, 1e20), curr(len2 + 1, 1e20);
    prev[0] = 0.0;
    for (int j = 1; j <= len2; j++) {
        prev[j] = prev[j - 1] + point_cost2[j - 1] + time_step2[j - 1];
    }

    for (int i = 1; i <= len1; i++) {
        curr[0] = prev[0] + point_cost1[i - 1] + time_step1[i - 1];
        for (int j = 1; j <= len2; j++) {
            double match = prev[j - 1]
                + abs_pow(static_cast<double>(seq1[i - 1] - seq2[j - 1]))
                + static_cast<double>(nu) * std::abs(static_cast<double>(ts1[i - 1] - ts2[j - 1]))
                + static_cast<double>(lambda_);

            double del = prev[j] + point_cost1[i - 1] + time_step1[i - 1];
            double ins = curr[j - 1] + point_cost2[j - 1] + time_step2[j - 1];

            curr[j] = std::min({match, del, ins});
        }
        std::swap(prev, curr);
    }

    return static_cast<float>(std::pow(prev[len2], 1.0 / p));
}

std::pair<std::vector<float>, std::vector<float>> downsample(
    const std::vector<float>& seq,
    const std::vector<float>& ts,
    int target_len
) {
    if (static_cast<int>(seq.size()) <= target_len) return {seq, ts};

    std::vector<float> down_seq;
    std::vector<float> down_ts;
    down_seq.reserve(target_len);
    down_ts.reserve(target_len);

    float step = static_cast<float>(seq.size() - 1) / static_cast<float>(target_len - 1);
    for (int i = 0; i < target_len; i++) {
        int idx = static_cast<int>(i * step);
        down_seq.push_back(seq[idx]);
        down_ts.push_back(ts[idx]);
    }
    return {down_seq, down_ts};
}

py::array_t<float> compute_twed_matrix(
    const std::vector<std::vector<float>>& sequences,
    const std::vector<std::vector<float>>& timestamps,
    float lambda_, float nu, int p,
    int max_len, int batch_size = 100
) {
    int n = static_cast<int>(sequences.size());

    std::vector<std::vector<float>> proc_seqs(n), proc_tss(n);
    for (int i = 0; i < n; i++) {
        auto ds_dt = downsample(sequences[i], timestamps[i], max_len);
        proc_seqs[i] = std::move(ds_dt.first);
        proc_tss[i] = std::move(ds_dt.second);
    }

    py::array_t<float> result({n, n});
    auto buf = result.request();
    float* dist_ptr = static_cast<float*>(buf.ptr);
    std::fill(dist_ptr, dist_ptr + static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);

    for (int i = 0; i < n; i += batch_size) {
        check_interrupt();
        int i_end = std::min(i + batch_size, n);
        for (int ii = i; ii < i_end; ii++) {
            check_interrupt();
            for (int j = ii + 1; j < n; j++) {
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

#ifdef WITH_CUDA
py::array_t<float> compute_twed_matrix_cuda(
    const std::vector<std::vector<float>>& sequences,
    const std::vector<std::vector<float>>& timestamps,
    float lambda_, float nu, int p,
    int max_len, int pair_batch_size = 4096,
    int progress_every_batches = 0
) {
    int n = static_cast<int>(sequences.size());
    // if (n == 0) return py::array_t<float>({0, 0});
    if (n == 0) return py::array_t<float>(std::vector<ptrdiff_t>{0, 0});
    if (pair_batch_size <= 0) throw std::runtime_error("pair_batch_size must be > 0");

    std::vector<std::vector<float>> proc_seqs(n), proc_tss(n);
    for (int i = 0; i < n; i++) {
        auto ds_dt = downsample(sequences[i], timestamps[i], max_len);
        proc_seqs[i] = std::move(ds_dt.first);
        proc_tss[i] = std::move(ds_dt.second);
    }

    std::vector<int> offsets(n + 1, 0);
    std::vector<int> lengths(n, 0);
    for (int i = 0; i < n; i++) {
        lengths[i] = static_cast<int>(proc_seqs[i].size());
        offsets[i + 1] = offsets[i] + lengths[i];
    }

    std::vector<float> flat_seqs(offsets[n]);
    std::vector<float> flat_tss(offsets[n]);
    for (int i = 0; i < n; i++) {
        int off = offsets[i];
        std::copy(proc_seqs[i].begin(), proc_seqs[i].end(), flat_seqs.begin() + off);
        std::copy(proc_tss[i].begin(), proc_tss[i].end(), flat_tss.begin() + off);
    }

    py::array_t<float> result({n, n});
    auto buf = result.request();
    float* dist_ptr = static_cast<float*>(buf.ptr);
    std::fill(dist_ptr, dist_ptr + static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);

    void* cuda_ctx = nullptr;
    {
        std::string error_msg;
        bool ok = twed_cuda_create_context(
            flat_seqs, flat_tss, offsets, lengths,
            max_len, pair_batch_size, &cuda_ctx, error_msg
        );
        if (!ok) {
            throw std::runtime_error("CUDA init failed: " + error_msg);
        }
    }
    struct CudaCtxGuard {
        void* ctx;
        ~CudaCtxGuard() {
            if (ctx) twed_cuda_destroy_context(ctx);
        }
    } ctx_guard{cuda_ctx};

    std::vector<int> pair_i;
    std::vector<int> pair_j;
    pair_i.reserve(pair_batch_size);
    pair_j.reserve(pair_batch_size);
    const long long total_pairs = static_cast<long long>(n) * static_cast<long long>(n - 1) / 2;
    long long done_pairs = 0;
    int batch_count = 0;
    auto t_start = std::chrono::steady_clock::now();

    auto flush_batch = [&]() {
        check_interrupt();
        if (pair_i.empty()) return;
        const int current_batch_size = static_cast<int>(pair_i.size());
        std::vector<float> out_distances;
        std::string error_msg;
        bool ok = twed_cuda_run_pairs(
            cuda_ctx,
            pair_i, pair_j,
            lambda_, nu, p,
            out_distances, error_msg
        );
        if (!ok) {
            throw std::runtime_error("CUDA TWED failed: " + error_msg);
        }
        for (size_t k = 0; k < pair_i.size(); k++) {
            int a = pair_i[k];
            int b = pair_j[k];
            float d = out_distances[k];
            dist_ptr[a * n + b] = d;
            dist_ptr[b * n + a] = d;
        }
        done_pairs += current_batch_size;
        batch_count++;
        if (progress_every_batches > 0 &&
            (batch_count % progress_every_batches == 0 || done_pairs == total_pairs)) {
            auto t_now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            const double pct = (total_pairs > 0) ? (100.0 * static_cast<double>(done_pairs) / static_cast<double>(total_pairs)) : 100.0;
            const double rate = (elapsed > 0.0) ? (static_cast<double>(done_pairs) / elapsed) : 0.0;
            const double eta = (rate > 0.0) ? ((static_cast<double>(total_pairs - done_pairs)) / rate) : -1.0;
            std::cout << "   [CUDA] pair progress: "
                      << done_pairs << "/" << total_pairs
                      << " (" << std::fixed << std::setprecision(2) << pct << "%)"
                      << ", elapsed " << std::setprecision(1) << elapsed << "s";
            if (eta >= 0.0) {
                std::cout << ", ETA " << eta << "s";
            }
            std::cout << std::endl;
        }
        pair_i.clear();
        pair_j.clear();
    };

    for (int i = 0; i < n; i++) {
        check_interrupt();
        for (int j = i + 1; j < n; j++) {
            pair_i.push_back(i);
            pair_j.push_back(j);
            if (static_cast<int>(pair_i.size()) >= pair_batch_size) {
                flush_batch();
            }
        }
    }
    flush_batch();
    return result;
}
#endif

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
    int n = static_cast<int>(sequences.size());
    int budget = std::max(1, static_cast<int>(n * ratio));

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

    while (static_cast<int>(selected_indices.size()) < budget) {
        check_interrupt();
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

#ifdef WITH_CUDA
std::vector<int> kcenter_greedy_twed_cuda(
    const std::vector<std::vector<float>>& sequences,
    const std::vector<std::vector<float>>& timestamps,
    float ratio,
    float lambda_ = 0.1,
    float nu = 0.001,
    int p = 2,
    int max_len = 500,
    int pair_batch_size = 4096,
    int progress_every_batches = 0,
    unsigned int seed = 42
) {
    int n = static_cast<int>(sequences.size());
    int budget = std::max(1, static_cast<int>(n * ratio));

    py::array_t<float> dist_matrix = compute_twed_matrix_cuda(
        sequences, timestamps, lambda_, nu, p, max_len, pair_batch_size, progress_every_batches
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

    while (static_cast<int>(selected_indices.size()) < budget) {
        check_interrupt();
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
#endif

std::vector<int> kcenter_greedy_simple(
    const std::vector<std::vector<float>>& sequences,
    float ratio,
    float lambda_ = 0.1,
    float nu = 0.001,
    int p = 2,
    int max_len = 500,
    unsigned int seed = 42
) {
    int n = static_cast<int>(sequences.size());
    std::vector<std::vector<float>> timestamps(n);
    for (int i = 0; i < n; i++) {
        int len = static_cast<int>(sequences[i].size());
        timestamps[i].resize(len);
        for (int j = 0; j < len; j++) timestamps[i][j] = static_cast<float>(j);
    }
    return kcenter_greedy_twed(sequences, timestamps, ratio, lambda_, nu, p, max_len, seed);
}

#ifdef WITH_CUDA
std::vector<int> kcenter_greedy_simple_cuda(
    const std::vector<std::vector<float>>& sequences,
    float ratio,
    float lambda_ = 0.1,
    float nu = 0.001,
    int p = 2,
    int max_len = 500,
    int pair_batch_size = 4096,
    int progress_every_batches = 0,
    unsigned int seed = 42
) {
    int n = static_cast<int>(sequences.size());
    std::vector<std::vector<float>> timestamps(n);
    for (int i = 0; i < n; i++) {
        int len = static_cast<int>(sequences[i].size());
        timestamps[i].resize(len);
        for (int j = 0; j < len; j++) timestamps[i][j] = static_cast<float>(j);
    }
    return kcenter_greedy_twed_cuda(
        sequences, timestamps, ratio,
        lambda_, nu, p, max_len, pair_batch_size, progress_every_batches, seed
    );
}
#endif

PYBIND11_MODULE(twed_kcenter, m) {
    m.def("kcenter_greedy_simple", &kcenter_greedy_simple,
          "K-Center Greedy with TWED (auto timestamps)",
          py::arg("sequences"), py::arg("ratio"),
          py::arg("lambda_") = 0.1, py::arg("nu") = 0.001,
          py::arg("p") = 2, py::arg("max_len") = 500, py::arg("seed") = 42);
    m.def("twed_distance", &twed_distance, "Compute TWED distance");
#ifdef WITH_CUDA
    m.def("kcenter_greedy_simple_cuda", &kcenter_greedy_simple_cuda,
          "K-Center Greedy with TWED using CUDA pair-level parallelism",
          py::arg("sequences"), py::arg("ratio"),
          py::arg("lambda_") = 0.1, py::arg("nu") = 0.001,
          py::arg("p") = 2, py::arg("max_len") = 500,
          py::arg("pair_batch_size") = 4096,
          py::arg("progress_every_batches") = 0,
          py::arg("seed") = 42);
#endif
}
