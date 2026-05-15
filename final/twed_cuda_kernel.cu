// twed_cuda_kernel.cu
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace {

struct TwedCudaContext {
    float* d_flat_seqs = nullptr;
    float* d_flat_tss = nullptr;
    int* d_offsets = nullptr;
    int* d_lengths = nullptr;

    int* d_pair_i = nullptr;
    int* d_pair_j = nullptr;
    double* d_work = nullptr;
    float* d_out = nullptr;

    int pair_capacity = 0;
    int max_len = 0;
};

inline void set_error(std::string& error_msg, const char* prefix, cudaError_t err) {
    error_msg = std::string(prefix) + cudaGetErrorString(err);
}

inline void free_ctx(TwedCudaContext* ctx) {
    if (!ctx) return;
    if (ctx->d_flat_seqs) cudaFree(ctx->d_flat_seqs);
    if (ctx->d_flat_tss) cudaFree(ctx->d_flat_tss);
    if (ctx->d_offsets) cudaFree(ctx->d_offsets);
    if (ctx->d_lengths) cudaFree(ctx->d_lengths);
    if (ctx->d_pair_i) cudaFree(ctx->d_pair_i);
    if (ctx->d_pair_j) cudaFree(ctx->d_pair_j);
    if (ctx->d_work) cudaFree(ctx->d_work);
    if (ctx->d_out) cudaFree(ctx->d_out);
    delete ctx;
}

__device__ inline double abs_pow(double x, int p) {
    double ax = fabs(x);
    if (p == 1) return ax;
    if (p == 2) return ax * ax;
    return pow(ax, static_cast<double>(p));
}

__global__ void twed_pairs_kernel(
    const float* flat_seqs,
    const float* flat_tss,
    const int* offsets,
    const int* lengths,
    const int* pair_i,
    const int* pair_j,
    int num_pairs,
    float lambda_, float nu, int p,
    int max_len,
    double* work,
    float* out_distances
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    int i = pair_i[tid];
    int j = pair_j[tid];

    int off1 = offsets[i];
    int off2 = offsets[j];
    int len1 = lengths[i];
    int len2 = lengths[j];

    const float* seq1 = flat_seqs + off1;
    const float* ts1 = flat_tss + off1;
    const float* seq2 = flat_seqs + off2;
    const float* ts2 = flat_tss + off2;

    int stride = max_len + 1;
    double* base = work + static_cast<size_t>(tid) * static_cast<size_t>(2 * stride);
    double* prev = base;
    double* curr = base + stride;

    prev[0] = 0.0;
    for (int jj = 1; jj <= len2; jj++) {
        double cost = abs_pow(static_cast<double>(seq2[jj - 1]), p);
        double time_cost = (jj > 1) ? static_cast<double>(nu) * (ts2[jj - 1] - ts2[jj - 2]) : 0.0;
        prev[jj] = prev[jj - 1] + cost + time_cost;
    }

    for (int ii = 1; ii <= len1; ii++) {
        double cost = abs_pow(static_cast<double>(seq1[ii - 1]), p);
        double time_cost = (ii > 1) ? static_cast<double>(nu) * (ts1[ii - 1] - ts1[ii - 2]) : 0.0;
        curr[0] = prev[0] + cost + time_cost;

        for (int jj = 1; jj <= len2; jj++) {
            double match = prev[jj - 1]
                + abs_pow(static_cast<double>(seq1[ii - 1] - seq2[jj - 1]), p)
                + static_cast<double>(nu) * fabs(static_cast<double>(ts1[ii - 1] - ts2[jj - 1]))
                + static_cast<double>(lambda_);

            double del = prev[jj]
                + abs_pow(static_cast<double>(seq1[ii - 1]), p)
                + ((ii > 1) ? static_cast<double>(nu) * (ts1[ii - 1] - ts1[ii - 2]) : 0.0);

            double ins = curr[jj - 1]
                + abs_pow(static_cast<double>(seq2[jj - 1]), p)
                + ((jj > 1) ? static_cast<double>(nu) * (ts2[jj - 1] - ts2[jj - 2]) : 0.0);

            curr[jj] = fmin(match, fmin(del, ins));
        }

        double* tmp = prev;
        prev = curr;
        curr = tmp;
    }

    out_distances[tid] = static_cast<float>(pow(prev[len2], 1.0 / static_cast<double>(p)));
}

} // namespace

bool twed_cuda_create_context(
    const std::vector<float>& flat_seqs,
    const std::vector<float>& flat_tss,
    const std::vector<int>& offsets,
    const std::vector<int>& lengths,
    int max_len,
    int pair_capacity,
    void** out_ctx,
    std::string& error_msg
) {
    error_msg.clear();
    if (!out_ctx) {
        error_msg = "out_ctx is null";
        return false;
    }
    if (pair_capacity <= 0) {
        error_msg = "pair_capacity must be > 0";
        return false;
    }
    if (max_len <= 0) {
        error_msg = "max_len must be > 0";
        return false;
    }

    *out_ctx = nullptr;
    TwedCudaContext* ctx = new TwedCudaContext();
    ctx->pair_capacity = pair_capacity;
    ctx->max_len = max_len;

    cudaError_t err;

    err = cudaMalloc(&ctx->d_flat_seqs, flat_seqs.size() * sizeof(float));
    if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_flat_seqs failed: ", err); goto fail; }
    err = cudaMalloc(&ctx->d_flat_tss, flat_tss.size() * sizeof(float));
    if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_flat_tss failed: ", err); goto fail; }
    err = cudaMalloc(&ctx->d_offsets, offsets.size() * sizeof(int));
    if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_offsets failed: ", err); goto fail; }
    err = cudaMalloc(&ctx->d_lengths, lengths.size() * sizeof(int));
    if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_lengths failed: ", err); goto fail; }

    err = cudaMalloc(&ctx->d_pair_i, static_cast<size_t>(pair_capacity) * sizeof(int));
    if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_pair_i failed: ", err); goto fail; }
    err = cudaMalloc(&ctx->d_pair_j, static_cast<size_t>(pair_capacity) * sizeof(int));
    if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_pair_j failed: ", err); goto fail; }
    err = cudaMalloc(&ctx->d_out, static_cast<size_t>(pair_capacity) * sizeof(float));
    if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_out failed: ", err); goto fail; }

    {
        size_t work_size = static_cast<size_t>(pair_capacity)
            * static_cast<size_t>(2 * (max_len + 1))
            * sizeof(double);
        err = cudaMalloc(&ctx->d_work, work_size);
        if (err != cudaSuccess) { set_error(error_msg, "cudaMalloc d_work failed: ", err); goto fail; }
    }

    err = cudaMemcpy(ctx->d_flat_seqs, flat_seqs.data(), flat_seqs.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error(error_msg, "cudaMemcpy flat_seqs failed: ", err); goto fail; }
    err = cudaMemcpy(ctx->d_flat_tss, flat_tss.data(), flat_tss.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error(error_msg, "cudaMemcpy flat_tss failed: ", err); goto fail; }
    err = cudaMemcpy(ctx->d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error(error_msg, "cudaMemcpy offsets failed: ", err); goto fail; }
    err = cudaMemcpy(ctx->d_lengths, lengths.data(), lengths.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error(error_msg, "cudaMemcpy lengths failed: ", err); goto fail; }

    *out_ctx = ctx;
    return true;

fail:
    free_ctx(ctx);
    return false;
}

bool twed_cuda_run_pairs(
    void* ctx_ptr,
    const std::vector<int>& pair_i,
    const std::vector<int>& pair_j,
    float lambda_, float nu, int p,
    std::vector<float>& out_distances,
    std::string& error_msg
) {
    error_msg.clear();
    TwedCudaContext* ctx = static_cast<TwedCudaContext*>(ctx_ptr);
    if (!ctx) {
        error_msg = "ctx is null";
        return false;
    }

    int num_pairs = static_cast<int>(pair_i.size());
    if (num_pairs != static_cast<int>(pair_j.size())) {
        error_msg = "pair_i and pair_j size mismatch";
        return false;
    }
    if (num_pairs > ctx->pair_capacity) {
        error_msg = "num_pairs exceeds context pair_capacity";
        return false;
    }
    if (num_pairs == 0) {
        out_distances.clear();
        return true;
    }

    out_distances.resize(num_pairs);

    cudaError_t err;
    err = cudaMemcpy(ctx->d_pair_i, pair_i.data(), static_cast<size_t>(num_pairs) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error(error_msg, "cudaMemcpy pair_i failed: ", err); return false; }
    err = cudaMemcpy(ctx->d_pair_j, pair_j.data(), static_cast<size_t>(num_pairs) * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { set_error(error_msg, "cudaMemcpy pair_j failed: ", err); return false; }

    {
        int threads = 128;
        int blocks = (num_pairs + threads - 1) / threads;
        twed_pairs_kernel<<<blocks, threads>>>(
            ctx->d_flat_seqs, ctx->d_flat_tss,
            ctx->d_offsets, ctx->d_lengths,
            ctx->d_pair_i, ctx->d_pair_j, num_pairs,
            lambda_, nu, p, ctx->max_len,
            ctx->d_work, ctx->d_out
        );
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) { set_error(error_msg, "kernel launch failed: ", err); return false; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { set_error(error_msg, "kernel execution failed: ", err); return false; }

    err = cudaMemcpy(out_distances.data(), ctx->d_out, static_cast<size_t>(num_pairs) * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { set_error(error_msg, "cudaMemcpy out_distances failed: ", err); return false; }

    return true;
}

void twed_cuda_destroy_context(void* ctx) {
    free_ctx(static_cast<TwedCudaContext*>(ctx));
}
