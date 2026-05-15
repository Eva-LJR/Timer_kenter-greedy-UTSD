#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void twed_matrix_cuda(
    torch::Tensor seqs,   // [n, max_len] float
    torch::Tensor tss,    // [n, max_len] float
    torch::Tensor lens,   // [n] int
    torch::Tensor dist,   // [n, n] float (output)
    float lambda_,
    float nu,
    int p
);

torch::Tensor twed_matrix(
    torch::Tensor seqs,
    torch::Tensor tss,
    torch::Tensor lens,
    float lambda_,
    float nu,
    int p
) {
    CHECK_CUDA(seqs);
    CHECK_CUDA(tss);
    CHECK_CUDA(lens);
    CHECK_CONTIGUOUS(seqs);
    CHECK_CONTIGUOUS(tss);
    CHECK_CONTIGUOUS(lens);
    
    int n = seqs.size(0);
    int max_len = seqs.size(1);
    auto dist = torch::empty({n, n}, seqs.options());
    
    twed_matrix_cuda(seqs, tss, lens, dist, lambda_, nu, p);
    return dist;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("twed_matrix", &twed_matrix, "Compute TWED distance matrix on GPU");
}