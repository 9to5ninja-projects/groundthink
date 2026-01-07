#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA functions
torch::Tensor selective_scan_fwd_cuda(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor w,
    torch::Tensor r,
    torch::Tensor state,
    torch::Tensor out
);

// Python binding
torch::Tensor selective_scan_forward(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor w,
    torch::Tensor r,
    torch::Tensor state) {
    
    // Check inputs
    TORCH_CHECK(k.is_cuda, "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda, "v must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda, "w must be a CUDA tensor");
    TORCH_CHECK(r.is_cuda, "r must be a CUDA tensor");
    TORCH_CHECK(state.is_cuda, "state must be a CUDA tensor");
    
    // Create output
    auto out = torch::empty_like(k);
    
    return selective_scan_fwd_cuda(k, v, w, r, state, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selective_scan_forward, "Selective Scan Forward (CUDA)");
}
