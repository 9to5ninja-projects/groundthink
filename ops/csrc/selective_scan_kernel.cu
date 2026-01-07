#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for Selective Scan Forward Pass
// Shapes: [B, H, T, D]
// State: [B, H, D, D]

template <typename scalar_t>
__global__ void selective_scan_fwd_kernel(
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ r,
    scalar_t* __restrict__ state,     // [B, H, D, D]
    scalar_t* __restrict__ out,
    int B, int H, int T, int D,
    int stride_b, int stride_h, int stride_t, int stride_d,
    int s_stride_b, int s_stride_h, int s_stride_d1, int s_stride_d2
) {
    // Parallelize over Batch and Head
    int b = blockIdx.x;
    int h = blockIdx.y;
    
    // Each thread handles a subset of the state dimension if needed, 
    // but D is small (64). Let's use 1 thread per (b,h) for simplicity first (naive) 
    // or parallelize D.
    // For D=64: A single warp (32 threads) or block can handle the D loop.
    
    // CURRENT APPROACH: One Thread Block per (B, H) sequence.
    // We load state into shared memory for speed.
    
    // Dimensions
    int tid = threadIdx.x;
    
    // Check bounds
    if (b >= B || h >= H) return;

    // Offsets for global memory
    long long offset_seq = (long long)b * stride_b + (long long)h * stride_h;
    long long offset_state = (long long)b * s_stride_b + (long long)h * s_stride_h;
    
    // Pointers to the start of this sequence
    const scalar_t* k_ptr = k + offset_seq;
    const scalar_t* v_ptr = v + offset_seq;
    const scalar_t* w_ptr = w + offset_seq;
    const scalar_t* r_ptr = r + offset_seq;
    scalar_t* out_ptr = out + offset_seq;
    
    // State pointer in global memory (read/write at start/end)
    scalar_t* s_ptr = state + offset_state;

    // Shared Memory for State S [D, D]
    // D=64 -> 4096 floats -> 16KB. Fits in Shared Mem.
    extern __shared__ float s_shared[]; 
    
    // 1. Load Initial State into Shared Memory
    // Parallel load by threads. BlockDim.x should cover D*D or we loop.
    int num_elements = D * D;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        s_shared[i] = static_cast<float>(s_ptr[i]);
    }
    __syncthreads();
    
    // 2. Loop over Time T
    for (int t = 0; t < T; ++t) {
        // Offset for this timestep
        long long t_offset = t * stride_t;
        
        // A. Load inputs k, v, w, r for this timestep
        // Each vector is size D.
        // We can load them into register or shared memory.
        
        // Wait, D=64. The matrix math is D*D.
        // Threads need to cooperate.
        
        // Strategy: Each thread computes part of the matrix update?
        // Let's assume block dim is (D, 1) or (32, 2) etc.
        // Max threads per block 1024. D*D = 4096. 
        // We can't map 1 thread to 1 matrix element easily if limited.
        // But D is small.
        
        // Let's perform the S update: S = (1-w)S + k * v^T
        // Element (i, j): S[i,j] = (1 - w[i]) * S[i,j] + k[i] * v[j]
        
        // Let's assign threads linearly over D*D elements.
        for (int idx = tid; idx < num_elements; ++idx) {
            int row = idx / D;
            int col = idx % D;
            
            float k_val = static_cast<float>(k_ptr[t_offset + row * stride_d]);
            float v_val = static_cast<float>(v_ptr[t_offset + col * stride_d]);
            float w_val = static_cast<float>(w_ptr[t_offset + row * stride_d]);
            
            // Decayed update
            s_shared[idx] = (1.0f - w_val) * s_shared[idx] + (k_val * v_val);
        }
        __syncthreads();
        
        // B. Compute Output: y = r @ S (i.e., y[j] = sum_i(r[i] * S[i,j]) ? Check math)
        // Python code: out = r_c[:, t] @ state -> [1, D] @ [D, D] -> [1,D]
        // y[col] = sum_{row} (r[row] * S[row, col])
        
        // Thread assignment: Each thread computes one output element y[col]?
        // Or stride if we have > D threads?
        // D=64, we likely have >=64 threads.
        
        if (tid < D) {
            int col = tid; 
            float sum = 0.0f;
            
            // NOTE: This inner loop is serial per thread, but parallel across columns (threads).
            // Good for D=64.
            for (int row = 0; row < D; ++row) {
                float r_val = static_cast<float>(r_ptr[t_offset + row * stride_d]);
                sum += r_val * s_shared[row * D + col];
            }
            
            // Write output
            out_ptr[t_offset + col * stride_d] = static_cast<scalar_t>(sum);
        }
        
        // We need all calculations to finish before next timestep modifies S
        __syncthreads();
    }
    
    // 3. Write Final State back to Global Memory
    for (int i = tid; i < num_elements; i += blockDim.x) {
        s_ptr[i] = static_cast<scalar_t>(s_shared[i]);
    }
}

torch::Tensor selective_scan_fwd_cuda(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor w,
    torch::Tensor r,
    torch::Tensor state,
    torch::Tensor out
) {
    const int B = k.size(0);
    const int H = k.size(1); // Assuming input is [B, H, T, D] or reshaped to it
    const int T = k.size(2);
    const int D = k.size(3);

    // Threads per block
    // We need to cover D (64) for output, and help with D*D (4096) for state.
    // 256 threads is a reasonable balance.
    const int threads = 256;
    
    // Grid dimensions: One block per sequence (Batch * Heads)
    const dim3 blocks(B, H);
    
    // Shared memory size: D*D floats
    const int shared_mem_size = D * D * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(k.scalar_type(), "selective_scan_fwd_cuda", ([&] {
        selective_scan_fwd_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            r.data_ptr<scalar_t>(),
            state.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, H, T, D,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            state.stride(0), state.stride(1), state.stride(2), state.stride(3)
        );
    }));

    return out;
}
