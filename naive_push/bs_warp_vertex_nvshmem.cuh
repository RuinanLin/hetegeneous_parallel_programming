// vertex parallel: each warp takes one vertex
__global__ void warp_vertex_nvshmem(AccType *count, GraphGPU g) {
    // __shared__ typename BlockReduce::TempStorage temp_storage;
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    // int warp_id = thread_id / WARP_SIZE;                    // global warp index
    // int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps
    // int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    // int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA

    // AccType count = 0;

}