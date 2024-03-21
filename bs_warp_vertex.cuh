// vertex paralle: each warp takes one vertex
__global__ void warp_vertex(vidType begin, vidType end, GraphGPU g, AccType *total) {
    
    // allocate a space for map-reduce
    __shared__ AccType partial_sum[BLOCK_SIZE];

    // calculate the global index of the warp
    int global_warp_idx = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    int total_warp_num = gridDim.x * blockDim.x / WARP_SIZE;
    
    // implement calculation of the vertecies
    AccType cnt = 0;
    for (vidType u = global_warp_idx; u < end; u += total_warp_num)
    {
        vidType *u_vector = g.N(u); // u_vector[] stores the vertecies that are adjacent to u and less than u
        vidType u_vector_size = g.getOutDegree(u);  // u_vector_size is the size of u_vector[]
        for (int v_idx = 0; v_idx < u_vector_size; v_idx++) // v_idx is the index in u_vector[]
        {
            vidType v = u_vector[v_idx];
            cnt += intersect_num(u_vector, u_vector_size, g.N(v), (vidType)g.getOutDegree(v));
        }
    }
    partial_sum[threadIdx.x] = cnt;

    // reduce
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (threadIdx.x < stride)
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
    }
    if (threadIdx.x == 0)
        atomicAdd(total, partial_sum[0]);

}

