// vertex parallel: each warp takes one vertex

__global__ void warp_clear_valid(GraphGPU g) {
    // about the threads and blocks
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id = thread_id / WARP_SIZE;    // global warp index
    int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;           // total number of global warps
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA

    // the parameters of the message buffer
    // the 'size' unit is 4B, vidType
    vidType md = g.get_max_degree();
    int valid_offset = 0;
    int u_offset = valid_offset + 1;
    int degree_offset = u_offset + 1;
    int bitmask_offset = degree_offset + 1;
    int list_offset = bitmask_offset + ((md-1)/32+1);
    int single_message_size = list_offset + md;

    int bitmask_size = (md-1)/32+1;
    int message_block_size = single_message_size * num_warps;

    for (int message_block_id = 0; message_block_id < g.get_n_gpu()-1; message_block_id++) {
        int message_address_offset = message_block_id * message_block_size + single_message_size * warp_id;
        vidType *message_address = g.get_message_address(message_address_offset);
        if (thread_lane == 0)
            message_address[valid_offset] = 0;
    }
}


__global__ void warp_vertex_nvshmem_local(AccType *total, GraphGPU g, int u_local_id_start, int u_local_id_end) {
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // about the threads and blocks
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id = thread_id / WARP_SIZE;    // global warp index
    int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;           // total number of global warps
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA

    // the parameters of the message buffer
    // the 'size' unit is 4B, vidType
    vidType md = g.get_max_degree();
    int valid_offset = 0;
    int u_offset = valid_offset + 1;
    int degree_offset = u_offset + 1;
    int bitmask_offset = degree_offset + 1;
    int list_offset = bitmask_offset + ((md-1)/32+1);
    int single_message_size = list_offset + md;

    int bitmask_size = (md-1)/32+1;
    int message_block_size = single_message_size * num_warps;

    AccType count = 0;
    int u_partition_num = nvshmem_my_pe();

    int u_local_id = warp_id + u_local_id_start;
    if (u_local_id < g.get_n_real_vertices()) {
        vidType u = g.get_vertex_in_vertex_list(u_local_id);
        eidType u_list_begin_id = g.edge_begin(u);
        eidType u_list_end_id = g.edge_end(u);
        for (eidType v_id = u_list_begin_id; v_id < u_list_end_id; v_id++) {
            vidType v = g.getEdgeDst(v_id);
            int v_partition_num = g.get_vertex_partition_number(v);
            if (v_partition_num == u_partition_num) {
                count += intersect_num(g.N(u), g.get_degree(u), g.N(v), g.get_degree(v));
            } else {
                if (thread_lane == 0) {   // only one thread need to operate this
                    // get message block id
                    int message_block_id;
                    if (v_partition_num > u_partition_num)
                        message_block_id = u_partition_num;
                    else
                        message_block_id = u_partition_num - 1;

                    // get the message nvshmem address offset (unit: vidType)
                    int message_address_offset = message_block_id * message_block_size + single_message_size * warp_id;
                    vidType *message_address = g.get_message_address(message_address_offset);

                    ////////////////////// put the message ///////////////////////

                    // check whether already valid, if not set 1, and record information
                    if (nvshmem_uint32_g((uint32_t *)(message_address + valid_offset), v_partition_num) != 1) {
                        nvshmem_uint32_p((uint32_t *)(message_address + valid_offset), 1, v_partition_num);
                        nvshmem_uint32_p((uint32_t *)(message_address + u_offset), u, v_partition_num);
                        nvshmem_uint32_p((uint32_t *)(message_address + degree_offset), u_list_end_id - u_list_begin_id, v_partition_num);
                        for (int bitmask_32bit_id = 0; bitmask_32bit_id < bitmask_size; bitmask_32bit_id++) {
                            nvshmem_uint32_p((uint32_t *)(message_address + bitmask_offset + bitmask_32bit_id), 0, v_partition_num);
                        }
                        nvshmem_putmem(message_address + list_offset, g.N(u), g.get_degree(u) * sizeof(vidType), v_partition_num);
                    }

                    // refresh the bitmask
                    int v_id_in_u = v_id - u_list_begin_id;
                    uint32_t bitmask_32bit_value = nvshmem_uint32_g((uint32_t *)(message_address + bitmask_offset + v_id_in_u / 32), v_partition_num);
                    bitmask_32bit_value |= (1u << (v_id_in_u % 32));
                    nvshmem_uint32_p((uint32_t *)(message_address + bitmask_offset + v_id_in_u / 32), bitmask_32bit_value, v_partition_num);
                }
            }
        }
    }

    AccType block_num = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) atomicAdd(total, block_num);
}


__global__ void warp_vertex_nvshmem_global(AccType *total, GraphGPU g) {
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // about the threads and blocks
    int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int warp_id = thread_id / WARP_SIZE;    // global warp index
    int num_warps = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;           // total number of global warps
    int thread_lane = threadIdx.x & (WARP_SIZE-1);          // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;                // warp index within the CTA

    // the parameters of the message buffer
    // the 'size' unit is 4B, vidType
    vidType md = g.get_max_degree();
    int valid_offset = 0;
    int u_offset = valid_offset + 1;
    int degree_offset = u_offset + 1;
    int bitmask_offset = degree_offset + 1;
    int list_offset = bitmask_offset + ((md-1)/32+1);
    int single_message_size = list_offset + md;

    int bitmask_size = (md-1)/32+1;
    int message_block_size = single_message_size * num_warps;

    AccType count = 0;

    for (int message_block_id = 0; message_block_id < g.get_n_gpu()-1; message_block_id++) {
        int message_address_offset = message_block_id * message_block_size + single_message_size * warp_id;
        vidType *message_address = g.get_message_address(message_address_offset);
        if (message_address[valid_offset]) {
            // iterate through the bitmask
            for (int bitmask_32bit_id = 0; bitmask_32bit_id < bitmask_size; bitmask_32bit_id++) {
                uint32_t bitmask_32bit_value = message_address[bitmask_offset + bitmask_32bit_id];
                int bitmask_32bit_nonzero_id = 0;
                while (bitmask_32bit_value) {
                    while (!(bitmask_32bit_value & 1u)) {
                        bitmask_32bit_value >>= 1;
                        bitmask_32bit_nonzero_id++;
                    }
                    int v_id_in_u = bitmask_32bit_id * 32 + bitmask_32bit_nonzero_id;
                    vidType v = message_address[list_offset + v_id_in_u];

                    count += intersect_num(message_address + list_offset, message_address[degree_offset], g.N(v), g.get_degree(v));
                    
                    bitmask_32bit_value >>= 1;
                    bitmask_32bit_nonzero_id++;
                }
            }
            // turn 'valid' to 0
            if (thread_lane == 0)
                message_address[valid_offset] = 0;
        }
    }

    AccType block_num = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) atomicAdd(total, block_num);
}
