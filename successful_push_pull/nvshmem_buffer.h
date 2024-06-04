#pragma once

class NVSHMEMBuffer {
private:
    // parameters
    int slot_size;
    int num_warp_groups;

    // control
    uint64_t *valid;        // [ num_warp_groups ]
    uint64_t *ready;        // [ num_warp_groups ]
    int *producer_dest_pe;  // [ num_warp_groups ]
    int *consumer_src_pe;   // [ num_warp_groups ]

    // content
    vidType *content;       // [ num_warp_groups, slot_size ]

public:
    NVSHMEMBuffer(int slot_sz, int n_warp_groups) : slot_size(slot_sz), num_warp_groups(n_warp_groups) {
        valid = (uint64_t *)nvshmem_malloc(num_warp_groups * sizeof(uint64_t));
        ready = (uint64_t *)nvshmem_malloc(num_warp_groups * sizeof(uint64_t));
        cudaMalloc((void **)&producer_dest_pe, num_warp_groups * sizeof(int));
        cudaMalloc((void **)&consumer_src_pe, num_warp_groups * sizeof(int));
        content = (vidType *)nvshmem_malloc(num_warp_groups * slot_size * sizeof(vidType));

        uint64_t *valid_h = (uint64_t *)malloc(num_warp_groups * sizeof(uint64_t));
        for (int i = 0; i < num_warp_groups; i++) {
            valid_h[i] = 0;
        }
        cudaMemcpy(valid, valid_h, num_warp_groups * sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t *ready_h = (uint64_t *)malloc(num_warp_groups * sizeof(uint64_t));
        for (int i = 0; i < num_warp_groups; i++) {
            ready_h[i] = 1;
        }
        cudaMemcpy(ready, ready_h, num_warp_groups * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // this function should be called by a single thread
    __device__ void producer_init_dest_pe(int buffer_id, int dest_pe_id) {
        producer_dest_pe[buffer_id] = dest_pe_id;
    }

    __device__ void producer_warp_write_msg(int buffer_id, vidType *src, int nelem) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);

        // if (thread_lane == 0 && nvshmem_my_pe() == 0 && buffer_id == 0) {
        //     printf("pe 0 warp 0 reads ready[0] = \n");
        // } __syncwarp();
        
        nvshmem_signal_wait_until(&ready[buffer_id], NVSHMEM_CMP_EQ, 1);
        __syncwarp();

        // if (thread_lane == 0 && nvshmem_my_pe() == 0 && buffer_id == 0) {
        //     printf("pe 0 warp 0 reads ready++[0] = \n");
        // } __syncwarp();

        if (thread_lane == 0) {
            ready[buffer_id] = 0;
        } __syncwarp();
        nvshmem_fence();    // TODO: can we delete this?
        nvshmemx_int32_put_signal_warp(&content[buffer_id * slot_size], src, nelem, &valid[buffer_id], 1, NVSHMEM_SIGNAL_SET, producer_dest_pe[buffer_id]);
    }

    // this function should be called by a single thread
    __device__ void consumer_init_source_pe(int buffer_id, int src_pe_id) {
        consumer_src_pe[buffer_id] = src_pe_id;
    }

    __device__ vidType *consumer_warp_get_msg(int buffer_id) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        
        nvshmem_signal_wait_until(&valid[buffer_id], NVSHMEM_CMP_EQ, 1);
        __syncwarp();
        if (thread_lane == 0) {
            valid[buffer_id] = 0;
        } __syncwarp();
        nvshmem_fence();    // TODO: can we delete this?
        return &content[buffer_id * slot_size];
    }

    __device__ void consumer_warp_release_msg(int buffer_id) {
        int thread_lane = threadIdx.x & (WARP_SIZE-1);
        if (thread_lane == 0) {
            nvshmemx_signal_op(&ready[buffer_id], 1, NVSHMEM_SIGNAL_SET, consumer_src_pe[buffer_id]);
        } __syncwarp();
    }
};






































// class NVSHMEMBuffer {
// private:
//     // parameters
//     int slot_size;          // how many elements a single slot can hold
//     int num_warp_groups;    // total number of producers/consumers

//     // recv
//     uint64_t *valid;        // [ num_warp_groups ]  // every time a remote GPU's sender finishes writing data, the valid[warp_id] will be set

//     // send
//     uint64_t *ready;        // [ num_warp_groups ]  // every time a remote GPU's recver finishes dealing with the data, the ready[warp_id] will be set

//     // roles record info
//     int *senders_dest;      // [ num_warp_groups ]  // which GPU will the sender send its message to
//     int *recvers_source;    // [ num_warp_groups ]  // which GPU will the recver receive the messages from

//     // content
//     int *msg_type;          // [ num_warp_groups ]  // 0: normal; 1: exit
//     int *msg_len;           // [ num_warp_groups ]
//     vidType *content;       // [ num_warp_groups, slot_size ]

// public:
//     NVSHMEMBuffer() : slot_size(slot_sz), num_warp_groups(n_warp_groups) {
//         valid = (uint64_t *)nvshmem_malloc(num_warp_groups * sizeof(uint64_t));
//         ready = (uint64_t *)nvshmem_malloc(num_warp_groups * sizeof(uint64_t));
//         CUDA_SAFE_CALL(cudaMalloc((void **)&senders_dest, num_warp_groups * sizeof(int)));
//         CUDA_SAFE_CALL(cudaMalloc((void **)&recvers_source, num_warp_groups * sizeof(int)));
//         msg_type = (int *)nvshmem_malloc(num_warp_groups * sizeof(int));
//         msg_len = (int *)nvshmem_malloc(num_warp_groups * sizeof(int));
//         content = (vidType *)nvshmem_malloc(num_warp_groups * slot_size * sizeof(vidType));

//         uint64_t *valid_h = (uint64_t *)malloc(num_warp_groups * sizeof(uint64_t));
//         for (int i = 0; i < num_warp_groups; i++) {
//             valid_h[i] = 0;
//         }
//         CUDA_SAFE_CALL(cudaMemcpy(valid, valid_h, num_warp_groups * sizeof(uint64_t), cudaMemcpyHostToDevice));
//         free(valid_h);

//         uint64_t *ready_h = (uint64_t *)malloc(num_warp_groups * sizeof(uint64_t));
//         for (int i = 0; i < num_warp_groups; i++) {
//             ready_h[i] = 1;
//         }
//         CUDA_SAFE_CALL(cudaMemcpy(ready, ready_h, num_warp_groups * sizeof(uint64_t), cudaMemcpyHostToDevice));
//         free(ready_h);
//     }


//     __device__ int init_sender_dest_info(warp_id) {
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);

//         int ndevices = nvshmem_n_pes();
//         int dest_warp_group_size = num_warp_groups / (ndevices-1);
//         if (warp_id >= dest_warp_group_size * (ndevices-1)) {
//             if (thread_lane == 0) {
//                 senders_dest[warp_id] = -1;
//             }
//             __syncwarp();   // TODO: can we remove it?
//             return -1;
//         } else {
//             if (thread_lane == 0) {
//                 int row = warp_id / (ndevices-1);
//                 senders_dest[warp_id] = (row < nvshmem_my_pe())? row : row+1;
//             }
//             __syncwarp();   // TODO: can we remove it?
//         }

//         __threadfence();    // TODO: can we remove it?
//         return senders_dest[warp_id];
//     }


//     __device__ void init_recver_source_info(warp_id) {
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);

//         int ndevices = nvshmem_n_pes();
//         int dest_warp_group_size = num_warp_groups / (ndevices-1);
//         if (
//     }


//     __device__ void init_recver_dest_info()


//     __device__ void producer_write_msg(int warp_id, int message_size, vidType *message, int dest_id, int type) {
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);

//         nvshmem_signal_wait_until(&ready[warp_id], NVSHMEM_CMP_EQ, 1);
//         if (thread_lane == 0) {
//             ready[warp_id] = 0;
//             nvshmem_fence();    // TODO: can we remove it?
//         }
//         __syncwarp();   // TODO: can we remove it?

//         // write the type
//         if (thread_lane == 0) {
//             nvshmem_int_p(&msg_type[warp_id], type, dest_id);
//         }
//         __syncwarp();   // TODO: can we remove it?

//         if (type == 0) {
//             // write the message_size
//             if (thread_lane == 0) {
//                 nvshmem_int_p(&msg_len[warp_id], message_size, dest_id);
//             }
//             __syncwarp();   // TODO: can we remove it?
//             nvshmemx_int32_put_signal_warp(&content[warp_id * slot_size], message, message_size, &valid[warp_id], 1, NVSHMEM_SIGNAL_SET, dest_id);
//             nvshmem_fence();    // TODO: can we remove it?
//         }
//     }

//     __device__ int consumer_get_msg_pointer(int warp_id, vidType **start) {
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);

//         while (1) {
//             // wait for a new message
//             nvshmem_signal_wait_until(&valid[warp_id], NVSHMEM_CMP_EQ, 1);

//             if (thread_lane == 0) {
//                 valid[warp_id] = 0;
//                 nvshmem_fence();    // TODO: can we remove it?
//             }
//             __syncwarp();   // TODO: can we remove it?

//             nvshmem_fence();    // TODO: can we remove it?
//             if (msg_type[warp_id] == 1) {   // it's time to stop
//                 return 1;
//             } else {
//                 *start = &content[warp_id * slot_size];
//                 return 0;
//             }
//         }
//     }


//     __device__ void consumer_release(int warp_id) {
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);

//         if (thread_lane == 0) {
//             nvshmemx_signal_op(&ready[warp_id], 1, NVSHMEM_SIGNAL_SET, );
//         }
//     }
// }

















































// class NVSHMEMBuffer {
// private:
//     // parameters
//     int my_id;
//     int ndevices;
//     int slot_size;                  // how many elements this slot can hold
//     int num_warp_groups;            // total numbers of producers/consumers

//     // recv
//     int *valid;                     // [ num_warp_groups, ndevices-1 ]      // every time a remote GPU's sender finishes writing data, the valid[warp_id][dest_row_id_here] will be set
//     uint64_t *valid_signal_count;   // [ num_warp_groups, 1 ]               // count the number of valid signals. it is useful when the recver warp checks whether there are new messages

//     // send
//     uint64_t *ready;                // [ num_warp_groups, ndevices-1 ]      // every time a remote GPU's recver finishes dealing with the data, the ready[warp_id][dest_row_id_here] will be set

//     // termination
//     int *num_working_producers;     // [ num_warp_groups, 1 ]               // how many peer producer warps are still going to produce new messages

//     // content
//     int *msg_type;                  // [ num_warp_groups, ndevices-1 ]      // 0: normal; 1: exit
//     int *msg_len;                   // [ num_warp_groups, ndevices-1 ]
//     vidType *content;               // [ num_warp_groups, ndevices-1, slot_size ]

// public:
//     NVSHMEMBuffer(int mype, int num_devices, int slot_sz, int num_producers_each_warp_group, int n_warp_groups) :
//         my_id(mype), ndevices(num_devices), slot_size(slot_sz), num_warp_groups(n_warp_groups) {
//         valid = (int *)nvshmem_malloc(num_warp_groups * (ndevices-1) * sizeof(int));
//         valid_signal_count = (uint64_t *)nvshmem_malloc(num_warp_groups * sizeof(uint64_t));
//         ready = (uint64_t *)nvshmem_malloc(num_warp_groups * (ndevices-1) * sizeof(uint64_t));
//         CUDA_SAFE_CALL(cudaMalloc((void **)&num_working_producers, num_warp_groups * sizeof(int)));
//         msg_type = (int *)nvshmem_malloc(num_warp_groups * (ndevices-1) * sizeof(int));
//         msg_len = (int *)nvshmem_malloc(num_warp_groups * (ndevices-1) * sizeof(int));
//         content = (vidType *)nvshmem_malloc(num_warp_groups * (ndevices-1) * slot_size * sizeof(vidType));

//         int *valid_h = (int *)malloc(num_warp_groups * (ndevices-1) * sizeof(int));
//         for (int i = 0; i < num_warp_groups * (ndevices-1); i++) {
//             valid_h[i] = 0;
//         }
//         CUDA_SAFE_CALL(cudaMemcpy(valid, valid_h, num_warp_groups * (ndevices-1) * sizeof(int), cudaMemcpyHostToDevice));

//         uint64_t *valid_signal_count_h = (uint64_t *)malloc(num_warp_groups * sizeof(uint64_t));
//         for (int i = 0; i < num_warp_groups; i++) {
//             valid_signal_count_h[i] = 0;
//         }
//         CUDA_SAFE_CALL(cudaMemcpy(valid_signal_count, valid_signal_count_h, num_warp_groups * sizeof(uint64_t), cudaMemcpyHostToDevice));

//         uint64_t *ready_h = (uint64_t *)malloc(num_warp_groups * (ndevices-1) * sizeof(uint64_t));
//         for (int i = 0; i < num_warp_groups * (ndevices-1); i++) {
//             ready_h[i] = 1;
//         }
//         CUDA_SAFE_CALL(cudaMemcpy(ready, ready_h, num_warp_groups * (ndevices-1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

//         int *num_working_producers_h = (int *)malloc(num_warp_groups * sizeof(int));
//         for (int i = 0; i < num_warp_groups; i++) {
//             num_working_producers_h[i] = num_producers_each_warp_group;
//         }
//         CUDA_SAFE_CALL(cudaMemcpy(num_working_producers, num_working_producers_h, num_warp_groups * sizeof(int), cudaMemcpyHostToDevice));
//     }

//     inline __device__ int get_my_id() { return my_id; }
//     inline __device__ int get_ndevices() { return ndevices; }

//     __device__ void producer_write_msg(int warp_id, int message_size, vidType *message, int dest_id, int type) {
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);

//         int dest_row_id_here = (dest_id < my_id)? dest_id : dest_id - 1;
//         nvshmem_signal_wait_until(&ready[warp_id * (ndevices-1) + dest_row_id_here], NVSHMEM_CMP_EQ, 1);
//         ready[warp_id * (ndevices-1) + dest_row_id_here] = 0;
//         nvshmem_fence();    // can it be deleted?

//         int my_row_id_there = (my_id < dest_id)? my_id : my_id - 1;
//         if (thread_lane == 0) {
//             nvshmem_int_p(&msg_type[warp_id * (ndevices-1) + my_row_id_there], type, dest_id);
//         }
//         __syncwarp();   // can it be deleted?
//         if (type == 0) {
//             if (thread_lane == 0) {
//                 nvshmem_int_p(&msg_len[warp_id * (ndevices-1) + my_row_id_there], message_size, dest_id);
//             }
//             __syncwarp();
//             nvshmemx_int32_put_warp(&content[warp_id * (ndevices-1) * slot_size + my_row_id_there * slot_size], message, message_size, dest_id);
//             nvshmem_fence();    // can it be deleted?
//         }

//         if (thread_lane == 0) {
//             nvshmem_int_p(&valid[warp_id * (ndevices-1) + my_row_id_there], 1, dest_id);
//             nvshmemx_signal_op(&valid_signal_count[warp_id], 1, NVSHMEM_SIGNAL_ADD, dest_id);
//             nvshmem_fence();    // can it be deleted?
//         }
//         __syncwarp();   // can it be deleted?
//     }

//     __device__ int consumer_get_msg_pointer(int warp_id, vidType **start, int *degree, int *dest) {
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);

//         while (1) {
//             // first, check whether all the sender warps have exited
//             if (num_working_producers[warp_id] == 0) {
//                 return 1;
//             }

//             // wait for a new message
//             nvshmem_signal_wait_until(&valid_signal_count[warp_id], NVSHMEM_CMP_GT, 0);

//             // find the new message
//             for (int dest_id = 0; dest_id < ndevices; dest_id++) {
//                 if (dest_id == my_id) continue;
//                 int dest_row_id_here = (dest_id < my_id)? dest_id : dest_id-1;
//                 nvshmem_fence();    // can it be deleted?
//                 if (valid[warp_id * (ndevices-1) + dest_row_id_here] == 0) continue;
//                 if (thread_lane == 0) {
//                     valid[warp_id * (ndevices-1) + dest_row_id_here] = 0;
//                     nvshmem_uint64_atomic_add(&valid_signal_count[warp_id], -1, my_id); // can we change it back to normal?
//                     nvshmem_fence();    // can it be deleted?
//                 }
//                 __syncwarp();

//                 nvshmem_fence();    // can it be deleted?

//                 if (msg_type[warp_id * (ndevices-1) + dest_row_id_here] == 1) {
//                     if (thread_lane == 0) {
//                         // printf("pe %d recver warp %d received an exit message. old num_working_producers = %d\n", nvshmem_my_pe(), warp_id, num_working_producers[warp_id]);
//                         num_working_producers[warp_id]--;
//                     }
//                     __syncwarp();   // can it be deleted?
//                     consumer_release(warp_id, dest_id);
//                 } else {
//                     *degree = msg_len[warp_id * (ndevices-1) + dest_row_id_here];
//                     *start = &content[warp_id * (ndevices-1) * slot_size + dest_row_id_here * slot_size];
//                     *dest = dest_id;
//                     return 0;
//                 }
//             }
//         }
//     }

//     __device__ void consumer_release(int warp_id, int dest_id) {
//         int my_row_id_there = (my_id < dest_id)? my_id : my_id - 1;
//         int thread_lane = threadIdx.x & (WARP_SIZE-1);
//         if (thread_lane == 0) {
//             nvshmemx_signal_op(&ready[warp_id * (ndevices-1) + my_row_id_there], 1, NVSHMEM_SIGNAL_SET, dest_id);
//         }
//         __syncwarp();
//     }
// };