#include <stdint.h>

#define PRE_ASSUMED_NDEVICES    4
#define NUM_WARPS_IN_A_BLOCK    8
#define PRE_ASSUMED_MAX_DEGREE  512

#define WARP_SIZE               32

typedef int32_t vidType;

typedef struct {
    int num_working_workers;
    int valid_for_sender[(PRE_ASSUMED_NDEVICES-1) * (NUM_WARPS_IN_A_BLOCK-1)];

    int msg_len[(PRE_ASSUMED_NDEVICES-1) * (NUM_WARPS_IN_A_BLOCK-1)];
    vidType msg_buffer[(PRE_ASSUMED_NDEVICES-1) * (NUM_WARPS_IN_A_BLOCK-1) * PRE_ASSUMED_MAX_DEGREE];
} shared_memory_buffer_t;


__device__ void init_smbuffer_block(shared_memory_buffer_t *smbuffer) {
    // about the threads and blocks
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    int warp_lane = threadIdx.x / WARP_SIZE;

    if (warp_lane == 0 && thread_lane == 0) {
        smbuffer->num_working_workers = NUM_WARPS_IN_A_BLOCK - 1;
    }
    if (warp_lane < NUM_WARPS_IN_A_BLOCK - 1 && thread_lane == 0) {
        for (int row = 0; row < PRE_ASSUMED_NDEVICES-1; row++) {
            smbuffer->valid_for_sender[row * (NUM_WARPS_IN_A_BLOCK-1) + warp_lane] = 0;
            smbuffer->msg_len[row * (NUM_WARPS_IN_A_BLOCK-1) + warp_lane] = 0;
        }
    }
    __threadfence_block();
    __syncthreads();     // TODO: can we remove it?
}


__device__ void producer_wait_warp(shared_memory_buffer_t *smbuffer, int warp_lane, int mype, int npes) {
    while (1) {
        int wait_counter = npes - 1;
        __threadfence_block();
        for (int row = 0; row < npes-1; row++) {
            if (smbuffer->valid_for_sender[row * (NUM_WARPS_IN_A_BLOCK-1) + warp_lane] == 0) {
                wait_counter--;
            }
        }
        if (wait_counter == 0) {
            break;
        }
    }
}


__device__ void producer_write_warp(shared_memory_buffer_t *smbuffer, int dest_id, int warp_lane, int position, vidType value, int mype) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    if (thread_lane == 0) {
        int row = (dest_id < mype)? dest_id : dest_id-1;
        smbuffer->msg_buffer[row * (NUM_WARPS_IN_A_BLOCK-1) * PRE_ASSUMED_MAX_DEGREE + warp_lane * PRE_ASSUMED_MAX_DEGREE + position] = value;
        if (position >= smbuffer->msg_len[row * (NUM_WARPS_IN_A_BLOCK-1) + warp_lane]) {
            smbuffer->msg_len[row * (NUM_WARPS_IN_A_BLOCK-1) + warp_lane] = position + 1;
        }
        __threadfence_block();  // TODO: can we remove it?
    }
    __syncwarp();   // TODO: can we remove it?
}


__device__ void producer_issue_warp(shared_memory_buffer_t *smbuffer, int mype, int dest_id, int warp_lane) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    if (thread_lane == 0) {
        int row = (dest_id < mype)? dest_id : dest_id-1;
        __threadfence_block();  // TODO: can we remove it?
        smbuffer->valid_for_sender[row * (NUM_WARPS_IN_A_BLOCK-1) + warp_lane] = 1;
        __threadfence_block();  // TODO: can we remove it?
    }
    __syncwarp();   // TODO: can we remove it?
}


__device__ void producer_exit_warp(shared_memory_buffer_t *smbuffer) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);
    if (thread_lane == 0) {
        atomicSub(&(smbuffer->num_working_workers), 1);
        __threadfence_block();  // TODO: can we remove it?
    }
}


// if time to terminate, return 1; otherwise return 0.
__device__ int consumer_get_msg_warp(shared_memory_buffer_t *smbuffer, vidType **msg_head, int *msg_length, int *dest_id, int *from_warp, int mype, int npes) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    while (1) {
        int done = 0;
        __threadfence_block();
        if (smbuffer->num_working_workers == 0) {
            done = 1;
        }

        for (int dest = 0; dest < npes; dest++) {
            if (dest == mype) continue;
            int row = (dest < mype)? dest : dest-1;
            for (int from_warp_id = 0; from_warp_id < (NUM_WARPS_IN_A_BLOCK-1); from_warp_id++) {
                __threadfence_block();
                if (smbuffer->valid_for_sender[row * (NUM_WARPS_IN_A_BLOCK-1) + from_warp_id] == 1) {
                    if (thread_lane == 0) {
                        *msg_head = &(smbuffer->msg_buffer[row * (NUM_WARPS_IN_A_BLOCK-1) * PRE_ASSUMED_MAX_DEGREE + from_warp_id * PRE_ASSUMED_MAX_DEGREE]);
                        *msg_length = smbuffer->msg_len[row * (NUM_WARPS_IN_A_BLOCK-1) + from_warp_id];
                        *dest_id = dest;
                        *from_warp = from_warp_id;
                    }
                    __syncwarp();   // TODO: can we remove it?
                    return 0;
                }
            }
        }

        if (done == 1) {
            return 1;
        }
    }
}


__device__ void consumer_release_msg_warp(shared_memory_buffer_t *smbuffer, int dest_id, int from_warp, int mype) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1);

    int row = (dest_id < mype)? dest_id : dest_id - 1;
    if (thread_lane == 0) {
        smbuffer->valid_for_sender[row * (NUM_WARPS_IN_A_BLOCK-1) + from_warp] = 0;
        smbuffer->msg_len[row * (NUM_WARPS_IN_A_BLOCK-1) + from_warp] = 0;
        __threadfence_block();
    }
    __syncwarp();   // TODO: can we remove it?
}