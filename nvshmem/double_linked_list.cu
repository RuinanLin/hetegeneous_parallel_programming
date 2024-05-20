#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define WARP_SIZE       32
#define DATA_LEN        1000
#define BUFFER_LEN      16
#define MAX_ELEM_SIZE   100

typedef struct {
    int type;   // 0: normal, 1: finish
    int package_size;
    int content[MAX_ELEM_SIZE];
} message_t;

typedef struct {
    int next[BUFFER_LEN+2];
    int prev[BUFFER_LEN+2];
    int empty_list_lock;
    int full_list_lock;
    int empty_list_elem_cnt;
    int full_list_elem_cnt;
} list_ctrl_t;

typedef enum {EMPTY_LIST, FULL_LIST} list_type_t;

__device__ int get_my_package_size(int mype) {
    int my_package_size;
    switch (mype) {
        case 0: my_package_size = 100; break;
        case 1: my_package_size = 20; break;
        case 2: my_package_size = 4; break;
        case 3: my_package_size = 50; break;
        default: my_package_size = 0;
    }
    return my_package_size;
}

__device__ void aquire_list_lock(int *lock, int dest) {
    while (nvshmem_int_atomic_swap(lock, 1, dest))
        ;
}

__device__ void release_list_lock(int *lock, int dest) {
    nvshmem_int_atomic_swap(lock, 0, dest);
}

__device__ void print_list(list_ctrl_t *list_controller, int dest, list_type_t type) {
    int mype = nvshmem_my_pe();
    printf("PE[%d] modifying PE[%d]'s list %d.\n\t", mype, dest, type);
    
    // full list or empty list?
    nvshmem_fence();
    int head_node_id = (type == EMPTY_LIST) ? BUFFER_LEN : BUFFER_LEN+1;
    int node_id = head_node_id;
    do {
        node_id = nvshmem_int_g(&(list_controller->next[node_id]), dest);
        printf("%d ", node_id);
    } while (node_id != head_node_id);
    printf("\n");

    // print next, prev
    printf("\t");
    for (size_t i = 0; i < BUFFER_LEN+2; i++) {
        int next_id = nvshmem_int_g(&(list_controller->next[i]), dest);
        printf("%d ", next_id);
    }
    printf("\n\t");
    for (size_t i = 0; i < BUFFER_LEN+2; i++) {
        int prev_id = nvshmem_int_g(&(list_controller->prev[i]), dest);
        printf("%d ", prev_id);
    }
    printf("\n");

    // print number
    int *elem_cnt_p = (type == EMPTY_LIST) ? &(list_controller->empty_list_elem_cnt) : &(list_controller->full_list_elem_cnt);
    int elem_cnt = nvshmem_int_g(elem_cnt_p, dest);
    printf("\telem_cnt = %d\n", elem_cnt);

    nvshmem_fence();
}

/******************************************** producer *********************************************/

__device__ int get_empty_queue_head(list_ctrl_t *list_controller, int dest) {
    // aquire the empty list lock
    int current_empty_list_elem_cnt;
    while (1) {
        // aquire the empty list lock
        aquire_list_lock(&(list_controller->empty_list_lock), dest);
        // aquire_list_lock(&(list_controller->empty_list_lock), 0);

        // does there exist available elements in empty list?
        nvshmem_fence();
        current_empty_list_elem_cnt = nvshmem_int_g(&(list_controller->empty_list_elem_cnt), dest);
        if (current_empty_list_elem_cnt > 0)    // yes, there exists
            break;
        else {
            // release the lock and come again in the next loop
            release_list_lock(&(list_controller->empty_list_lock), dest);
            // release_list_lock(&(list_controller->empty_list_lock), 0);
            // printf("PE[%d]: waiting for %d to consume ...\n", nvshmem_my_pe(), dest);
        }
    }
    
    // get current status
    nvshmem_fence();
    int current_head_id = nvshmem_int_g(&(list_controller->next[BUFFER_LEN]), dest);    // 'BUFFER_LEN' is the head node
    int next_head_id = nvshmem_int_g(&(list_controller->next[current_head_id]), dest);

    // modify the record
    nvshmem_int_p(&(list_controller->next[BUFFER_LEN]), next_head_id, dest);
    nvshmem_int_p(&(list_controller->prev[next_head_id]), BUFFER_LEN, dest);
    nvshmem_fence();

    // refresh the 'empty_list_elem_cnt'
    nvshmem_int_p(&(list_controller->empty_list_elem_cnt), current_empty_list_elem_cnt - 1, dest);
    nvshmem_fence();

    // print the dest's current empty list
    // print_list(list_controller, dest, EMPTY_LIST);
    // nvshmem_fence();

    // release the empty list lock
    release_list_lock(&(list_controller->empty_list_lock), dest);
    // release_list_lock(&(list_controller->empty_list_lock), 0);

    return current_head_id;
}

__device__ void put_full_queue_tail(list_ctrl_t *list_controller, int elem_id, int dest) {
    // get the full list lock
    aquire_list_lock(&(list_controller->full_list_lock), dest);
    // aquire_list_lock(&(list_controller->empty_list_lock), 0);
    
    // get the current full queue tail
    nvshmem_fence();
    int current_tail_id = nvshmem_int_g(&(list_controller->prev[BUFFER_LEN+1]), dest);

    // insert to the tail of the list
    nvshmem_int_p(&(list_controller->prev[BUFFER_LEN+1]), elem_id, dest);
    nvshmem_int_p(&(list_controller->next[elem_id]), BUFFER_LEN+1, dest);
    nvshmem_int_p(&(list_controller->next[current_tail_id]), elem_id, dest);
    nvshmem_int_p(&(list_controller->prev[elem_id]), current_tail_id, dest);
    nvshmem_fence();

    // refresh the 'full_list_elem_cnt'
    nvshmem_fence();
    int current_full_list_elem_cnt = nvshmem_int_g(&(list_controller->full_list_elem_cnt), dest);
    nvshmem_int_p(&(list_controller->full_list_elem_cnt), current_full_list_elem_cnt + 1, dest);
    nvshmem_fence();

    // print the dest's current full list
    // print_list(list_controller, dest, FULL_LIST);
    // nvshmem_fence();

    // release the full list lock
    release_list_lock(&(list_controller->full_list_lock), dest);
    // release_list_lock(&(list_controller->empty_list_lock), 0);
}

__device__ void send_message(message_t *data_buffer, list_ctrl_t *list_controller, int type, int start_num_this_round, int my_package_size, int dest) {
    // get the empty list head
    int elem_id = get_empty_queue_head(list_controller, dest);

    // put the message
    message_t *my_message_elem;
    my_message_elem = data_buffer + elem_id;
    nvshmem_int_p(&(my_message_elem->type), type, dest);
    nvshmem_int_p(&(my_message_elem->package_size), my_package_size, dest);
    if (type == 0) {    // normal message
        for (size_t i = 0; i < my_package_size; i++) {
            nvshmem_int_p(&(my_message_elem->content[i]), start_num_this_round + i, dest);
        }
    }
    nvshmem_fence();

    // put it into the full list
    put_full_queue_tail(list_controller, elem_id, dest);
}

/********************************************** consumer *********************************************/

__device__ int get_full_queue_head(list_ctrl_t *list_controller) {
    // aquire the full list lock
    int mype = nvshmem_my_pe();
    int current_full_list_elem_cnt;
    while (1) {
        // aquire the full list lock
        aquire_list_lock(&(list_controller->full_list_lock), mype);

        // does there exist available elements in full list?
        nvshmem_fence();
        current_full_list_elem_cnt = list_controller->full_list_elem_cnt;
        if (current_full_list_elem_cnt > 0)     // yes, there exists
            break;
        else {
            // release the lock and come again in the next loop
            release_list_lock(&(list_controller->full_list_lock), mype);
        }
    }
    
    // get current status
    nvshmem_fence();
    int current_head_id = list_controller->next[BUFFER_LEN+1];
    int next_head_id = list_controller->next[current_head_id];

    // modify the record
    list_controller->next[BUFFER_LEN+1] = next_head_id;
    list_controller->prev[next_head_id] = BUFFER_LEN+1;
    nvshmem_fence();

    // refresh the 'full_list_elem_cnt'
    list_controller->full_list_elem_cnt--;
    nvshmem_fence();

    // print my current full list
    // print_list(list_controller, mype, FULL_LIST);
    // nvshmem_fence();

    // release the full list lock
    release_list_lock(&(list_controller->full_list_lock), mype);
    // release_list_lock(&(list_controller->empty_list_lock), 0);

    return current_head_id;
}

__device__ void put_empty_queue_tail(list_ctrl_t *list_controller, int elem_id) {
    // get the empty list lock
    int mype = nvshmem_my_pe();
    aquire_list_lock(&(list_controller->empty_list_lock), mype);
    // aquire_list_lock(&(list_controller->empty_list_lock), 0);
    
    // get current status
    nvshmem_fence();
    int current_tail_id = list_controller->prev[BUFFER_LEN];

    // insert to the tail of the list
    list_controller->next[current_tail_id] = elem_id;
    list_controller->prev[elem_id] = current_tail_id;
    list_controller->prev[BUFFER_LEN] = elem_id;
    list_controller->next[elem_id] = BUFFER_LEN;

    // refresh the 'empty_list_elem_cnt'
    nvshmem_fence();
    list_controller->empty_list_elem_cnt++;
    nvshmem_fence();

    // print my current empty list
    // print_list(list_controller, mype, EMPTY_LIST);
    // nvshmem_fence();

    // release the empty list lock
    release_list_lock(&(list_controller->empty_list_lock), mype);
    // release_list_lock(&(list_controller->empty_list_lock), 0);
}

__device__ void recv_message(message_t *data_buffer, list_ctrl_t *list_controller, int *finish_count, int *message_sum) {
    // get the full list head
    int elem_id = get_full_queue_head(list_controller);

    // decode the message
    nvshmem_fence();
    if (data_buffer[elem_id].type == 0) {   // normal
        int package_size = data_buffer[elem_id].package_size;
        int local_count = 0;
        for (size_t i = 0; i < package_size; i++) {
            local_count += data_buffer[elem_id].content[i];
        }
        *message_sum += local_count;
    } else {    // finish
        // printf("PE[%d] gets the finish signal.\n", nvshmem_my_pe());
        (*finish_count)++;
    }
    nvshmem_fence();

    // insert into the empty list's tail
    put_empty_queue_tail(list_controller, elem_id);
}

/******************************************************************************************/

__global__ void send_recv(message_t *data_buffer, list_ctrl_t *list_controller) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();

    int thread_lane = threadIdx.x & (WARP_SIZE-1);  // thread index within the warp
    int warp_lane = threadIdx.x / WARP_SIZE;        // warp index within the CTA

    if (thread_lane == 0) {
        if (warp_lane > 0) {   // sender
            int my_package_size = get_my_package_size(mype);
            // send its messages out
            for (int start_num_this_round = 0; start_num_this_round < DATA_LEN; start_num_this_round += my_package_size) {
                for (int dest = 0; dest < npes; dest++) {
                    if (dest == mype) continue;
                    send_message(data_buffer, list_controller, 0, start_num_this_round, my_package_size, dest);
                }
            }
            // send the finish message
            for (int dest = 0; dest < npes; dest++) {
                if (dest == mype) continue;
                send_message(data_buffer, list_controller, 1, 0, 0, dest);
                // printf("PE[%d] finished.\n", mype);
            }
        } else {    // receiver
            int finish_count = 0;
            int message_sum = 0;
            while (finish_count < (npes - 1) * (3 - 1)) {
                recv_message(data_buffer, list_controller, &finish_count, &message_sum);
            }
            printf("PE[%d]: message_sum = %d\n", mype, message_sum);
        }
    }
}

void init_list_controller(list_ctrl_t *list_controller) {
    // empty list
    for (size_t i = 0; i < BUFFER_LEN; i++) {
        list_controller->next[i] = i+1;
    }
    list_controller->next[BUFFER_LEN] = 0;
    list_controller->next[BUFFER_LEN+1] = BUFFER_LEN+1;

    // full list
    list_controller->prev[0] = BUFFER_LEN;
    for (size_t i = 0; i < BUFFER_LEN; i++) {
        list_controller->prev[i+1] = i;
    }
    list_controller->prev[BUFFER_LEN+1] = BUFFER_LEN+1;

    list_controller->empty_list_lock = 0;
    list_controller->full_list_lock = 0;
    list_controller->empty_list_elem_cnt = BUFFER_LEN;
    list_controller->full_list_elem_cnt = 0;
}

int main(void) {
    nvshmem_init();

    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);

    list_ctrl_t list_controller_h;
    init_list_controller(&list_controller_h);

    message_t *data_buffer = (message_t *)nvshmem_malloc(BUFFER_LEN * sizeof(message_t));
    list_ctrl_t *list_controller = (list_ctrl_t *)nvshmem_malloc(sizeof(list_ctrl_t));

    cudaMemcpy(list_controller, &list_controller_h, sizeof(list_ctrl_t), cudaMemcpyHostToDevice);

    dim3 gridDim(1), blockDim(3 * WARP_SIZE);
    void *args[] = { &data_buffer, &list_controller };

    nvshmem_barrier_all();
    nvshmemx_collective_launch((const void *)send_recv, gridDim, blockDim, args, 0, 0);
    nvshmem_barrier_all();

    nvshmem_free(data_buffer);
    nvshmem_free(list_controller);

    nvshmem_finalize();
    return 0;
}