#pragma once
#include "graph.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "graph_nvshmem.h"

#define INDEX_BITS  6
#define TAG_BITS    (sizeof(vidType) - INDEX_BITS)

#define NUM_LINES   (1 << (INDEX_BITS))
#define NUM_WAYS    8

class GraphCache {
private:
    vidType max_degree;

    int *line_lock;         // [ NUM_LINES ]
    int *valid_array;       // [ NUM_LINES, NUM_WAYS ]
    int *tag_array;         // [ NUM_LINES, NUM_WAYS ]
    vidType *degree_array;  // [ NUM_LINES, NUM_WAYS ]
    vidType *data_array;    // [ NUM_LINES, NUM_WAYS, max_degree ]

public:
    GraphCache(int md) : max_degree(md) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&line_lock, NUM_LINES * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&valid_array, NUM_LINES * NUM_WAYS * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&tag_array, NUM_LINES * NUM_WAYS * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&degree_array, NUM_LINES * NUM_WAYS * sizeof(vidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&data_array, NUM_LINES * NUM_WAYS * max_degree * sizeof(vidType)));

        CUDA_SAFE_CALL(cudaMemset(line_lock, 0, NUM_LINES * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemset(valid_array, 0, NUM_LINES * NUM_WAYS * sizeof(int)));
    }
};