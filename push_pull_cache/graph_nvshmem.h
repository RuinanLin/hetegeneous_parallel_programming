#pragma once
#include "graph.h"
#include "nvshmem.h"
#include "nvshmemx.h"
#include "metis.h"

class GraphNVSHMEM {
protected:
    vidType n_vertices;         // number of vertices
    vidType n_real_vertices;    // how many vertices are stored in this GPU
    eidType n_edges;            // number of edges in this GPU (NOTICE!!! IN THIS GPU!!!)
    vidType max_subg_nv;        // the maximum number of vertices among all the subgraphs
    eidType max_subg_ne;        // the maximum number of edges among all the subgraphs
    vidType max_degree;         // the maximum degree in the whole initial graph
    int mype;                   // my GPU index
    size_t n_total_warps_in_a_grid;

    eidType *rowptr;            // 'rowptr' on GPU
    vidType *colidx;            // 'colidx' on GPU
    vidType *vertex_list;       // vertices stored on this GPU (this list is stored on GPU)
    idx_t *part;                // which vertex is on which GPU (this list is stored on GPU)
    vidType *pull_buffer;       // a scratchpad, storing the processing pulled data

public:
    GraphNVSHMEM(Graph &g, int mype_id, size_t n_total_warps) : mype(mype_id), n_total_warps_in_a_grid(n_total_warps) { init(g); }

    void init(Graph &hg) {
        // initialize parameters of the (sub)graph
        n_vertices = hg.V();
        n_real_vertices = hg.get_n_real_vertices();
        n_edges = hg.E();
        max_subg_nv = hg.get_max_subg_nv();
        max_subg_ne = hg.get_max_subg_ne();
        max_degree = hg.get_max_degree();

        // allocate GPU memory
        rowptr = (eidType *)nvshmem_malloc((n_vertices+1) * sizeof(eidType));
        colidx = (vidType *)nvshmem_malloc(max_subg_ne * sizeof(vidType));
        CUDA_SAFE_CALL(cudaMalloc((void **)&vertex_list, n_real_vertices * sizeof(vidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&part, n_vertices * sizeof(idx_t)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&pull_buffer, n_total_warps_in_a_grid * max_degree * sizeof(vidType)));

        // copy data
        CUDA_SAFE_CALL(cudaMemcpy(rowptr, hg.out_rowptr(), (n_vertices+1) * sizeof(eidType), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(colidx, hg.out_colidx(), n_edges * sizeof(vidType), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(vertex_list, hg.get_vertex_list(), n_real_vertices * sizeof(vidType), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(part, hg.get_metis_part(), n_vertices * sizeof(idx_t), cudaMemcpyHostToDevice));
    }

    inline __device__ __host__ vidType get_n_real_vertices() { return n_real_vertices; }
    inline __device__ __host__ vidType get_vertex_in_vertex_list(int local_id) { return vertex_list[local_id]; }
    inline __device__ __host__ eidType edge_begin(vidType u) { return rowptr[u]; }
    inline __device__ __host__ eidType edge_end(vidType u) { return rowptr[u+1]; }
    inline __device__ __host__ vidType getEdgeDst(eidType v_id_in_u_list) { return colidx[v_id_in_u_list]; }
    inline __device__ __host__ int get_vertex_partition_number(vidType v) { return (int)(part[v]); }
    inline __device__ __host__ vidType *N(vidType u) { return colidx + rowptr[u]; }
    inline __device__ __host__ vidType get_degree(vidType u) { return (vidType)(rowptr[u+1] - rowptr[u]); }
    inline __device__ __host__ vidType *get_my_pull_buffer(int warp_id) { return pull_buffer + warp_id * max_degree; }
    inline __device__ __host__ eidType *get_rowptr() { return rowptr; }
    inline __device__ __host__ vidType *get_colidx() { return colidx; }
};