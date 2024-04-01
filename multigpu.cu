#include <cub/cub.cuh>
//#include "edgelist.h"
#include "graph_gpu.h"
#include "graph_partition.h"
#include "scheduler.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "bs_warp_edge.cuh"
#include "bs_warp_vertex.cuh"
#include <thread>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>
#include <omp.h>
#include "set_intersect.cuh"

typedef std::pair<vidType, vidType> directedEdge;

int get_type3_subgraph_num(int device_count);
void get_type3_subgraph_tuple(int *this_subgraph_tuple, int this_type3_subgraph_idx, int num_partitions);
int get_type3_subgraph_idx(int device_count, vidType u, vidType v, vidType w);
void sort_three_vertices(vidType *u, vidType *v, vidType *w);
__global__ void warp_vertex_type1(AccType *d_count, vidType *d_edges, eidType *d_row_pointers, vidType *d_num_vertices);
__global__ void warp_vertex_type3(vidType *d_num_partitions, vidType *d_this_num_vertices, AccType *d_count, vidType *d_edges_0_to_1, eidType *d_row_pointers_0_to_1, vidType *d_edges_0_to_2, eidType *d_row_pointers_0_to_2, vidType *d_edges_1_to_2, eidType *d_row_pointers_1_to_2);

/**************************************** Definition of Type1SubGraph ***************************************/

class Type1SubGraph {
  protected:
    vidType num_vertices;                 // how many vertices in the super graph
    std::ofstream logFile;                // its own logFile

    std::vector<vidType> edges;           // column indices of CSR format, starting point only inner
    eidType *row_pointers;                // row pointers of CSR format, starting point only inner

    vidType current_from;                 // when creating the subgraph, record where the current start index, to inspect changes

  public:
    void init(Graph &g, int this_part_idx);
    void destroy();
    void add_edge(vidType from, vidType to);
    void end_adding_edge();
    vidType get_out_degree(vidType u);
    vidType N(vidType u, vidType n);
    vidType get_num_vertices();
    eidType get_num_edges();
    vidType *get_edges_pointer();
    eidType *get_row_pointers_pointer();
    eidType *get_inner_edge_starts_pointer();
    eidType *get_inner_edge_ends_pointer();
    void print_edge_info();
};

// initialize the subgraph
void Type1SubGraph::init(Graph &g, int this_part_idx)
{
  // initialize its own logFile
  std::string file_name = "Type1SubGraph_log" + std::to_string(this_part_idx) + ".txt";
  logFile.open(file_name);
  if (!logFile)
  {
    std::cerr << "Cannot open " << file_name << "\n";
    exit(-1);
  }
  // logFile << "logFile created!\nType1SubGraph " << this_part_idx << " initialization starts ...\n";

  // initialize private variables
  num_vertices = g.V();
  current_from = num_vertices;
  // logFile << "\tPrivate variables initialized!\n";
  // logFile << "\t\tnum_vertices = " << num_vertices << "\n";
  // logFile << "\t\tcurrent_from = " << current_from << "\n";

  // initialize the CSR
  // logFile << "\tAllocate row_pointers ...\n";
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&row_pointers, (num_vertices + 1) * sizeof(eidType), cudaHostAllocDefault));
  row_pointers[num_vertices] = 0;

  // finish initialization and exit
  // logFile << "Initialization succeeded!\n";
}

// destroy the subgraph
void Type1SubGraph::destroy()
{
  // logFile << "Start destroying Type1SubGraph " << this_partition_idx << " ...\n";

  // free the "edges_from_outside" and "edges_to_outside"
  // logFile << "\tFree the allocated memory ...\n";
  CUDA_SAFE_CALL(cudaFreeHost(row_pointers));
  // logFile << "\tAllocated memory freed!\n";

  // close the logFile
  // logFile << "Gracefully finishing ...\n";
  logFile.close();
}

// push a new edge to the temp_edges
void Type1SubGraph::add_edge(vidType from, vidType to)
{
  // judge whether we should record the row_pointers
  if (from != current_from)
  {
    for (vidType u = current_from + 1; u <= from; u++)
      row_pointers[u] = edges.size();
    current_from = from;
  }

  // push it into the edges vector
  edges.push_back(to);

  // the last element of row_pointers records the num_edges
  row_pointers[num_vertices]++;
}

void Type1SubGraph::end_adding_edge()
{
  for (vidType u = current_from + 1; u < num_vertices; u++)
    row_pointers[u] = edges.size();
}

// get the out-degree of vertex u
vidType Type1SubGraph::get_out_degree(vidType u) { return row_pointers[u + 1] - row_pointers[u]; }

// get the n-th neighbor of u
vidType Type1SubGraph::N(vidType u, vidType n) { return edges[row_pointers[u] + n]; }

vidType Type1SubGraph::get_num_vertices() { return num_vertices; }
eidType Type1SubGraph::get_num_edges() { return row_pointers[num_vertices]; }
vidType * Type1SubGraph::get_edges_pointer() { return &edges[0]; }
eidType * Type1SubGraph::get_row_pointers_pointer() { return row_pointers; }

void Type1SubGraph::print_edge_info()
{
  logFile << num_vertices << "\n";
  for (vidType u = 0; u < num_vertices; u++)
  {
    logFile << u << ": ";
    vidType u_deg = get_out_degree(u);
    for (vidType v_idx = 0; v_idx < u_deg; v_idx++)
    {
      vidType v = edges[row_pointers[u] + v_idx];
      logFile << v << " ";
    }
    logFile << "\n";
  }
}

/**************************************** Definition of Type1SubGraphGPU ***************************************/

class Type1SubGraphGPU {
  protected:
    vidType h_num_vertices;
    vidType *d_num_vertices;
    eidType num_edges;                    // how many edges in this partition

    vidType *h_edges;
    vidType *d_edges;                     // column indices of CSR format, starting point only inner
    eidType *h_row_pointers;
    eidType *d_row_pointers;              // row pointers of CSR format, starting point only inner

    AccType h_count;
    AccType *d_count;

  public:
    void init(Type1SubGraph &g);
    void launch(cudaStream_t stream);
    AccType get_count();
};

// initialize the Type1SubGraphGPU
void Type1SubGraphGPU::init(Type1SubGraph &g)
{
  // initialize the protected variables
  h_num_vertices = g.get_num_vertices();
  num_edges = g.get_num_edges();
  h_edges = g.get_edges_pointer();
  h_row_pointers = g.get_row_pointers_pointer();
  h_count = 0;
}

// launch the device
void Type1SubGraphGPU::launch(cudaStream_t stream)
{
  // allocate device memory
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_vertices, sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_num_vertices, &h_num_vertices, sizeof(vidType), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges, num_edges * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_edges, h_edges, num_edges * sizeof(vidType), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_pointers, (h_num_vertices + 1) * sizeof(eidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_row_pointers, h_row_pointers, (h_num_vertices + 1) * sizeof(eidType), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_count, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_count, &h_count, sizeof(AccType), cudaMemcpyHostToDevice, stream));

  // launch kernel
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (h_num_vertices - 1) / WARPS_PER_BLOCK + 1;
  if (nblocks > 65536) nblocks = 65536;
  warp_vertex_type1<<<nblocks, nthreads, 0, stream>>>(d_count, d_edges, d_row_pointers, d_num_vertices);

  // copy the result out and free the allocated device memory
  CUDA_SAFE_CALL(cudaMemcpyAsync(&h_count, d_count, sizeof(AccType), cudaMemcpyDeviceToHost, stream));
  CUDA_SAFE_CALL(cudaFree(d_num_vertices));
  CUDA_SAFE_CALL(cudaFree(d_edges));
  CUDA_SAFE_CALL(cudaFree(d_row_pointers));
  CUDA_SAFE_CALL(cudaFree(d_count));
}

// get the count of this Type1SubGraphGPU
AccType Type1SubGraphGPU::get_count() { return h_count; }

/**************************************** Definition of Type3SubGraph ***************************************/

class Type3SubGraph {
  protected:
    int this_type3_subgraph_idx;                                            // the global index among all the Type3SubGraphs
    int num_partitions;                                                     // how many partitions in the super graph
    int this_subgraph_tuple[3];                                             // records the three indices of the partitions, sorted
    vidType super_num_vertices;                                             // how many vertices in the super graph
    vidType partition_num_vertices_tuple[3];                                // how many vertices are there in each partition
    std::ofstream logFile;                                                  // its own logFile

    vidType *edges_0_to_1;
    eidType *row_pointers_0_to_1;
    std::vector<directedEdge> temp_edges_0_to_1;

    vidType *edges_0_to_2;
    eidType *row_pointers_0_to_2;
    std::vector<directedEdge> temp_edges_0_to_2;

    vidType *edges_1_to_2;
    eidType *row_pointers_1_to_2;
    std::vector<directedEdge> temp_edges_1_to_2;

    int creation_finished;

  public:
    void init(Graph &g, int num_part, int this_type3_subgraph_index);
    void destroy();
    void add_edge(vidType u, int u_partition_idx, vidType v, int v_partition_idx);
    void reduce();
    vidType get_out_degree_0_to_1(vidType u);
    vidType get_out_degree_0_to_2(vidType u);
    vidType get_out_degree_1_to_2(vidType v);
    vidType N_0_to_1(vidType u, vidType n);
    vidType N_0_to_2(vidType u, vidType n);
    vidType N_1_to_2(vidType v, vidType n);
    int get_num_partitions();
    vidType *get_partition_num_vertices_tuple();
    vidType *get_edges_0_to_1();
    eidType get_num_edges_0_to_1();
    eidType *get_row_pointers_0_to_1();
    vidType *get_edges_0_to_2();
    eidType get_num_edges_0_to_2();
    eidType *get_row_pointers_0_to_2();
    vidType *get_edges_1_to_2();
    eidType get_num_edges_1_to_2();
    eidType *get_row_pointers_1_to_2();
};

// initialize the subgraph
void Type3SubGraph::init(Graph &g, int num_part, int this_type3_subgraph_index)
{
  // initialize its own logFile
  std::string file_name = "Type3SubGraph_log" + std::to_string(this_type3_subgraph_index) + ".txt";
  logFile.open(file_name);
  if (!logFile)
  {
    std::cerr << "Cannot open " << file_name << "\n";
    exit(-1);
  }
  // logFile << "logFile created!\nType3SubGraph " << this_type3_subgraph_index << " initialization starts ...\n";

  // initialize private variables
  this_type3_subgraph_idx = this_type3_subgraph_index;
  num_partitions = num_part;
  get_type3_subgraph_tuple(this_subgraph_tuple, this_type3_subgraph_idx, num_partitions);
  super_num_vertices = g.V();
  int normal_vertex_number_each_partition = (super_num_vertices - 1) / num_partitions + 1;
  partition_num_vertices_tuple[0] = (super_num_vertices % num_partitions > this_subgraph_tuple[0]) ? normal_vertex_number_each_partition + 1 : normal_vertex_number_each_partition;
  partition_num_vertices_tuple[1] = (super_num_vertices % num_partitions > this_subgraph_tuple[1]) ? normal_vertex_number_each_partition + 1 : normal_vertex_number_each_partition;
  partition_num_vertices_tuple[2] = (super_num_vertices % num_partitions > this_subgraph_tuple[2]) ? normal_vertex_number_each_partition + 1 : normal_vertex_number_each_partition;
  creation_finished = 0;
  // logFile << "\tPrivate variables initialized!\n";
  // logFile << "\t\tthis_type3_subgraph_idx = " << this_type3_subgraph_idx << "\n";
  // logFile << "\t\tnum_partitions = " << num_partitions << "\n";
  // logFile << "\t\tthis_subgraph_tuple = (" << this_subgraph_tuple[0] << ", " << this_subgraph_tuple[1] << ", " << this_subgraph_tuple[2] << ")\n";
  // logFile << "\t\tsuper_num_vertices = " << super_num_vertices << "\n";
  // logFile << "\t\tpartition_num_vertices_tuple = (" << partition_num_vertices_tuple[0] << ", " << partition_num_vertices_tuple[1] << ", " << partition_num_vertices_tuple[2] << ")\n";

  // allocate memory
  // logFile << "\tAllocate row_pointers ...\n";
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&row_pointers_0_to_1, (partition_num_vertices_tuple[0] + 1) * sizeof(eidType), cudaHostAllocDefault));
  for (eidType u = 0; u < partition_num_vertices_tuple[0] + 1; u++)
    row_pointers_0_to_1[u] = 0;
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&row_pointers_0_to_2, (partition_num_vertices_tuple[0] + 1) * sizeof(eidType), cudaHostAllocDefault));
  for (eidType u = 0; u < partition_num_vertices_tuple[0] + 1; u++)
    row_pointers_0_to_2[u] = 0;
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&row_pointers_1_to_2, (partition_num_vertices_tuple[1] + 1) * sizeof(eidType), cudaHostAllocDefault));
  for (eidType u = 0; u < partition_num_vertices_tuple[1] + 1; u++)
    row_pointers_1_to_2[u] = 0;
  
  // finish initialization and exit
  // logFile << "Initialization finished!\n";
}

// destroy the subgraph
void Type3SubGraph::destroy()
{
  // logFile << "Start destroying Type3SubGraph " << this_type3_subgraph_idx << " ...\n";

  // free the allocated memory
  // logFile << "\tFree the allocated memory ...\n";
  CUDA_SAFE_CALL(cudaFreeHost(edges_0_to_1));
  CUDA_SAFE_CALL(cudaFreeHost(row_pointers_0_to_1));
  CUDA_SAFE_CALL(cudaFreeHost(edges_0_to_2));
  CUDA_SAFE_CALL(cudaFreeHost(row_pointers_0_to_2));
  CUDA_SAFE_CALL(cudaFreeHost(edges_1_to_2));
  CUDA_SAFE_CALL(cudaFreeHost(row_pointers_1_to_2));
  // logFile << "\tAllocated memory freed!\n";

  // close the logFile
  // logFile << "Gracefully finishing ...\n";
  logFile.close();
}

// add (u, v) to the Type3SubGraph (u < v is guaranteed)
void Type3SubGraph::add_edge(vidType u, int u_partition_idx, vidType v, int v_partition_idx)
{
  directedEdge edge(u, v);

  // there are 3 cases if u < v is guaranteed
  if (u_partition_idx == this_subgraph_tuple[0] && v_partition_idx == this_subgraph_tuple[1])
  {
    temp_edges_0_to_1.push_back(edge);
    row_pointers_0_to_1[u / num_partitions]++;
  }
  else if (u_partition_idx == this_subgraph_tuple[0] && v_partition_idx == this_subgraph_tuple[2])
  {
    temp_edges_0_to_2.push_back(edge);
    row_pointers_0_to_2[u / num_partitions]++;
  }
  else
  {
    temp_edges_1_to_2.push_back(edge);
    row_pointers_1_to_2[u / num_partitions]++;
  }
}

// reorder and create the CSR format
void Type3SubGraph::reduce()
{
  // allocate memory for 'edges'
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&edges_0_to_1, temp_edges_0_to_1.size() * sizeof(vidType), cudaHostAllocDefault));
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&edges_0_to_2, temp_edges_0_to_2.size() * sizeof(vidType), cudaHostAllocDefault));
  CUDA_SAFE_CALL(cudaHostAlloc((void **)&edges_1_to_2, temp_edges_1_to_2.size() * sizeof(vidType), cudaHostAllocDefault));

  // perform scanning on row_pointers
  for (vidType counter_idx = 0; counter_idx < partition_num_vertices_tuple[0]; counter_idx++)
  {
    row_pointers_0_to_1[counter_idx + 1] += row_pointers_0_to_1[counter_idx];
    row_pointers_0_to_2[counter_idx + 1] += row_pointers_0_to_2[counter_idx];
  }
  for (vidType counter_idx = 0; counter_idx < partition_num_vertices_tuple[1]; counter_idx++)
    row_pointers_1_to_2[counter_idx + 1] += row_pointers_1_to_2[counter_idx];

  // pop the edges from back one by one and place it to the correct place
  while (temp_edges_0_to_1.size() > 0)
  {
    directedEdge edge = temp_edges_0_to_1[temp_edges_0_to_1.size() - 1];
    edges_0_to_1[--row_pointers_0_to_1[edge.first / num_partitions]] = edge.second;
    temp_edges_0_to_1.pop_back();
  }
  while (temp_edges_0_to_2.size() > 0)
  {
    directedEdge edge = temp_edges_0_to_2[temp_edges_0_to_2.size() - 1];
    edges_0_to_2[--row_pointers_0_to_2[edge.first / num_partitions]] = edge.second;
    temp_edges_0_to_2.pop_back();
  }
  while (temp_edges_1_to_2.size() > 0)
  {
    directedEdge edge = temp_edges_1_to_2[temp_edges_1_to_2.size() - 1];
    edges_1_to_2[--row_pointers_1_to_2[edge.first / num_partitions]] = edge.second;
    temp_edges_1_to_2.pop_back();
  }

  // pull up the 'creation_finished' flag
  creation_finished = 1;

  // sort and print the result into the logFile
  // logFile << "Type3SubGraph " << this_type3_subgraph_idx << " has finished reducing!\n";

  // logFile << "\t" << this_subgraph_tuple[0] << " to " << this_subgraph_tuple[1] << ":\n";
  for (vidType u = this_subgraph_tuple[0]; u < super_num_vertices; u += num_partitions)
  {
    std::sort(edges_0_to_1 + row_pointers_0_to_1[u / num_partitions], edges_0_to_1 + row_pointers_0_to_1[u / num_partitions + 1]);
    // logFile << "\t\t" << u << ": ";
    // vidType u_0_to_1_deg = get_out_degree_0_to_1(u);
    // for (vidType v_idx = 0; v_idx < u_0_to_1_deg; v_idx++)
    //   logFile << N_0_to_1(u, v_idx) << " ";
    // logFile << "\n";
  }

  // logFile << "\t" << this_subgraph_tuple[0] << " to " << this_subgraph_tuple[2] << ":\n";
  for (vidType u = this_subgraph_tuple[0]; u < super_num_vertices; u += num_partitions)
  {
    std::sort(edges_0_to_2 + row_pointers_0_to_2[u / num_partitions], edges_0_to_2 + row_pointers_0_to_2[u / num_partitions + 1]);
    // logFile << "\t\t" << u << ": ";
    // vidType u_0_to_2_deg = get_out_degree_0_to_2(u);
    // for (vidType w_idx = 0; w_idx < u_0_to_2_deg; w_idx++)
    //   logFile << N_0_to_2(u, w_idx) << " ";
    // logFile << "\n";
  }

  // logFile << "\t" << this_subgraph_tuple[1] << " to " << this_subgraph_tuple[2] << ":\n";
  for (vidType v = this_subgraph_tuple[1]; v < super_num_vertices; v += num_partitions)
  {
    std::sort(edges_1_to_2 + row_pointers_1_to_2[v / num_partitions], edges_1_to_2 + row_pointers_1_to_2[v / num_partitions + 1]);
    // logFile << "\t\t" << v << ": ";
    // vidType v_1_to_2_deg = get_out_degree_1_to_2(v);
    // for (vidType w_idx = 0; w_idx < v_1_to_2_deg; w_idx++)
    //   logFile << N_1_to_2(v, w_idx) << " ";
    // logFile << "\n";
  }
}

vidType Type3SubGraph::get_out_degree_0_to_1(vidType u)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'get_out_degree_0_to_1()' is called!\n";
    exit(-1);
  }
  return row_pointers_0_to_1[u / num_partitions + 1] - row_pointers_0_to_1[u / num_partitions];
}

vidType Type3SubGraph::get_out_degree_0_to_2(vidType u)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'get_out_degree_0_to_2()' is called!\n";
    exit(-1);
  }
  return row_pointers_0_to_2[u / num_partitions + 1] - row_pointers_0_to_2[u / num_partitions];
}

vidType Type3SubGraph::get_out_degree_1_to_2(vidType v)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'get_out_degree_1_to_2()' is called!\n";
    exit(-1);
  }
  return row_pointers_1_to_2[v / num_partitions + 1] - row_pointers_1_to_2[v / num_partitions];
}

vidType Type3SubGraph::N_0_to_1(vidType u, vidType n)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'N_0_to_1()' is called!\n";
    exit(-1);
  }
  return edges_0_to_1[row_pointers_0_to_1[u / num_partitions] + n];
}

vidType Type3SubGraph::N_0_to_2(vidType u, vidType n)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'N_0_to_2()' is called!\n";
    exit(-1);
  }
  return edges_0_to_2[row_pointers_0_to_2[u / num_partitions] + n];
}

vidType Type3SubGraph::N_1_to_2(vidType v, vidType n)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'N_1_to_2()' is called!\n";
    exit(-1);
  }
  return edges_1_to_2[row_pointers_1_to_2[v / num_partitions] + n];
}

int Type3SubGraph::get_num_partitions() { return num_partitions; }
vidType * Type3SubGraph::get_partition_num_vertices_tuple() { return partition_num_vertices_tuple; }
vidType * Type3SubGraph::get_edges_0_to_1() { return edges_0_to_1; }
eidType Type3SubGraph::get_num_edges_0_to_1() { return row_pointers_0_to_1[partition_num_vertices_tuple[0]]; }
eidType * Type3SubGraph::get_row_pointers_0_to_1() { return row_pointers_0_to_1; }
vidType * Type3SubGraph::get_edges_0_to_2() { return edges_0_to_2; }
eidType Type3SubGraph::get_num_edges_0_to_2() { return row_pointers_0_to_2[partition_num_vertices_tuple[0]]; }
eidType * Type3SubGraph::get_row_pointers_0_to_2() { return row_pointers_0_to_2; }
vidType * Type3SubGraph::get_edges_1_to_2() { return edges_1_to_2; }
eidType Type3SubGraph::get_num_edges_1_to_2() { return row_pointers_1_to_2[partition_num_vertices_tuple[1]]; }
eidType * Type3SubGraph::get_row_pointers_1_to_2() { return row_pointers_1_to_2; }

// given the device_count, calculate how many Type3SubGraphs
int get_type3_subgraph_num(int device_count)
{
  return (device_count * (device_count - 1) * (device_count - 2)) / 6;
}

// given the global index of Type3SubGraph, calculate its 3-tuple
void get_type3_subgraph_tuple(int *this_subgraph_tuple, int this_type3_subgraph_idx, int num_partitions)
{
  // check whether the requirement is legal, in the legal range
  int type3_subgraph_num = get_type3_subgraph_num(num_partitions);
  if (this_type3_subgraph_idx < 0 || this_type3_subgraph_idx >= type3_subgraph_num)
  {
    std::cout << "Error! 'this_type3_subgraph_idx' out of range in 'get_type3_subgraph_tuple()' call!\n";
    exit(-1);
  }

  // travers and attempt
  int first;
  int second;
  int third;
  for (first = 0; first < num_partitions - 2; first++)
  {
    int num_subgraphs_with_first = (num_partitions - first - 1) * (num_partitions - first - 2) / 2;
    if (num_subgraphs_with_first <= this_type3_subgraph_idx)
      this_type3_subgraph_idx -= num_subgraphs_with_first;
    else
    {
      this_subgraph_tuple[0] = first;
      for (second = first + 1; second < num_partitions - 1; second++)
      {
        int num_subgraphs_with_second = num_partitions - second - 1;
        if (num_subgraphs_with_second <= this_type3_subgraph_idx)
          this_type3_subgraph_idx -= num_subgraphs_with_second;
        else
        {
          this_subgraph_tuple[1] = second;
          third = second + 1 + this_type3_subgraph_idx;
          this_subgraph_tuple[2] = third;
          break;
        }
      }
      break;
    }
  }
}

// given the device_count and three seperate partition index, calculate the global index of the Type3SubGraph
int get_type3_subgraph_idx(int device_count, int u_device_idx, int v_device_idx, int w_device_idx)
{
  // sort the three partitions into increasing order
  sort_three_vertices(&u_device_idx, &v_device_idx, &w_device_idx);

  // accumulate index
  int type3_subgraph_idx = 0;
  for (vidType ui = 0; ui < u_device_idx; ui++)
    type3_subgraph_idx += (device_count - ui - 1) * (device_count - ui - 2) / 2;
  for (vidType vi = u_device_idx + 1; vi < v_device_idx; vi++)
    type3_subgraph_idx += device_count - vi - 1;
  type3_subgraph_idx += w_device_idx - v_device_idx - 1;
  return type3_subgraph_idx;
}

// sort the three vertices into increasing order
void sort_three_vertices(vidType *u, vidType *v, vidType *w)
{
  if (*u > *v)
  {
    vidType temp = *v;
    *v = *u;
    *u = temp;
  }
  if (*v > *w)
  {
    vidType temp = *w;
    *w = *v;
    *v = temp;
  }
  if (*u > *v)
  {
    vidType temp = *v;
    *v = *u;
    *u = temp;
  }
}

/**************************************** Definition of Type3SubGraph ***************************************/

class Type3SubGraphGPU {
  protected:
    int h_num_partitions;
    int *d_num_partitions;
    vidType *this_num_vertices;   // the array of the numbers of the three partitions
    vidType *d_this_num_vertices;

    vidType *h_edges_0_to_1;
    vidType *d_edges_0_to_1;
    eidType num_edges_0_to_1;
    eidType *h_row_pointers_0_to_1;
    eidType *d_row_pointers_0_to_1;

    vidType *h_edges_0_to_2;
    vidType *d_edges_0_to_2;
    eidType num_edges_0_to_2;
    eidType *h_row_pointers_0_to_2;
    eidType *d_row_pointers_0_to_2;

    vidType *h_edges_1_to_2;
    vidType *d_edges_1_to_2;
    eidType num_edges_1_to_2;
    eidType *h_row_pointers_1_to_2;
    eidType *d_row_pointers_1_to_2;

    AccType h_count;
    AccType *d_count;

  public:
    void init(Type3SubGraph &g);
    void launch(cudaStream_t stream);
    AccType get_count();
};

// initialize some of the protected variables
void Type3SubGraphGPU::init(Type3SubGraph &g)
{
  h_num_partitions = g.get_num_partitions();
  this_num_vertices = g.get_partition_num_vertices_tuple();

  h_edges_0_to_1 = g.get_edges_0_to_1();
  num_edges_0_to_1 = g.get_num_edges_0_to_1();
  h_row_pointers_0_to_1 = g.get_row_pointers_0_to_1();

  h_edges_0_to_2 = g.get_edges_0_to_2();
  num_edges_0_to_2 = g.get_num_edges_0_to_2();
  h_row_pointers_0_to_2 = g.get_row_pointers_0_to_2();

  h_edges_1_to_2 = g.get_edges_1_to_2();
  num_edges_1_to_2 = g.get_num_edges_1_to_2();
  h_row_pointers_1_to_2 = g.get_row_pointers_1_to_2();

  h_count = 0;
}

// launch the kernel function
void Type3SubGraphGPU::launch(cudaStream_t stream)
{
  // allocate device memory
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_num_partitions, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_num_partitions, &h_num_partitions, sizeof(int), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_this_num_vertices, 3 * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_this_num_vertices, this_num_vertices, 3 * sizeof(vidType), cudaMemcpyHostToDevice, stream));

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges_0_to_1, num_edges_0_to_1 * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_edges_0_to_1, h_edges_0_to_1, num_edges_0_to_1 * sizeof(vidType), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_pointers_0_to_1, (this_num_vertices[0] + 1) * sizeof(eidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_row_pointers_0_to_1, h_row_pointers_0_to_1, (this_num_vertices[0] + 1) * sizeof(eidType), cudaMemcpyHostToDevice, stream));

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges_0_to_2, num_edges_0_to_2 * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_edges_0_to_2, h_edges_0_to_2, num_edges_0_to_2 * sizeof(vidType), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_pointers_0_to_2, (this_num_vertices[0] + 1) * sizeof(eidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_row_pointers_0_to_2, h_row_pointers_0_to_2, (this_num_vertices[0] + 1) * sizeof(eidType), cudaMemcpyHostToDevice, stream));

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_edges_1_to_2, num_edges_1_to_2 * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_edges_1_to_2, h_edges_1_to_2, num_edges_1_to_2 * sizeof(vidType), cudaMemcpyHostToDevice, stream));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_pointers_1_to_2, (this_num_vertices[1] + 1) * sizeof(eidType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_row_pointers_1_to_2, h_row_pointers_1_to_2, (this_num_vertices[1] + 1) * sizeof(eidType), cudaMemcpyHostToDevice, stream));

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_count, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpyAsync(d_count, &h_count, sizeof(AccType), cudaMemcpyHostToDevice, stream));

  // launch kernel
  size_t nthreads = BLOCK_SIZE;
  size_t nblocks = (this_num_vertices[0] - 1) / WARPS_PER_BLOCK + 1;
  if (nblocks > 65536)
  {
    std::cout << "nblocks larger than 65536, nblocks = " << nblocks << "\n";
    nblocks = 65536;
  }

  warp_vertex_type3<<<nblocks, nthreads, 0, stream>>>(d_num_partitions, d_this_num_vertices, d_count, d_edges_0_to_1, d_row_pointers_0_to_1, d_edges_0_to_2, d_row_pointers_0_to_2, d_edges_1_to_2, d_row_pointers_1_to_2);

  // copy the result out and free the allocated device memory
  CUDA_SAFE_CALL(cudaMemcpyAsync(&h_count, d_count, sizeof(AccType), cudaMemcpyDeviceToHost, stream));
  CUDA_SAFE_CALL(cudaFree(d_num_partitions));
  CUDA_SAFE_CALL(cudaFree(d_this_num_vertices));
  CUDA_SAFE_CALL(cudaFree(d_edges_0_to_1));
  CUDA_SAFE_CALL(cudaFree(d_row_pointers_0_to_1));
  CUDA_SAFE_CALL(cudaFree(d_edges_0_to_2));
  CUDA_SAFE_CALL(cudaFree(d_row_pointers_0_to_2));
  CUDA_SAFE_CALL(cudaFree(d_edges_1_to_2));
  CUDA_SAFE_CALL(cudaFree(d_row_pointers_1_to_2));
  CUDA_SAFE_CALL(cudaFree(d_count));
}

// get the answer of the subgraph
AccType Type3SubGraphGPU::get_count() { return h_count; }

/**************************************** Definition of tool functions ***************************************/

__global__
void warp_vertex_type1(AccType *d_count, vidType *d_edges, eidType *d_row_pointers, vidType *d_num_vertices)
{
  // allocate a space for map-reduce
  __shared__ AccType partial_sum[BLOCK_SIZE];

  // calculate the global index of the warp
  int global_warp_idx = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
  int total_warp_num = gridDim.x * blockDim.x / WARP_SIZE;
  
  // implement calculation of the vertecies
  AccType cnt = 0;
  for (vidType u = global_warp_idx; u < *d_num_vertices; u += total_warp_num)
  {
    for (eidType v_idx_in_edges = d_row_pointers[u]; v_idx_in_edges < d_row_pointers[u + 1]; v_idx_in_edges++)
    {
      vidType v = d_edges[v_idx_in_edges];
      cnt += intersect_num(d_edges + d_row_pointers[u], (vidType)(d_row_pointers[u + 1] - d_row_pointers[u]), d_edges + d_row_pointers[v], (vidType)(d_row_pointers[v + 1] - d_row_pointers[v]));
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
      atomicAdd(d_count, partial_sum[0]);
}

__global__
void warp_vertex_type3(vidType *d_num_partitions, vidType *d_this_num_vertices, AccType *d_count, vidType *d_edges_0_to_1, eidType *d_row_pointers_0_to_1, vidType *d_edges_0_to_2, eidType *d_row_pointers_0_to_2, vidType *d_edges_1_to_2, eidType *d_row_pointers_1_to_2)
{
  // allocate a space for map-reduce
  __shared__ AccType partial_sum[BLOCK_SIZE];

  // calculate the global index of the warp
  int global_warp_idx = (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
  int total_warp_num = gridDim.x * blockDim.x / WARP_SIZE;
  
  // implement calculation of the vertecies
  AccType cnt = 0;
  for (vidType u_idx = global_warp_idx; u_idx < d_this_num_vertices[0]; u_idx += total_warp_num)
  {
    for (eidType v_idx_in_0_array = d_row_pointers_0_to_1[u_idx]; v_idx_in_0_array < d_row_pointers_0_to_1[u_idx + 1]; v_idx_in_0_array++)
    {
      vidType v = d_edges_0_to_1[v_idx_in_0_array];
      vidType v_idx = v / *d_num_partitions;
      cnt += intersect_num(d_edges_0_to_2 + d_row_pointers_0_to_2[u_idx], (vidType)(d_row_pointers_0_to_2[u_idx + 1] - d_row_pointers_0_to_2[u_idx]), d_edges_1_to_2 + d_row_pointers_1_to_2[v_idx], (vidType)(d_row_pointers_1_to_2[v_idx + 1] - d_row_pointers_1_to_2[v_idx]));
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
      atomicAdd(d_count, partial_sum[0]);
}

/**************************************** Definition of TCSolver ***************************************/

void TCSolver(Graph &g, uint64_t &total, int n_gpus, int chunk_size) {
  // start TCSolver
  std::string file_name = "log.txt";
  std::ofstream logFile(file_name);
  if (!logFile)
  {
    std::cerr << "Cannot open " << file_name << "\n";
    exit(-1);
  }
  // logFile << "TCSolver starts ...\n";

  // read important information out from the super graph
  // logFile << "Reading input graph ...\n";
  vidType super_graph_vertex_num = g.V();
  // logFile << "|V| = " << super_graph_vertex_num << "\n";

  // get the device_count of the system
  // logFile << "Looking for devices ...\n";
  int device_count;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
  // logFile << "\t" << device_count << " devices available!\n";
  if (n_gpus < device_count)
  {
    device_count = n_gpus;
    // logFile << "\t" << "We only use " << device_count << " gpus.\n";
  }

  // creating the subgraphs
  // logFile << "Creating the subgraphs ...\n";
  std::vector<Type1SubGraph> type1_subgraphs(device_count);
  
  for (int device_idx = 0; device_idx < device_count; device_idx++)
  {
    // logFile << "\tType1SubGraph " << device_idx << " starts ...\n";
    type1_subgraphs[device_idx].init(g, device_idx);
  }
  // logFile << "\tAll the Type1SubGraphs created!\n";
  int type3_subgraph_num = get_type3_subgraph_num(device_count);
  std::vector<Type3SubGraph> type3_subgraphs(device_count);
  for (int type3_subgraph_idx = 0; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx++)
    type3_subgraphs[type3_subgraph_idx].init(g, device_count, type3_subgraph_idx);
  // logFile << "\tAll the Type3SubGraphs created!\n";

  // map
  // logFile << "Start mapping ...\n";
  for (vidType u = 0; u < super_graph_vertex_num; u++)
  {
    int u_partition_idx = u % device_count;
    vidType u_deg = g.out_degree(u);
    for (vidType v_idx = 0; v_idx < u_deg; v_idx++)
    {
      vidType v = g.N(u, v_idx);
      // logFile << "(" << u << ", " << v << ")\n";
      int v_partition_idx = v % device_count;
      if (u_partition_idx == v_partition_idx)   // inner edge
        type1_subgraphs[u_partition_idx].add_edge(u, v);
      else
      {
        type1_subgraphs[u_partition_idx].add_edge(u, v);
        type1_subgraphs[v_partition_idx].add_edge(u, v);

        for (int w_partition_idx = 0; w_partition_idx < device_count; w_partition_idx++)
        {
          if (w_partition_idx != u_partition_idx && w_partition_idx != v_partition_idx)
          {
            int type3_subgraph_idx = get_type3_subgraph_idx(device_count, u_partition_idx, v_partition_idx, w_partition_idx);
            if (u_partition_idx < v_partition_idx)
              type3_subgraphs[type3_subgraph_idx].add_edge(u, u_partition_idx, v, v_partition_idx);
            else
              type3_subgraphs[type3_subgraph_idx].add_edge(v, v_partition_idx, u, u_partition_idx);
          }
        }
      }
    }
  }
  // logFile << "Finish mapping!\n";
  for (int device_idx = 0; device_idx < device_count; device_idx++)
  {
    type1_subgraphs[device_idx].end_adding_edge();
    // type1_subgraphs[device_idx].print_edge_info();
  }

  // reduce
  // logFile << "Start reducing ...\n";
  for (int type3_subgraph_idx = 0; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx++)
    type3_subgraphs[type3_subgraph_idx].reduce();
  // logFile << "Finish reducing ...\n";

  // initialize classes on GPU
  // logFile << "Start initializing classes ...\n";
  std::vector<Type1SubGraphGPU> type1_subgraphs_on_gpu(device_count);
  std::vector<Type3SubGraphGPU> type3_subgraphs_on_gpu(type3_subgraph_num);
  std::vector<std::thread> init_threads;
  for (int device_idx = 0; device_idx < device_count; device_idx++)
  {
    init_threads.push_back(std::thread([&, device_idx]() {
      // Type1SubGraphGPU
      type1_subgraphs_on_gpu[device_idx].init(type1_subgraphs[device_idx]);
      std::cout << "\tType1SubGraphGPU " << device_idx << " finishes!\n";

      // Type3SubGraphGPU
      for (int type3_subgraph_idx = device_idx; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx += device_count)
      {
        type3_subgraphs_on_gpu[type3_subgraph_idx].init(type3_subgraphs[type3_subgraph_idx]);
        std::cout << "\tType3SubGraphGPU " << type3_subgraph_idx << " finishes!\n";
      }
    }));
  }
  for (auto &thread: init_threads) thread.join();
  // logFile << "Finish initializing Type1SubGraphGPU!\n";

  // launch
  // logFile << "Start launching kernel functions ...\n";
  Timer t;
  std::vector<Timer> subt(device_count);
  std::vector<cudaStream_t> streams_for_type1(device_count);
  std::vector<cudaStream_t> streams_for_type3(type3_subgraph_num);
  t.Start();
  std::vector<std::thread> gpu_threads;
  for (int device_idx = 0; device_idx < device_count; device_idx++)
  {
    gpu_threads.push_back(std::thread([&, device_idx]() {
      // set the device in each thread

      subt[device_idx].Start();
      CUDA_SAFE_CALL(cudaSetDevice(device_idx));
      CUDA_SAFE_CALL(cudaDeviceSynchronize());

      // launch Type1SubGraphGPU
      CUDA_SAFE_CALL(cudaStreamCreate(&streams_for_type1[device_idx]));
      type1_subgraphs_on_gpu[device_idx].launch(streams_for_type1[device_idx]);

      // launch Type3SubGraphGPU
      for (int type3_subgraph_idx = device_idx; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx += device_count)
      {
        CUDA_SAFE_CALL(cudaStreamCreate(&streams_for_type3[type3_subgraph_idx]));
        type3_subgraphs_on_gpu[type3_subgraph_idx].launch(streams_for_type3[type3_subgraph_idx]);
      }

      CUDA_SAFE_CALL(cudaDeviceSynchronize());
      subt[device_idx].Stop();
    }));
  }
  for (auto &thread: gpu_threads) thread.join();
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  // logFile << "Kernel functions finished!\n";

  // gather results
  // logFile << "Gathering results ...\n";
  for (int device_idx = 0; device_idx < device_count; device_idx++)
  {
    AccType each_count = type1_subgraphs_on_gpu[device_idx].get_count();
    // logFile << "\tNumber of triangles in Type1SubGraph " << device_idx << " :" << each_count << "\n";
    total += each_count;
  }
  for (int type3_subgraph_idx = 0; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx++)
  {
    AccType each_count = type3_subgraphs_on_gpu[type3_subgraph_idx].get_count();
    // logFile << "\tNumber of triangles in Type3SubGraph " << type3_subgraph_idx << " :" << each_count << "\n";
    total += each_count;
  }

  // display time
  for (int device_idx = 0; device_idx < device_count; device_idx++)
    std::cout << "runtine[gpu" << device_idx << "] = " << subt[device_idx].Millisecs() << " msec\n";
  std::cout << "runtime = " << t.Seconds() << " sec\n";

  // end and exit
  // logFile << "Destoying Type1SubGraphs ...\n";
  for (int device_idx = 0; device_idx < device_count; device_idx++)
    type1_subgraphs[device_idx].destroy();
  // logFile << "All the Type1SubGraphs have been destoyed!\n";
  // logFile << "Destoying Type3SubGraphs ...\n";
  for (int type3_subgraph_idx = 0; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx++)
    type3_subgraphs[type3_subgraph_idx].destroy();
  // logFile << "All the Type3SubGraphs have been destoyed!\n";
  // logFile << "Gracefully finishing ...\n";
  logFile.close();
}