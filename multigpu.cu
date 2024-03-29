// Copyright (c) 2020 MIT
// Author: Xuhao Chen
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

/**************************************** Definition of SubGraphs ***************************************/

typedef std::pair<vidType, vidType> directedEdge;

class Type1SubGraph {
  protected:
    vidType super_num_vertices;           // how many vertices in the super graph
    int num_partitions;                   // the super graph is partitioned into how many partitions
    int this_partition_idx;               // the index of the partition corresponding to this subgraph
    vidType start_vertex_idx;             // the start index of the vertices in this partition, in the global view
    vidType this_num_vertices;            // how many vertices in this partition
    std::ofstream logFile;                // its own logFile

    vidType *edges;                       // column indices of CSR format, starting point only inner
    eidType *row_pointers;                // row pointers of CSR format, starting point only inner

    std::vector<directedEdge> temp_edges; // for the process of generating, allowing start from outer
    vidType *edge_counters;               // counts how many edges are from a particular vertex
    eidType this_num_edges;               // counts how many edges in this subgraph

  public:
    void init(Graph &g, int num_partitions, int this_partition_idx);
    void destroy();
    void add_edge(vidType from, vidType to);
};

// initialize the subgraph
void Type1SubGraph::init(Graph &g, int num_partitions, int this_partition_idx) :
    super_num_vertices(g.V()), num_partitions(num_partitions), this_partition_idx(this_partition_idx), this_num_edges(0)
{
  // initialize its own logFile
  std::string file_name = "Type1SubGraph_log" + std::to_string(this_partition_idx) + ".txt";
  logFile.open(file_name);
  if (!logFile)
  {
    std::cerr << "Cannot open " << file_name << "\n";
    exit(-1);
  }
  logFile << "logFile created!\nType1SubGraph " << this_partition_idx << " initialization starts ...\n";

  // initialize private variables
  int normal_vertex_number_each_partition = (super_num_vertices - 1) / num_partitions + 1;
  start_vertex_idx = normal_vertex_number_each_partition * this_partition_idx;
  this_num_vertices = (this_partition_idx == num_partitions - 1) ? (super_num_vertices - normal_vertex_number_each_partition * (num_partitions - 1)) : normal_vertex_number_each_partition;
  logFile << "\tPrivate variables initialized!\n";
  logFile << "\t\tsuper_num_vertices = " << super_num_vertices << "\n";
  logFile << "\t\tnum_partitions = " << num_partitions << "\n";
  logFile << "\t\tthis_partition_idx = " << this_partition_idx << "\n";
  logFile << "\t\tstart_vertex_idx = " << start_vertex_idx << "\n";
  logFile << "\t\tthis_num_vertices = " << this_num_vertices << "\n";

  // initialize the CSR
  logFile << "\tAllocate row_pointers ...\n";
  edges = NULL;   // no edges now
  row_pointers = (eidType *)malloc(this_num_vertices * sizeof(eidType));
  edge_counters = (vidType *)malloc(this_num_vertices * sizeof(vidType));
  for (vidType u = 0; u < this_num_vertices; u++) // clear the counters
    edge_counters[u] = 0;

  // finish initialization and exit
  logFile << "Initialization succeeded!\n";
}

// destroy the subgraph
void Type1SubGraph::destroy()
{
  logFile << "Start destroying Type1SubGraph " << this_partition_idx << " ...\n";

  // free the "edges_from_outside" and "edges_to_outside"
  logFile << "\tFree the allocated memory ...\n";
  free(row_pointers);
  logFile << "\tAllocated memory freed!\n";

  // close the logFile
  logFile << "Gracefully finishing ...\n";
  logFile.close();
}

// push a new edge to the temp_edges
void Type1SubGraph::add_edge(vidType from, vidType to)
{
  // refresh the counter of 'from'
  edge_counters[from]++;
  this_num_edges++;

  // create a pair
  directedEdge edge(from, to);
  temp_edges.push_back(edge);
}

// reorder the temp_edges and create the final CSR format
void Type1SubGraph::reduce()
{
  
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
  logFile << "TCSolver starts ...\n";

  // read important information out from the super graph
  logFile << "Reading input graph ...\n";
  vidType super_graph_vertex_num = g.V();
  logFile << "|V| = " << super_graph_vertex_num << "\n";

  // get the device_count of the system
  logFile << "Looking for devices ...\n";
  int device_count;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
  logFile << "\t" << device_count << " devices available!\n";

  // creating the subgraphs
  logFile << "Creating the subgraphs ...\n";
  std::vector<Type1SubGraph> type1_subgraphs(device_count);
  for (int device_idx = 0; device_idx < device_count; device_idx++)
    type1_subgraphs[device_idx].init(g, device_count, device_idx);
  logFile << "All the Type1SubGraph created!\n";

  // map
  logFile << "Start mapping ...\n";
  int normal_vertex_number_each_partition = (super_graph_vertex_num - 1) / device_count + 1;
  for (int device_idx = 0; device_idx < device_count; device_idx++)
  {
    logFile << "\tMapping part " << device_idx << " ...\n";
    vidType partition_start_vertex = normal_vertex_number_each_partition * device_idx;
    vidType partition_end_vertex = (partition_start_vertex + normal_vertex_number_each_partition > super_graph_vertex_num) ? super_graph_vertex_num : partition_start_vertex + normal_vertex_number_each_partition;
    for (vidType u = partition_start_vertex; u < partition_end_vertex; u++)
    {
      vidType u_deg = g.out_degree(u);
      for (vidType v_idx = 0; v_idx < u_deg; v_idx++)
      {
        vidType v = g.N(u, v_idx);
        int v_partition_idx = v % normal_vertex_number_each_partition;
        if (device_idx == v_partition_idx)  // inner edge
          type1_subgraphs[device_idx].add_edge(u, v);
        else  // cross edge
        {
          type1_subgraphs[device_idx].add_edge(u, v);
          type1_subgraphs[v_partition_idx].add_edge(v, u);
        }
      }
    }
  }
  logFile << "Finish mapping!\n";

  // reduce
  logFile << "Start reducing ...\n";
  for (int device_idx = 0; device_idx < device_count; device_idx++)
    type1_subgraphs[device_idx].reduce();
  logFile << "Finish reducing ...\n";

  // end and exit
  logFile << "Destoying Type1SubGraphs ...\n";
  for (int device_idx = 0; device_idx < device_count; device_idx++)
    type1_subgraphs[device_idx].destroy();
  logFile << "All the Type1SubGraphs have been destoyed!\n";
  logFile << "Gracefully finishing ...\n";
  logFile.close();
}

