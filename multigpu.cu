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

typedef std::pair<vidType, vidType> directedEdge;
typedef std::tuple<int, int, int> type3Tuple;

int get_type3_subgraph_num(int device_count);
type3Tuple get_type3_subgraph_tuple(int this_type3_subgraph_idx, int num_partitions);
int get_type3_subgraph_idx(int device_count, vidType u, vidType v, vidType w);
void sort_three_vertices(vidType *u, vidType *v, vidType *w);

/**************************************** Definition of Type1SubGraph ***************************************/

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
    int creation_finished;                // whether the subgraph has finished creating

  public:
    void init(Graph &g, int num_partitions, int this_partition_idx);
    void destroy();
    void add_edge(vidType from, vidType to);
    void reduce();
    vidType get_out_degree(vidType u);
    vidType N(vidType u, vidType n);
};

// initialize the subgraph
void Type1SubGraph::init(Graph &g, int num_part, int this_part_idx)
{
  // initialize its own logFile
  std::string file_name = "Type1SubGraph_log" + std::to_string(this_part_idx) + ".txt";
  logFile.open(file_name);
  if (!logFile)
  {
    std::cerr << "Cannot open " << file_name << "\n";
    exit(-1);
  }
  logFile << "logFile created!\nType1SubGraph " << this_part_idx << " initialization starts ...\n";

  // initialize private variables
  super_num_vertices = g.V();
  num_partitions = num_part;
  this_partition_idx = this_part_idx;
  vidType normal_vertex_number_each_partition = (super_num_vertices - 1) / num_partitions + 1;
  start_vertex_idx = normal_vertex_number_each_partition * this_partition_idx;
  this_num_vertices = (this_partition_idx == num_partitions - 1) ? (super_num_vertices - normal_vertex_number_each_partition * (num_partitions - 1)) : normal_vertex_number_each_partition;
  creation_finished = 0;
  logFile << "\tPrivate variables initialized!\n";
  logFile << "\t\tsuper_num_vertices = " << super_num_vertices << "\n";
  logFile << "\t\tnum_partitions = " << num_partitions << "\n";
  logFile << "\t\tthis_partition_idx = " << this_partition_idx << "\n";
  logFile << "\t\tstart_vertex_idx = " << start_vertex_idx << "\n";
  logFile << "\t\tthis_num_vertices = " << this_num_vertices << "\n";

  // initialize the CSR
  logFile << "\tAllocate row_pointers ...\n";
  edges = NULL;   // no edges now
  row_pointers = (eidType *)malloc((this_num_vertices + 1) * sizeof(eidType));
  for (eidType u = 0; u < this_num_vertices + 1; u++)
    row_pointers[u] = 0;

  // finish initialization and exit
  logFile << "Initialization succeeded!\n";
}

// destroy the subgraph
void Type1SubGraph::destroy()
{
  logFile << "Start destroying Type1SubGraph " << this_partition_idx << " ...\n";

  // free the "edges_from_outside" and "edges_to_outside"
  logFile << "\tFree the allocated memory ...\n";
  free(edges);
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
  row_pointers[from - start_vertex_idx]++;

  // create a pair
  directedEdge edge(from, to);
  temp_edges.push_back(edge);
}

// reorder the temp_edges and create the final CSR format
void Type1SubGraph::reduce()
{
  // allocate memory for 'edges'
  edges = (vidType *)malloc(temp_edges.size() * sizeof(vidType));

  // perform scanning on 'row_pointers', thus getting the indices
  for (vidType counter_idx = 0; counter_idx < this_num_vertices; counter_idx++)
    row_pointers[counter_idx + 1] += row_pointers[counter_idx];

  // pop the edges from back one by one and place it to the correct place
  while (temp_edges.size() > 0)
  {
    // pop the last pair out
    directedEdge edge = temp_edges[temp_edges.size() - 1];
    temp_edges.pop_back();

    // put it into the right place in 'edges'
    edges[--row_pointers[edge.first - start_vertex_idx]] = edge.second;
  }

  // pull up the 'creation_finished' flag
  creation_finished = 1;

  // sort and print the result into the logFile
  logFile << "Type1SubGraph " << this_partition_idx << " has finished reducing!\n";
  for (vidType u = start_vertex_idx; u < start_vertex_idx + this_num_vertices; u++)
  {
    std::sort(edges + row_pointers[u - start_vertex_idx], edges + row_pointers[u + 1 - start_vertex_idx]);
    logFile << "\t" << u << ": ";
    vidType u_deg = get_out_degree(u);
    for (vidType v_idx = 0; v_idx < u_deg; v_idx++)
      logFile << N(u, v_idx) << " ";
    logFile << "\n";
  }
}

// get the out-degree of vertex u
vidType Type1SubGraph::get_out_degree(vidType u)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'get_out_degree()' is called!\n";
    exit(-1);
  }
  return row_pointers[u - start_vertex_idx + 1] - row_pointers[u - start_vertex_idx];
}

// get the n-th neighbor of u
vidType Type1SubGraph::N(vidType u, vidType n)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'N()' is called!\n";
    exit(-1);
  }
  return edges[row_pointers[u - start_vertex_idx] + n];
}

/**************************************** Definition of Type3SubGraph ***************************************/

class Type3SubGraph {
  protected:
    int this_type3_subgraph_idx;                                            // the global index among all the Type3SubGraphs
    int num_partitions;                                                     // how many partitions in the super graph
    type3Tuple this_subgraph_tuple;                                         // records the three indices of the partitions, sorted
    vidType super_num_vertices;                                             // how many vertices in the super graph
    std::tuple<vidType, vidType, vidType> partition_num_vertices_tuple;     // how many vertices are there in each partition
    std::tuple<vidType, vidType, vidType> partition_start_vertex_idx_tuple; // each partition starts at which vertex
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
    void recude();
    vidType get_out_degree_0_to_1(vidType u);
    vidType get_out_degree_0_to_2(vidType u);
    vidType get_out_degree_1_to_2(vidType v);
    vidType N_0_to_1(vidType u, vidType n);
    vidType N_0_to_2(vidType u, vidType n);
    vidType N_1_to_2(vidType v, vidType n);
}

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
  logFile << "logFile created!\nType3SubGraph " << this_type3_subgraph_index << " initialization starts ...\n";

  // initialize private variables
  this_type3_subgraph_idx = this_type3_subgraph_index;
  num_partitions = num_part;
  this_subgraph_tuple = get_type3_subgraph_tuple(this_type3_subgraph_idx, num_partitions);
  super_num_vertices = g.V();
  vidType normal_vertex_number_each_partition = (super_num_vertices - 1) / num_partitions + 1;
  std::get<0>(partition_num_vertices_tuple) = normal_vertex_number_each_partition;
  std::get<1>(partition_num_vertices_tuple) = normal_vertex_number_each_partition;
  std::get<2>(partition_num_vertices_tuple) = (std::get<2>(this_subgraph_tuple) == num_partitions - 1) ? (super_num_vertices - normal_vertex_number_each_partition * (num_partitions - 1)) : normal_vertex_number_each_partition;
  std::get<0>(partition_start_vertex_idx_tuple) = normal_vertex_number_each_partition * std::get<0>(this_subgraph_tuple);
  std::get<1>(partition_start_vertex_idx_tuple) = normal_vertex_number_each_partition * std::get<1>(this_subgraph_tuple);
  std::get<2>(partition_start_vertex_idx_tuple) = normal_vertex_number_each_partition * std::get<2>(this_subgraph_tuple);
  creation_finished = 0;
  logFile << "\tPrivate variables initialized!\n";
  logFile << "\t\tthis_type3_subgraph_idx = " << this_type3_subgraph_idx << "\n";
  logFile << "\t\tnum_partitions = " << num_partitions << "\n";
  logFile << "\t\tthis_subgraph_tuple = (" << std::get<0>(this_subgraph_tuple) << ", " << std::get<1>(this_subgraph_tuple) << ", " << std::get<2>(this_subgraph_tuple) << ")\n";
  logFile << "\t\tsuper_num_vertices = " << super_num_vertices << "\n";
  logFile << "\t\tpartition_num_vertices_tuple = (" << std::get<0>(partition_num_vertices_tuple) << ", " << std::get<1>(partition_num_vertices_tuple) << ", " << std::get<2>(partition_num_vertices_tuple) << ")\n";
  logFile << "\t\tpartition_start_vertex_idx_tuple = (" << std::get<0>(partition_start_vertex_idx_tuple) << ", " << std::get<1>(partition_start_vertex_idx_tuple) << ", " << std::get<2>(partition_start_vertex_idx_tuple) << ")\n";

  // allocate memory
  logFile << "\tAllocate row_pointers ...\n"
  row_pointers_0_to_1 = (eidType *)malloc((std::get<0>(partition_num_vertices_tuple) + 1) * sizeof(eidType));
  for (eidType u = 0; u < std::get<0>(partition_num_vertices_tuple) + 1; u++)
    row_pointers_0_to_1[u] = 0;
  row_pointers_0_to_2 = (eidType *)malloc((std::get<0>(partition_num_vertices_tuple) + 1) * sizeof(eidType));
  for (eidType u = 0; u < std::get<0>(partition_num_vertices_tuple) + 1; u++)
    row_pointers_0_to_2[u] = 0;
  row_pointers_1_to_2 = (eidType *)malloc((std::get<1>(partition_num_vertices_tuple) + 1) * sizeof(eidType));
  for (eidType u = 0; u < std::get<1>(partition_num_vertices_tuple) + 1; u++)
    row_pointers_1_to_2[u] = 0;
  
  // finish initialization and exit
  logFile << "Initialization finished!\n";
}

// destroy the subgraph
void Type3SubGraph::destroy()
{
  logFile << "Start destroying Type3SubGraph " << this_type3_subgraph_idx << " ...\n";

  // free the allocated memory
  logFile << "\tFree the allocated memory ...\n";
  free(edges_0_to_1);
  free(row_pointers_0_to_1);
  free(edges_0_to_2);
  free(row_pointers_0_to_2);
  free(edges_1_to_2);
  free(row_pointers_1_to_2);
  logFile << "\tAllocated memory freed!\n";

  // close the logFile
  logFile << "Gracefully finishing ...\n";
  logFile.close();
}

// add (u, v) to the Type3SubGraph (u < v is guaranteed)
void Type3SubGraph::add_edge(vidType u, int u_partition_idx, vidType v, int v_partition_idx)
{
  directedEdge edge(u, v);
  // there are 3 cases if u < v is guaranteed
  if (u_partition_idx == std::get<0>(this_subgraph_tuple) && v_partition_idx == std::get<1>(this_subgraph_tuple))
  {
    temp_edges_0_to_1.push_back(edge);
    row_pointers_0_to_1[u - std::get<0>(partition_start_vertex_idx_tuple)]++;
  }
  else if (u_partition_idx == std::get<0>(this_subgraph_tuple) && v_partition_idx == std::get<2>(this_subgraph_tuple))
  {
    temp_edges_0_to_2.push_back(edge);
    row_pointers_0_to_2[u - std::get<0>(partition_start_vertex_idx_tuple)]++;
  }
  else
  {
    temp_edges_1_to_2.push_back(edge);
    row_pointers_1_to_2[u - std::get<1>(partition_start_vertex_idx_tuple)]++;
  }
}

// reorder and create the CSR format
void Type3SubGraph::recude()
{
  // allocate memory for 'edges'
  edges_0_to_1 = (vidType *)malloc(temp_edges_0_to_1.size() * sizeof(vidType));
  edges_0_to_2 = (vidType *)malloc(temp_edges_0_to_2.size() * sizeof(vidType));
  edges_1_to_2 = (vidType *)malloc(temp_edges_1_to_2.size() * sizeof(vidType));

  // perform scanning on row_pointers
  for (vidType counter_idx = 0; counter_idx < std::get<0>(partition_num_vertices_tuple); counter_idx++)
  {
    row_pointers_0_to_1[counter_idx + 1] += row_pointers_0_to_1[counter_idx];
    row_pointers_0_to_2[counter_idx + 1] += row_pointers_0_to_2[counter_idx];
  }
  for (vidType counter_idx = 0; counter_idx < std::get<1>(partition_num_vertices_tuple); counter_idx++)
    row_pointers_1_to_2[counter_idx + 1] += row_pointers_1_to_2[counter_idx];

  // pop the edges from back one by one and place it to the correct place
  vidType start_idx = std::get<0>(partition_start_vertex_idx_tuple);
  while (temp_edges_0_to_1.size() > 0)
  {
    directedEdge edge = temp_edges_0_to_1[temp_edges_0_to_1.size() - 1];
    edges_0_to_1[--row_pointers_0_to_1[edge.first - start_idx]] = edge.second;
    temp_edges_0_to_1.pop_back();
  }
  while (temp_edges_0_to_2.size() > 0)
  {
    directedEdge edge = temp_edges_0_to_2[temp_edges_0_to_2.size() - 1];
    edges_0_to_2[--row_pointers_0_to_2[edge.first - start_idx]] = edge.second;
    temp_edges_0_to_2.pop_back();
  }
  start_idx = std::get<1>(partition_start_vertex_idx_tuple);
  while (temp_edges_1_to_2.size() > 0)
  {
    directedEdge edge = temp_edges_1_to_2[temp_edges_1_to_2.size() - 1];
    edges_1_to_2[--row_pointers_1_to_2[edge.first - start_idx]] = edge.second;
    temp_edges_1_to_2.pop_back();
  }

  // pull up the 'creation_finished' flag
  creation_finished = 1;

  // sort and print the result into the logFile
  logFile << "Type3SubGraph " << this_type3_subgraph_idx << " has finished reducing!\n";

  logFile << "\t" << std::get<0>(this_subgraph_tuple) << " to " << std::get<1>(this_subgraph_tuple) << ":\n";
  vidType partition_start_vertex_idx_0 = std::get<0>(partition_start_vertex_idx_tuple);
  vidType partition_num_vertices_0 = std::get<0>(partition_num_vertices_tuple);
  for (vidType u = partition_start_vertex_idx_0; u < partition_start_vertex_idx_0 + partition_num_vertices_0; u++)
  {
    std::sort(edges_0_to_1 + row_pointers_0_to_1[u - partition_start_vertex_idx_0], edges_0_to_1 + row_pointers_0_to_1[u + 1 - partition_start_vertex_idx_0]);
    logFile << "\t\t" << u << ": ";
    vidType u_0_to_1_deg = get_out_degree_0_to_1(u);
    for (vidType v_idx = 0; v_idx < u_0_to_1_deg; v_idx++)
      logFile << N_0_to_1(u, v_idx) << " ";
    logFile << "\n";
  }

  logFile << "\t" << std::get<0>(this_subgraph_tuple) << " to " << std::get<2>(this_subgraph_tuple) << ":\n";
  for (vidType u = partition_start_vertex_idx_0; u < partition_start_vertex_idx_0 + partition_num_vertices_0; u++)
  {
    std::sort(edges_0_to_2 + row_pointers_0_to_2[u - partition_start_vertex_idx_0], edges_0_to_2 + row_pointers_0_to_2[u + 1 - partition_start_vertex_idx_0]);
    logFile << "\t\t" << u << ": ";
    vidType u_0_to_2_deg = get_out_degree_0_to_2(u);
    for (vidType w_idx = 0; w_idx < u_0_to_2_deg; w_idx++)
      logFile << N_0_to_2(u, w_idx) << " ";
    logFile << "\n";
  }

  logFile << "\t" << std::get<1>(this_subgraph_tuple) << " to " << std::get<2>(this_subgraph_tuple) << ":\n";
  vidType partition_start_vertex_idx_1 = std::get<1>(partition_start_vertex_idx_tuple);
  vidType partition_num_vertices_1 = std::get<1>(partition_num_vertices_tuple);
  for (vidType v = partition_start_vertex_idx_1; v < partition_start_vertex_idx_1 + partition_num_vertices_1; v++)
  {
    std::sort(edges_1_to_2 + row_pointers_1_to_2[v - partition_start_vertex_idx_1], edges_1_to_2 + row_pointers_1_to_2[v + 1 - partition_start_vertex_idx_1]);
    logFile << "\t\t" << v << ": ";
    vidType v_1_to_2_deg = get_out_degree_1_to_2(v);
    for (vidType w_idx = 0; w_idx < v_0_to_1_deg; w_idx++)
      logFile << N_1_to_2(v, w_idx) << " ";
    logFile << "\n";
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
  vidType start_vertex_idx = std::get<0>(partition_start_vertex_idx_tuple);
  return row_pointers_0_to_1[u - start_vertex_idx + 1] - row_pointers_0_to_1[u - start_vertex_idx];
}

vidType Type3SubGraph::get_out_degree_0_to_2(vidType u)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'get_out_degree_0_to_2()' is called!\n";
    exit(-1);
  }
  vidType start_vertex_idx = std::get<0>(partition_start_vertex_idx_tuple);
  return row_pointers_0_to_2[u - start_vertex_idx + 1] - row_pointers_0_to_2[u - start_vertex_idx];
}

vidType Type3SubGraph::get_out_degree_1_to_2(vidType v)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'get_out_degree_1_to_2()' is called!\n";
    exit(-1);
  }
  vidType start_vertex_idx = std::get<1>(partition_start_vertex_idx_tuple);
  return row_pointers_1_to_2[v - start_vertex_idx + 1] - row_pointers_1_to_2[v - start_vertex_idx];
}

vidType Type3SubGraph::N_0_to_1(vidType u, vidType n)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'N_0_to_1()' is called!\n";
    exit(-1);
  }
  vidType start_vertex_idx = std::get<0>(partition_start_vertex_idx_tuple);
  return edges_0_to_1[row_pointers_0_to_1[u - start_vertex_idx] + n];
}

vidType Type3SubGraph::N_0_to_2(vidType u, vidType n)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'N_0_to_2()' is called!\n";
    exit(-1);
  }
  vidType start_vertex_idx = std::get<0>(partition_start_vertex_idx_tuple);
  return edges_0_to_2[row_pointers_0_to_2[u - start_vertex_idx] + n];
}

vidType Type3SubGraph::N_1_to_2(vidType v, vidType n)
{
  // the function must be called after the creation of the subgraph has finished
  if (creation_finished == 0)
  {
    std::cout << "Error! Subgraph creation has not finished, but 'N_1_to_2()' is called!\n";
    exit(-1);
  }
  vidType start_vertex_idx = std::get<1>(partition_start_vertex_idx_tuple);
  return edges_1_to_2[row_pointers_1_to_2[v - start_vertex_idx] + n];
}

/**************************************** Definition of tool functions ***************************************/

// given the device_count, calculate how many Type3SubGraphs
int get_type3_subgraph_num(int device_count)
{
  return (device_count * (device_count - 1) * (device_count - 2)) / 6;
}

// given the global index of Type3SubGraph, calculate its 3-tuple
type3Tuple get_type3_subgraph_tuple(int this_type3_subgraph_idx, int num_partitions)
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
  type3Tuple tuple;
  for (first = 0; first < num_partitions - 2; first++)
  {
    int num_subgraphs_with_first = (num_partitions - first - 1) * (num_partitions - first - 2) / 2;
    if (num_subgraphs_with_first <= this_type3_subgraph_idx)
      this_type3_subgraph_idx -= num_subgraphs_with_first;
    else
    {
      std::get<0>(tuple) = first;
      for (second = first + 1; second < num_partitions - 1; second++)
      {
        int num_subgraphs_with_second = num_partitions - second - 1;
        if (num_subgraphs_with_second <= this_type3_subgraph_idx)
          this_type3_subgraph_idx -= num_subgraphs_with_second;
        else
        {
          std::get<1>(tuple) = second;
          third = second + 1 + this_type3_subgraph_idx;
          std::get<2>(tuple) = third;
          break;
        }
      }
      break;
    }
  }
  return tuple;
}

// given the device_count and three seperate partition index, calculate the global index of the Type3SubGraph
int get_type3_subgraph_idx(int device_count, vidType u, vidType v, vidType w)
{
  // sort the three vertices into increasing order
  sort_three_vertices(&u, &v, &w);

  // accumulate index
  int type3_subgraph_idx = 0;
  for (vidType ui = 0; ui < u; ui++)
    type3_subgraph_idx += (device_count - ui - 1) * (device_count - ui - 2) / 2;
  for (vidType vi = ui + 1; vi < v; vi++)
    type3_subgraph_idx += device_count - vi - 1;
  type3_subgraph_idx += w - v - 1;
  return type3_subgraph_idx;
}

// sort the three vertices into increasing order
void sort_three_vertices(vidType *u, vidType *v, vidType *w)
{
  if (*u > *v) vidType temp = *v, *v = *u, *u = temp;
  if (*v > *w) vidType temp = *w, *w = *v, *v = temp;
  if (*u > *v) vidType temp = *v, *v = *u, *u = temp;
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
  logFile << "\tAll the Type1SubGraphs created!\n";
  int type3_subgraph_num = get_type3_subgraph_num(device_count);
  std::vector<Type3SubGraph> type3_subgraphs(device_count);
  for (int type3_subgraph_idx = 0 type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx++)
    type3_subgraphs[type3_subgraph_idx].init(g, device_count, type3_subgraph_idx);
  logFile << "\tAll the Type3SubGraphs created!\n";

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
        int v_partition_idx = v / normal_vertex_number_each_partition;
        if (device_idx == v_partition_idx)  // inner edge
          type1_subgraphs[device_idx].add_edge(u, v);
        else  // cross edge
        {
          // First, record in their coresponding Type1SubGraphs
          type1_subgraphs[device_idx].add_edge(u, v);
          type1_subgraphs[v_partition_idx].add_edge(v, u);

          // Second, record into all the related Type3SubGraphs
          for (vidType w = 0; w < device_count; w++)
          {
            if (w != u && w != v)
            {
              int type3_subgraph_idx = get_type3_subgraph_idx(device_count, u, v, w);
              if (u < v)
                type3_subgraphs[type3_subgraph_idx].add_edge(u, device_idx, v, v_partition_idx);
              else
                type3_subgraphs[type3_subgraph_idx].add_edge(v, v_partition_idx, u, device_idx);
            }
          }
        }
      }
    }
  }
  logFile << "Finish mapping!\n";

  // reduce
  logFile << "Start reducing ...\n";
  for (int device_idx = 0; device_idx < device_count; device_idx++)
    type1_subgraphs[device_idx].reduce();
  for (int type3_subgraph_idx = 0; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx++)
    type3_subgraphs[type3_subgraph_idx].reduce();
  logFile << "Finish reducing ...\n";

  // end and exit
  logFile << "Destoying Type1SubGraphs ...\n";
  for (int device_idx = 0; device_idx < device_count; device_idx++)
    type1_subgraphs[device_idx].destroy();
  logFile << "All the Type1SubGraphs have been destoyed!\n";
  logFile << "Destoying Type3SubGraphs ...\n";
  for (int type3_subgraph_idx = 0; type3_subgraph_idx < type3_subgraph_num; type3_subgraph_idx++)
    type3_subgraphs[type3_subgraph_idx].destroy();
  logFile << "All the Type3SubGraphs have been destoyed!\n";
  logFile << "Gracefully finishing ...\n";
  logFile.close();
}
