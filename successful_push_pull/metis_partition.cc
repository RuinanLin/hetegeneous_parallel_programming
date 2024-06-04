#include "graph.h"
#include "graph_partition.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <graph> num_gpu(4) [chunk_size(1024)] [adj_sorted(1)]\n";
        std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph 4\n";
        exit(1);
    }
    std::cout << "Triangle Counting: we assume the neighbor lists are sorted.\n";
    Graph g(argv[1], USE_DAG); // use DAG

    int n_partitions = atoi(argv[2]);
    PartitionedGraph pg(&g, n_partitions);
    pg.metis_partition();
    pg.metis_write_into_file();
}