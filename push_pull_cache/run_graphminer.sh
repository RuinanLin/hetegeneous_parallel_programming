#!/bin/bash
read -p "Please enter the graph name: " name
module load nvhpc
SDK_HOME="$(echo $(dirname $(which nvc++)) | sed "s/\/compilers\/bin.*//g")"
NVSHMEM_HOME="$SDK_HOME/comm_libs/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"
export NVSHMEM_BOOTSTRAP_PMI=PMI-2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/rlin2/metis-5.1.0/build/Linux-x86_64/my_build/lib
cd hetegeneous_parallel_programming/GraphMiner/src/triangle
make metis_partition
mkdir ../../../../inputs/$name/metis
./metis_partition ../../../../inputs/$name/graph 4
make tc_multigpu_nvshmem USE_METIS=1
srun --account=bcsh-delta-gpu --partition=gpuA100x4-interactive -G 4 -n 4 -N 1 --mem=240G ../../bin/tc_multigpu_nvshmem ../../../../inputs/$name/metis/pgraph 4