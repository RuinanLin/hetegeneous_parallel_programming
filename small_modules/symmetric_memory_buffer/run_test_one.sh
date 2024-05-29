module load nvhpc
SDK_HOME="$(echo $(dirname $(which nvc++)) | sed "s/\/compilers\/bin.*//g")"
NVSHMEM_HOME="$SDK_HOME/comm_libs/nvshmem"
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"
export NVSHMEM_BOOTSTRAP_PMI=PMI-2
nvcc -rdc=true -ccbin nvc++ -I $NVSHMEM_HOME/include test_nvshmem_buffer_warp_one_warp.cu -c -o test_nvshmem_buffer_warp_one_warp.o
nvc++ test_nvshmem_buffer_warp_one_warp.o -o test_nvshmem_buffer_warp_one_warp.out -cuda -gpu=cc80 -L $NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda -lcudart
srun --account=bcsh-delta-gpu --partition=gpuA100x4-interactive -G 4 -n 4 -N 1 ./test_nvshmem_buffer_warp_one_warp.out