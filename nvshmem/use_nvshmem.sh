SDK_HOME="$(echo $(dirname $(which nvc++)) | sed "s/\/compilers\/bin.*//g")"
NVSHMEM_HOME="$SDK_HOME/comm_libs/nvshmem"

# nvshmem bootstrap requires libs in path
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH"
# delta requires PMI-2
export NVSHMEM_BOOTSTRAP_PMI=PMI-2

nvcc -rdc=true -ccbin nvc++ -I $NVSHMEM_HOME/include simple.cu -c -o simple.o
nvc++ simple.o -o simple.out -cuda -gpu=cc80 -L $NVSHMEM_HOME/lib -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda -lcudart