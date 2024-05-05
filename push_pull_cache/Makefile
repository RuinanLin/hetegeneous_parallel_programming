include ../common.mk
INCLUDES+=-I./gpu_kernels -I$(TBB_HOME)/include
all: tc_omp_base tc_gpu_base tc_multigpu_base tc_multigpu

tc_omp_base: $(OBJS) omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

tc_omp_simd: $(OBJS) omp_simd.o intersect.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_simd.o intersect.o -o $@ -lgomp
	mv $@ $(BIN)

tc_cilk_base: $(OBJS) cilk_base.o 
	$(CLANGXX) $(CILKFLAGS) $(OBJS) cilk_base.o -o $@
	mv $@ $(BIN)

tc_tbb_base: $(OBJS) tbb_base.o
	$(CXX) $(CXXFLAGS) -L$(TBB_HOME)/lib/intel64/gcc4.8 $(OBJS) tbb_base.o -o $@ -lgomp -ltbb
	mv $@ $(BIN)

tc_gpu_base: $(OBJS) gpu_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) gpu_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu_base: $(OBJS) multigpu_base.o scheduler.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu_base.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_multigpu: $(OBJS) multigpu.o scheduler.o graph_partition.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) multigpu.o graph_partition.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_dist_cpu: $(OBJS) dist_cpu.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_cpu.o -o $@ -lgomp
	mv $@ $(BIN)

tc_dist_gpu: $(OBJS) dist_gpu.o gpu_kernel_wrapper.o
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) dist_gpu.o gpu_kernel_wrapper.o -o $@ $(LIBS)
	mv $@ $(BIN)

metis_partition: graph.o graph_partition.o VertexSet.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) metis_partition.cc -c -o metis_partition.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) metis_partition.o graph.o graph_partition.o VertexSet.o -o $@ $(LIBS) -L/u/rlin2/metis-5.1.0/build/Linux-x86_64/my_build/lib -lmetis
	rm *.o

tc_multigpu_nvshmem: $(OBJS) graph_partition.o
	$(NVCC) -DUSE_METIS -rdc=true -ccbin nvc++ $(INCLUDES) -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen/gcc-8.5.0/nvhpc-22.11-eionxjt/Linux_x86_64/22.11/comm_libs/nvshmem/include multigpu_nvshmem_pushpull.cu -c -o multigpu_nvshmem_pushpull.o
	nvc++ $(INCLUDES) $(OBJS) multigpu_nvshmem_pushpull.o graph_partition.o -o $@ $(LIBS) -cuda -gpu=cc80 -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen/gcc-8.5.0/nvhpc-22.11-eionxjt/Linux_x86_64/22.11/comm_libs/nvshmem/lib -L/u/rlin2/metis-5.1.0/build/Linux-x86_64/my_build/lib -lnvshmem_host -lnvshmem_device -lnvidia-ml -lcuda -lcudart -lmetis
	mv $@ $(BIN)
	rm *.o

clean:
	rm *.o
