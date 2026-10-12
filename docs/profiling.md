# CUDA Profiling Guide

## Tools

### NVIDIA Nsight Systems
Professional-grade profiling tool for comprehensive analysis.

```bash
nsys profile --trace cuda,osrt ./your_program
```

### NVIDIA Nsight Compute
Detailed kernel-level profiling and analysis.

```bash
ncu --set full -f ./your_program
```

### NVIDIA Visual Profiler
Legacy tool but still useful for:
- Timeline visualization
- Memory transfer analysis
- Kernel execution timing

## Key Metrics

### Throughput
- **Peak Bandwidth**: GPU's theoretical maximum memory bandwidth
- **Effective Bandwidth**: Actual achieved bandwidth
- **Utilization**: (Effective / Peak) * 100%

### Occupancy
- **Registers per thread**: Affects how many thread blocks can run
- **Shared memory per block**: Limits thread block size
- **Goal**: 50-100% occupancy (higher is not always faster)

### Latency
- **Global memory latency**: ~200-400 cycles
- **Hide latency** with sufficient parallelism

## Common Performance Issues

### Bank Conflicts
**Symptom**: Serialized memory accesses in shared memory
**Solution**: Pad arrays, use different addressing patterns

### Register Pressure
**Symptom**: Low occupancy, spilling to local memory
**Solution**: Reduce registers per thread, use `__launch_bounds__`

### Warp Divergence
**Symptom**: Threads in same warp take different code paths
**Solution**: Restructure conditionals, use ballot operations

### Memory Misalignment
**Symptom**: Non-coalesced global memory accesses
**Solution**: Ensure thread access patterns are aligned

## Profiling Workflow

1. **Get Baseline**: Profile unoptimized kernel
2. **Identify Bottleneck**: Check achievable throughput vs peak
3. **Hypothesis**: Form optimization hypothesis
4. **Implement**: Make targeted optimization
5. **Measure**: Profile again to verify improvement
6. **Iterate**: Repeat until satisfied or hitting limits

## Example: Vector Addition Profiling

```bash
# Profile with detailed metrics
ncu --set full -f -c 0 -k vector_add ./vector_add_app

# Expected results:
# - Memory bandwidth: ~400-500 GB/s (80-90% of peak)
# - Kernel time: ~1-10 microseconds
# - Occupancy: ~100%
```

## Memory Analysis

### Coalescing Check
- Verify stride-1 memory accesses
- Check memory access patterns in nsys timeline

### Cache Efficiency
- L1 cache hit rate: Aim for >80% for good reuse
- L2 cache: Shared resource, profile carefully

### Register Usage
View with:
```bash
nvcc -Xptxas -v -c kernel.cu
```
