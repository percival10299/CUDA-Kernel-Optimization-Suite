"""
Benchmark suite for CUDA kernels
"""
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def benchmark_vector_add(sizes, runs=10):
    """Benchmark vector addition with different sizes"""
    times = []
    
    for size in sizes:
        total_time = 0.0
        
        for _ in range(runs):
            # Create test data
            a = torch.randn(size, device='cuda')
            b = torch.randn(size, device='cuda')
            
            # Warm up
            _ = a + b
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            c = a + b
            torch.cuda.synchronize()
            total_time += time.time() - start
        
        avg_time = total_time / runs
        times.append(avg_time)
        print(f"Vector Add {size}: {avg_time*1000:.3f} ms")
    
    return times


def benchmark_reduction(sizes, runs=10):
    """Benchmark reduction with different sizes"""
    times = []
    
    for size in sizes:
        total_time = 0.0
        
        for _ in range(runs):
            a = torch.randn(size, device='cuda')
            
            start = time.time()
            _ = torch.sum(a)
            torch.cuda.synchronize()
            total_time += time.time() - start
        
        avg_time = total_time / runs
        times.append(avg_time)
        print(f"Reduction {size}: {avg_time*1000:.3f} ms")
    
    return times


def main():
    print("CUDA Kernel Benchmarks")
    print("=" * 50)
    
    sizes = [1024, 10240, 102400, 1024000]
    
    print("\nVector Addition Benchmark")
    vec_times = benchmark_vector_add(sizes)
    
    print("\nReduction Benchmark")
    red_times = benchmark_reduction(sizes)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results = {
        'sizes': sizes,
        'vector_add': vec_times,
        'reduction': red_times
    }
    
    import json
    with open(results_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
