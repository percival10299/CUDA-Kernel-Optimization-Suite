# CUDA Kernel Optimization

A comprehensive collection of CUDA kernel implementations and optimizations, including vector operations, matrix multiplication, and reduction algorithms.

## Project Structure

- **kernels/**: CUDA kernel implementations
  - `vector_add/`: Vector addition kernels
  - `reduction/`: Reduction algorithms (naive, shared memory, warp optimization)
  - `matmul/`: Matrix multiplication implementations
  - `softmax/`: Softmax kernels

- **triton/**: Triton-based implementations for GPU kernels
- **benchmarks/**: Performance benchmarking suite
- **tests/**: Correctness verification tests
- **docs/**: Documentation and optimization notes

## Requirements

- CUDA Toolkit 11.0+
- Python 3.8+
- CMake 3.18+

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running Benchmarks

```bash
python benchmarks/benchmark.py
```

## License

MIT
