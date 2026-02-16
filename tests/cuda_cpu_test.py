#!/usr/bin/env python3
"""
CUDA vs CPU Performance Test - NumPy Version
Works with AMD Athlon II X3 455 + NVIDIA GeForce 8800 GT

NOTE: PyTorch is incompatible with AMD Athlon II X3 455 (lacks AVX instruction set).
This test uses NumPy for CPU performance benchmarking.
"""
import time
import numpy as np

def main():
    print("\n" + "=" * 70)
    print("CPU Performance Benchmark Test")
    print("Hardware: AMD Athlon II X3 455 + NVIDIA GeForce 8800 GT")
    print("=" * 70 + "\n")
    
    print("Environment Information:")
    print(f"  NumPy version: {np.__version__}")
    print(f"  NumPy backend: {np.__config__.show()}")
    
    print("\n" + "-" * 70)
    print("Running CPU Performance Tests...")
    print("-" * 70 + "\n")
    
    # Test different matrix sizes
    sizes = [500, 1000, 2000]
    
    for size in sizes:
        print(f"\nTesting matrix size: {size}x{size}")
        print("-" * 40)
        
        try:
            # Create random matrices
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # Warm up
            _ = np.dot(a[:10, :10], b[:10, :10])
            
            # Time matrix multiplication
            start = time.perf_counter()
            c = np.dot(a, b)
            elapsed = time.perf_counter() - start
            
            gflops = (2 * size**3) / (elapsed * 1e9)
            
            print(f"  Time:   {elapsed:.4f} seconds")
            print(f"  Result: {c.shape}")
            print(f"  GFLOPs: {gflops:.2f} (billion floating-point operations/sec)")
            print(f"  Status: ✓ Success")
            
        except Exception as e:
            print(f"  Status: ✗ Failed - {e}")
    
    print("\n" + "-" * 70)
    print("Hardware Compatibility Notes:")
    print("-" * 70)
    print("""
AMD Athlon II X3 455 (2010):
  - CPU Architecture: K10 (pre-Bulldozer)
  - Supported Instructions: SSE3, SSE4a (NOT SSE4.1, SSE4.2, or AVX)
  - Modern PyTorch: Requires AVX or SSE4.2 (INCOMPATIBLE)
  - Status: NumPy-compatible, PyTorch-incompatible

NVIDIA GeForce 8800 GT (2007):
  - GPU Architecture: G92 (pre-Fermi)
  - Compute Capability: 1.1
  - Modern CUDA: Requires Compute Capability 3.5+ (INCOMPATIBLE)
  - Status: Requires legacy CUDA 6.5 or older

Recommendations:
  1. For LLM development: Use CPU-only NumPy/SciPy for learning
  2. For GPU acceleration: Consider upgrading GPU or using CPU mode
  3. For deep learning: Modern frameworks require newer hardware
    """)
    
    print("=" * 70)
    print("Benchmark completed!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()