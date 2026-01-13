# Optimization of Serialization Mechanism (v0.15)

## 1. Context & Motivation

Current `main` branch relies on `torch.distributed.rpc.internal._internal_rpc_pickler` for zero-copy serialization. While functional, it has limitations:
1.  **Dependency on Internal API**: `_internal_rpc_pickler` is a private PyTorch API, posing stability risks.
2.  **Limited Scope**: It primarily optimizes Tensors.
3.  **Performance overhead**: The internal pickler has some overhead for complex nested structures.

This PR (`feature/optimize-serialization-v0.15`) introduces a custom, lightweight zero-copy serialization mechanism.

## 2. Key Changes

### 2.1 Removed Internal Dependency
- Removed usage of `torch.distributed.rpc.internal._internal_rpc_pickler`.
- Implemented `_pack_data` and `_unpack_data` in `transfer_queue/utils/serial_utils.py` to recursively traverse data structures.

### 2.2 Custom Zero-Copy Logic
- **Tensors**: Converted to `memoryview` (CPU, contiguous) and separated from the pickle stream.
- **Large Strings**: Strings larger than 10KB (`LARGE_OBJECT_THRESHOLD`) are now treated as zero-copy buffers, reducing pickling overhead for large text payloads.
- **Protocol**: 
    - **Serialization**: Returns `[structure_bytes, buffer_1, buffer_2, ...]` where `structure_bytes` contains the metadata/skeleton of the object.
    - **Deserialization**: Reconstructs the object by recursively filling in the metadata placeholders with data from buffers.

## 3. Benchmark Results

Tests were conducted on **feature/optimize-serialization-v0.15**, **main (zero-copy enabled)**, and **main (zero-copy disabled)**.

### Summary
- **Significant performance boost** compared to `main` (zero-copy enabled), especially in **GET** operations for large payloads.
- **Massive improvement** over non-zero-copy baseline (as expected).
- **Large Strings**: The optimization for strings likely contributes to the flexibility and performance stability.

### Throughput Comparison (Gbps)

| Config | Operation | Main (No Zero-Copy) | Main (Zero-Copy) | **Optimized v0.15** | vs Main (ZC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Small** (50MB) | PUT | 0.89 | 1.47 | **1.56** | +6.1% |
| | GET | 1.14 | 2.20 | **2.53** | +15.0% |
| **Medium** (500MB)| PUT | 2.90 | 6.31 | **6.82** | +8.1% |
| | GET | 3.31 | 5.85 | **6.95** | +18.8% |
| **Large** (3GB) | PUT | 4.32 | 10.67 | **12.34** | +15.7% |
| | GET | 4.57 | 4.64 | **8.41** | **+81.2%** |
| **Huge** (10GB) | PUT | 4.31 | 9.78 | **11.08** | +13.3% |
| | GET | 4.49 | 4.26 | **5.50** | +29.1% |

### Resource Usage (CPU Cost)
The optimized version also demonstrates efficient resource usage. For the **Large** scenario:
- **Main (Zero-Copy)**: ~875 CPU seconds
- **Optimized v0.15**: ~572 CPU seconds
- **Reduction**: ~35% less CPU time usage for the same workload.

## 4. Conclusion
The new custom serialization mechanism not only removes the dependency on unstable internal PyTorch APIs but also delivers significant performance improvements (up to **81%** in specific read scenarios) and reduced CPU consumption.
