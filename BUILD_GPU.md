# GPU Mining Build Instructions

This document provides detailed instructions for building Nockchain with GPU mining support using the enhanced Makefile system.

## Quick Start

### 1. Check GPU Configuration
```bash
make gpu-check
```

### 2. Check GPU Dependencies  
```bash
make gpu-deps-check
```

### 3. Build with GPU Support (Default)
```bash
make build
make install-nockchain
```

### 4. Build CPU-Only Version
```bash
make GPU_SUPPORT=false build
make GPU_SUPPORT=false install-nockchain
```

## Build Targets

### Standard Build Targets (with GPU support by default)

- `make build` - Build all components with GPU support
- `make build-rust` - Build Rust components with GPU support  
- `make test` - Run tests with GPU support
- `make install-nockchain` - Install nockchain binary with GPU support

### GPU-Specific Build Targets

- `make build-nockchain-gpu` - Build nockchain with both CUDA and OpenCL
- `make build-nockchain-cuda` - Build nockchain with CUDA only
- `make build-nockchain-opencl` - Build nockchain with OpenCL only
- `make build-nockchain-cpu` - Build nockchain with CPU mining only

### GPU-Specific Install Targets

- `make install-nockchain-gpu` - Install with both CUDA and OpenCL support
- `make install-nockchain-cuda` - Install with CUDA support only
- `make install-nockchain-opencl` - Install with OpenCL support only
- `make install-nockchain-cpu` - Install with CPU mining only

### GPU-Specific Test Targets

- `make test-gpu` - Run GPU mining tests specifically
- `make test-cpu` - Run CPU mining tests only

## Configuration

### Environment Variables

You can control GPU support using environment variables:

```bash
# Enable/disable GPU support entirely
export GPU_SUPPORT=true          # Default: true
export GPU_SUPPORT=false         # Disable GPU mining

# Control specific GPU backends
export CUDA_SUPPORT=true         # Default: true
export OPENCL_SUPPORT=true       # Default: true
```

### Using .env File

Copy the example configuration:
```bash
cp .env.example .env
```

Edit `.env` to set your preferred configuration:
```bash
# For NVIDIA GPU with CUDA only
GPU_SUPPORT=true
CUDA_SUPPORT=true
OPENCL_SUPPORT=false

# For AMD/Intel GPU with OpenCL only  
GPU_SUPPORT=true
CUDA_SUPPORT=false
OPENCL_SUPPORT=true

# For CPU-only mining
GPU_SUPPORT=false
```

### Command-Line Overrides

Override settings for individual commands:
```bash
# Build without GPU support temporarily
make GPU_SUPPORT=false build

# Install with CUDA only
make CUDA_SUPPORT=true OPENCL_SUPPORT=false install-nockchain

# Test with OpenCL only
make CUDA_SUPPORT=false OPENCL_SUPPORT=true test
```

## GPU Prerequisites

### NVIDIA GPU (CUDA)

1. **Install NVIDIA drivers**:
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-driver-470

   # Verify installation
   nvidia-smi
   ```

2. **Install CUDA toolkit**:
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-cuda-toolkit

   # Verify installation
   nvcc --version
   ```

3. **Set environment variables** (if needed):
   ```bash
   export CUDA_ROOT=/usr/local/cuda
   export CUDA_PATH=/usr/local/cuda
   export PATH=$CUDA_PATH/bin:$PATH
   ```

### AMD/Intel GPU (OpenCL)

1. **Install OpenCL headers**:
   ```bash
   # Ubuntu/Debian
   sudo apt install opencl-headers opencl-dev ocl-icd-opencl-dev

   # CentOS/RHEL
   sudo yum install opencl-headers ocl-icd-devel
   ```

2. **Install GPU-specific drivers**:
   ```bash
   # AMD GPU
   sudo apt install mesa-opencl-icd

   # Intel GPU  
   sudo apt install intel-opencl-icd

   # Verify installation
   clinfo
   ```

## Build Examples

### Example 1: Full GPU Support (CUDA + OpenCL)
```bash
# Check dependencies
make gpu-deps-check

# Build with full GPU support
make GPU_SUPPORT=true CUDA_SUPPORT=true OPENCL_SUPPORT=true build

# Install
make install-nockchain-gpu
```

### Example 2: NVIDIA GPU Only
```bash
# Build with CUDA only
make CUDA_SUPPORT=true OPENCL_SUPPORT=false build

# Install CUDA version
make install-nockchain-cuda
```

### Example 3: AMD/Intel GPU Only
```bash
# Build with OpenCL only
make CUDA_SUPPORT=false OPENCL_SUPPORT=true build  

# Install OpenCL version
make install-nockchain-opencl
```

### Example 4: CPU-Only Build
```bash
# Build without GPU support
make GPU_SUPPORT=false build

# Install CPU-only version
make install-nockchain-cpu
```

### Example 5: Development Build
```bash
# Build for development with GPU support
cargo build --features gpu

# Run tests
make test-gpu

# Run specific GPU mining tests
cargo test --features gpu gpu_mining
```

## Troubleshooting

### Common Issues

1. **CUDA compilation errors**:
   ```bash
   # Check CUDA installation
   nvcc --version
   
   # Verify CUDA paths
   echo $CUDA_ROOT
   echo $CUDA_PATH
   
   # Build with specific CUDA version
   make install-nockchain-cuda
   ```

2. **OpenCL not found**:
   ```bash
   # Check OpenCL installation
   clinfo
   
   # Install missing OpenCL components
   sudo apt install opencl-dev ocl-icd-opencl-dev
   
   # Build with OpenCL only
   make install-nockchain-opencl
   ```

3. **No GPU detected**:
   ```bash
   # Check GPU dependencies
   make gpu-deps-check
   
   # Fallback to CPU mining
   make install-nockchain-cpu
   ```

4. **Build failures**:
   ```bash
   # Clean and rebuild
   make clean
   make build
   
   # Build with verbose output
   RUST_LOG=debug make build
   ```

### Debug Information

Get detailed build information:
```bash
# Show current configuration
make gpu-check

# Show environment variables
make build  # Shows env vars before building

# Build with debug logging
RUST_LOG=debug make build
```

## Performance Validation

### Benchmarking

Test GPU performance:
```bash
# Build with GPU support
make install-nockchain-gpu

# Run GPU mining benchmarks
cargo test --release --features gpu benchmark_

# Compare CPU vs GPU performance
make test-cpu
make test-gpu
```

### Runtime Testing

Test the built binary:
```bash
# Test GPU mining (fakenet)
./target/release/nockchain --mine --mining-pubkey test_key --gpu-mining --fakenet

# Test CPU mining (fakenet)  
./target/release/nockchain --mine --mining-pubkey test_key --no-gpu --fakenet

# Check GPU status in logs
RUST_LOG=info ./target/release/nockchain --mine --mining-pubkey test_key --gpu-mining --fakenet
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Build with GPU Support

on: [push, pull_request]

jobs:
  build-gpu:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Install GPU dependencies
      run: |
        sudo apt update
        sudo apt install -y opencl-headers opencl-dev ocl-icd-opencl-dev
        
    - name: Build with GPU support
      run: |
        make gpu-deps-check
        make GPU_SUPPORT=true CUDA_SUPPORT=false OPENCL_SUPPORT=true build
        
  build-cpu:
    runs-on: ubuntu-latest  
    steps:
    - uses: actions/checkout@v2
    
    - name: Build CPU-only version
      run: |
        make GPU_SUPPORT=false build
```

### Docker Example

```dockerfile
# Multi-stage build with GPU support
FROM nvidia/cuda:12.0-devel-ubuntu20.04 as gpu-builder

# Install dependencies
RUN apt update && apt install -y curl build-essential opencl-headers opencl-dev

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source
COPY . /nockchain
WORKDIR /nockchain

# Build with GPU support
RUN make GPU_SUPPORT=true build

# Runtime stage
FROM nvidia/cuda:12.0-runtime-ubuntu20.04
COPY --from=gpu-builder /nockchain/target/release/nockchain /usr/local/bin/
CMD ["nockchain"]
```

## Advanced Configuration

### Custom Feature Combinations

For advanced users, you can specify exact feature combinations:

```bash
# Build with custom features
cargo build --release --features "jemalloc,cuda,opencl"

# Install with specific features
cargo install --path crates/nockchain --features "gpu,jemalloc"
```

### Cross-Compilation

```bash
# Cross-compile for different architectures
rustup target add x86_64-unknown-linux-musl
cargo build --target x86_64-unknown-linux-musl --features gpu
```

This comprehensive build system ensures reliable GPU mining support while maintaining flexibility for different hardware configurations and deployment scenarios.