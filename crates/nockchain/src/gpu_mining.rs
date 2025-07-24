use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{warn, info};
use zkvm_jetpack::form::PRIME;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;

#[cfg(feature = "opencl")]
use opencl3::context::Context;
#[cfg(feature = "opencl")]
use opencl3::kernel::Kernel;
#[cfg(feature = "opencl")]
use opencl3::program::Program;
#[cfg(feature = "opencl")]
use opencl3::command_queue::CommandQueue;

// H100 optimized constants
pub const GPU_BATCH_SIZE: usize = 8 * 1024 * 1024; // Process 8M nonces per batch for H100
const H100_MAX_THREADS_PER_BLOCK: u32 = 1024; // H100 optimal block size
const H100_MAX_BLOCKS: u32 = 65536; // Maximum blocks for H100
const H100_SM_COUNT: u32 = 132; // H100 has 132 SMs
const TIP5_HASH_SIZE: usize = 5; // TIP5 produces 5 u64 elements

pub struct GpuMiningResult {
    pub found_solution: bool,
    pub hash: Vec<u64>,
    pub nonce: Vec<u64>,
    pub processed_count: u64,
}

pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<CudaDevice>),
    #[cfg(feature = "opencl")]
    OpenCL {
        context: Context,
        queue: CommandQueue,
        program: Program,
        kernel: Kernel,
    },
    None,
}

pub struct GpuMiner {
    backend: GpuBackend,
    is_running: Arc<AtomicBool>,
    batch_size: usize,
    device_info: Option<GpuDeviceInfo>,
}

#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub memory_gb: u64,
    pub sm_count: u32,
    pub max_threads_per_block: u32,
}

impl GpuMiner {
    pub fn new() -> Result<Self, String> {
        let (backend, device_info) = Self::initialize_gpu_backend()?;
        let batch_size = if device_info.is_some() {
            GPU_BATCH_SIZE
        } else {
            1024 // Fallback batch size
        };
        
        Ok(Self {
            backend,
            is_running: Arc::new(AtomicBool::new(false)),
            batch_size,
            device_info,
        })
    }

    // Simple constructor that always returns a non-available miner for testing
    #[allow(dead_code)]
    pub fn new_unavailable() -> Result<Self, String> {
        Ok(Self {
            backend: GpuBackend::None,
            is_running: Arc::new(AtomicBool::new(false)),
            batch_size: 1024,
            device_info: None,
        })
    }

    fn initialize_gpu_backend() -> Result<(GpuBackend, Option<GpuDeviceInfo>), String> {
        // Try CUDA first (preferred for H100)
        #[cfg(feature = "cuda")]
        {
            match Self::init_cuda() {
                Ok((device, device_info)) => {
                    info!("CUDA GPU backend initialized successfully: {}", device_info.name);
                    info!("Device specs: {} GB memory, {} SMs, compute {}.{}", 
                          device_info.memory_gb, device_info.sm_count, 
                          device_info.compute_capability.0, device_info.compute_capability.1);
                    return Ok((GpuBackend::Cuda(device), Some(device_info)));
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA backend: {}", e);
                }
            }
        }
        
        // Try OpenCL as fallback
        #[cfg(feature = "opencl")]
        {
            match Self::init_opencl() {
                Ok((context, queue, program, kernel)) => {
                    info!("OpenCL GPU backend initialized successfully");
                    return Ok((GpuBackend::OpenCL { context, queue, program, kernel }, None));
                }
                Err(e) => {
                    warn!("Failed to initialize OpenCL backend: {}", e);
                }
            }
        }
        
        warn!("No GPU backend available - using CPU mining");
        Ok((GpuBackend::None, None))
    }

    #[cfg(feature = "cuda")]
    fn init_cuda() -> Result<(std::sync::Arc<CudaDevice>, GpuDeviceInfo), String> {
        info!("Initializing CUDA device for H100 mining");
        
        // Initialize CUDA device (H100 should be device 0)
        let device = CudaDevice::new(0).map_err(|e| format!("Failed to create CUDA device: {}", e))?;
        
        // Get device information
        let device_name = device.name().map_err(|e| format!("Failed to get device name: {}", e))?;
        // TODO: Fix CUDA API - placeholder memory value
        let total_memory = 80 * 1024 * 1024 * 1024; // Assume 80GB for H100
        
        // Create device info struct
        let device_info = GpuDeviceInfo {
            name: device_name.clone(),
            compute_capability: (9, 0), // H100 is compute capability 9.0
            memory_gb: total_memory / (1024 * 1024 * 1024),
            sm_count: H100_SM_COUNT,
            max_threads_per_block: H100_MAX_THREADS_PER_BLOCK,
        };
        
        info!("CUDA device initialized: {}", device_name);
        info!("Memory: {} GB, SMs: {}, Max threads/block: {}", 
              device_info.memory_gb, device_info.sm_count, device_info.max_threads_per_block);
        
        // Compile and load the mining kernel
        let ptx_src = include_str!("kernels/tip5_mining.cu");
        
        let ptx = compile_ptx(ptx_src)
            .map_err(|e| format!("Failed to compile H100 kernel: {}", e))?;
            
        device.load_ptx(ptx, "tip5_mining", &["tip5_mine_batch"])
            .map_err(|e| format!("Failed to load H100 kernel: {}", e))?;
            
        info!("H100 TIP5 mining kernel loaded successfully");
        info!("H100 CUDA backend ready for high-performance mining");
        
        Ok((device, device_info))
    }

    #[cfg(feature = "opencl")]
    #[allow(dead_code)]
    fn init_opencl() -> Result<(Context, CommandQueue, Program, Kernel), String> {
        // Disabled for compilation compatibility
        Err("OpenCL initialization disabled".into())
    }

    pub fn is_available(&self) -> bool {
        !matches!(self.backend, GpuBackend::None)
    }

    pub async fn mine_batch(
        self,
        version: [u64; 5],
        header: [u64; 5], 
        target: [u64; 5],
        pow_len: u64,
        start_nonce: u64,
    ) -> Result<GpuMiningResult, String> {
        if !self.is_available() {
            return Err("GPU backend not available".into());
        }

        match &self.backend {
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(device) => {
                self.mine_batch_cuda(device, &version, &header, &target, pow_len, start_nonce).await
            }
            #[cfg(feature = "opencl")]
            GpuBackend::OpenCL { .. } => {
                Err("OpenCL mining disabled for H100 deployment".into())
            }
            GpuBackend::None => {
                Err("No GPU backend available".into())
            }
        }
    }

    #[cfg(feature = "cuda")]
    async fn mine_batch_cuda(
        &self,
        device: &std::sync::Arc<CudaDevice>,
        version: &[u64; 5],
        header: &[u64; 5],
        target: &[u64; 5],
        pow_len: u64,
        start_nonce: u64,
    ) -> Result<GpuMiningResult, String> {
        let batch_size = self.batch_size;
        info!("Starting H100 CUDA mining: {} nonces from {}", batch_size, start_nonce);
        
        let start_time = std::time::Instant::now();
        
        // Allocate GPU memory for inputs
        let d_version = device.htod_copy(version.to_vec()).map_err(|e| format!("Failed to copy version to GPU: {}", e))?;
        let d_header = device.htod_copy(header.to_vec()).map_err(|e| format!("Failed to copy header to GPU: {}", e))?;
        let d_target = device.htod_copy(target.to_vec()).map_err(|e| format!("Failed to copy target to GPU: {}", e))?;
        
        // Allocate GPU memory for outputs
        let results_size = batch_size * TIP5_HASH_SIZE;
        let d_results = device.alloc_zeros::<u64>(results_size).map_err(|e| format!("Failed to allocate GPU results buffer: {}", e))?;
        let d_found = device.alloc_zeros::<u32>(1).map_err(|e| format!("Failed to allocate GPU found buffer: {}", e))?;
        let d_solution_nonce = device.alloc_zeros::<u64>(5).map_err(|e| format!("Failed to allocate GPU solution nonce buffer: {}", e))?;
        
        // H100 optimized launch configuration
        let threads_per_block = H100_MAX_THREADS_PER_BLOCK;
        let blocks_needed = (batch_size as u32 + threads_per_block - 1) / threads_per_block;
        let blocks_per_grid = std::cmp::min(blocks_needed, H100_MAX_BLOCKS);
        
        let launch_config = LaunchConfig {
            grid_dim: (blocks_per_grid, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        
        info!("H100 launch config: {} blocks × {} threads = {} total threads", 
              blocks_per_grid, threads_per_block, blocks_per_grid * threads_per_block);
        
        // Get the compiled kernel function
        let kernel_func = device.get_func("tip5_mining", "tip5_mine_batch")
            .ok_or("TIP5 mining kernel not found - ensure kernel compilation succeeded")?;
        
        // Launch the kernel on H100
        let kernel_params = (
            &d_version,
            &d_header,
            &d_target,
            pow_len,
            start_nonce,
            batch_size as u32,
            &d_results,
            &d_found,
            &d_solution_nonce,
        );
        
        unsafe {
            kernel_func.launch(launch_config, kernel_params).map_err(|e| format!("Failed to launch H100 kernel: {}", e))?;
        }
        
        // Wait for H100 kernel completion
        device.synchronize().map_err(|e| format!("Failed to synchronize H100 device: {}", e))?;
        
        let kernel_time = start_time.elapsed();
        
        // Copy results back from H100 memory
        let found_flag: Vec<u32> = device.dtoh_sync_copy(&d_found).map_err(|e| format!("Failed to copy found flag from GPU: {}", e))?;
        let solution_found = found_flag[0] != 0;
        
        let mut result = GpuMiningResult {
            found_solution: solution_found,
            hash: Vec::new(),
            nonce: Vec::new(),
            processed_count: batch_size as u64,
        };
        
        if solution_found {
            let solution_nonce: Vec<u64> = device.dtoh_sync_copy(&d_solution_nonce).map_err(|e| format!("Failed to copy solution nonce from GPU: {}", e))?;
            result.nonce = solution_nonce;
            
            // Calculate the winning hash for verification
            result.hash = self.calculate_tip5_hash_cpu(version, header, &result.nonce, target, pow_len).map_err(|e| format!("Failed to calculate winning hash: {}", e))?;
            
            info!("🎉 H100 found solution! Nonce: {:?}", result.nonce);
            info!("🎉 Winning hash: {:?}", result.hash);
        }
        
        let total_time = start_time.elapsed();
        let hash_rate = batch_size as f64 / total_time.as_secs_f64();
        
        info!("H100 mining complete: {} nonces in {:.2}ms (kernel: {:.2}ms)", 
              batch_size, total_time.as_millis(), kernel_time.as_millis());
        info!("H100 hash rate: {:.2} MH/s", hash_rate / 1_000_000.0);
        
        Ok(result)
    }


    fn calculate_tip5_hash_cpu(
        &self,
        version: &[u64; 5],
        header: &[u64; 5],
        nonce: &[u64],
        _target: &[u64; 5],
        pow_len: u64,
    ) -> Result<Vec<u64>, String> {
        // Implement CPU-based TIP5 hash calculation for verification
        // This is a simplified version - in practice, you'd use the existing TIP5 implementation
        use zkvm_jetpack::form::math::tip5::*;
        use zkvm_jetpack::form::Belt;

        let mut input_vec = Vec::new();
        
        // Add version
        for &v in version {
            input_vec.push(Belt(v % PRIME));
        }
        
        // Add header
        for &h in header {
            input_vec.push(Belt(h % PRIME));
        }
        
        // Add nonce (ensure it's 5 elements)
        for i in 0..5 {
            let nonce_val = if i < nonce.len() { nonce[i] } else { 0 };
            input_vec.push(Belt(nonce_val % PRIME));
        }
        
        // Add pow_len
        input_vec.push(Belt(pow_len % PRIME));

        // Calculate TIP5 hash using existing implementation
        let mut sponge = [0u64; 16];
        
        // Process input in chunks
        let chunks = input_vec.chunks(10); // TIP5 rate is 10
        for chunk in chunks {
            let mut padded_chunk = [Belt(0); 10];
            for (i, &belt) in chunk.iter().enumerate() {
                if i < 10 {
                    padded_chunk[i] = belt;
                }
            }
            
            // Absorb into sponge
            for i in 0..10 {
                sponge[i] = padded_chunk[i].0;
            }
            
            permute(&mut sponge);
        }

        // Extract digest
        let mut digest = Vec::new();
        for i in 0..5 {
            digest.push(sponge[i]);
        }

        Ok(digest)
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }
    
    pub fn get_device_info(&self) -> Option<&GpuDeviceInfo> {
        self.device_info.as_ref()
    }
    
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
    
    pub fn set_batch_size(&mut self, batch_size: usize) {
        // Validate batch size for H100
        let max_batch = H100_MAX_BLOCKS as usize * H100_MAX_THREADS_PER_BLOCK as usize;
        self.batch_size = std::cmp::min(batch_size, max_batch);
        info!("Set H100 batch size to: {}", self.batch_size);
    }
    
    pub fn get_optimal_batch_size(&self) -> usize {
        if let Some(device_info) = &self.device_info {
            // For H100, use memory-based calculation
            let memory_per_nonce = 8 * TIP5_HASH_SIZE + 8 * 5; // hash + nonce storage
            let available_memory = (device_info.memory_gb * 1024 * 1024 * 1024) / 2; // Use 50% of memory
            let max_nonces = available_memory / memory_per_nonce as u64;
            
            std::cmp::min(max_nonces as usize, GPU_BATCH_SIZE)
        } else {
            1024 // Fallback for CPU
        }
    }
    
    pub async fn benchmark(&self) -> Result<f64, String> {
        if !self.is_available() {
            return Err("GPU not available for benchmarking".into());
        }
        
        info!("Starting H100 mining benchmark...");
        
        // Use sample data for benchmarking
        let version = [1u64, 2, 3, 4, 5];
        let header = [6u64, 7, 8, 9, 10];
        let target = [u64::MAX; 5]; // Very high target so we don't find solutions
        let pow_len = 64;
        let start_nonce = 0;
        
        let benchmark_batch_size = std::cmp::min(self.batch_size, 1024 * 1024); // 1M nonces for benchmark
        // Use smaller batch for benchmark
        
        // Temporarily set smaller batch size for benchmark
        let miner = Self {
            backend: match &self.backend {
                #[cfg(feature = "cuda")]
                GpuBackend::Cuda(device) => GpuBackend::Cuda(device.clone()),
                #[cfg(feature = "opencl")]
                GpuBackend::OpenCL { .. } => return Err("OpenCL benchmarking not implemented".into()),
                GpuBackend::None => return Err("No GPU backend for benchmarking".into()),
            },
            is_running: Arc::new(AtomicBool::new(false)),
            batch_size: benchmark_batch_size,
            device_info: self.device_info.clone(),
        };
        
        let start_time = std::time::Instant::now();
        let result = miner.mine_batch(version, header, target, pow_len, start_nonce).await?;
        let elapsed = start_time.elapsed();
        
        let hash_rate = result.processed_count as f64 / elapsed.as_secs_f64();
        
        info!("H100 benchmark complete: {:.2} MH/s ({} nonces in {:.2}ms)", 
              hash_rate / 1_000_000.0, result.processed_count, elapsed.as_millis());
        
        Ok(hash_rate)
    }
}

/// Generate optimized nonce batch for H100 GPU mining
pub fn create_gpu_nonce_batch(start_nonce: u64, batch_size: usize) -> Vec<Vec<u64>> {
    let mut nonces = Vec::with_capacity(batch_size);
    
    // Use deterministic nonce generation for better GPU performance
    // This matches the kernel's nonce generation algorithm
    for i in 0..batch_size {
        let base_nonce = start_nonce.wrapping_add(i as u64);
        
        // Generate 5-element nonce tuple using same algorithm as CUDA kernel
        let nonce = vec![
            base_nonce % PRIME,
            (base_nonce.wrapping_mul(0x9e3779b97f4a7c15u64)) % PRIME,
            (base_nonce.wrapping_mul(0x85ebca6b15c44e5du64)) % PRIME,
            (base_nonce.wrapping_mul(0x635d2daa5dc32e17u64)) % PRIME,
            (base_nonce.wrapping_mul(0xa4093822299f31d0u64)) % PRIME,
        ];
        
        nonces.push(nonce);
    }
    
    nonces
}

/// Validate nonce format for GPU mining
pub fn validate_nonce(nonce: &[u64]) -> bool {
    nonce.len() == 5 && nonce.iter().all(|&n| n < PRIME)
}

/// Calculate theoretical hash rate for given hardware
pub fn calculate_theoretical_hashrate(sm_count: u32, base_clock_mhz: u32, threads_per_sm: u32) -> f64 {
    let total_cores = sm_count * threads_per_sm;
    let clock_hz = base_clock_mhz as f64 * 1_000_000.0;
    
    // Estimate cycles per hash (this would need profiling to be accurate)
    let cycles_per_hash = 100.0; // Conservative estimate for TIP5
    
    (total_cores as f64 * clock_hz) / cycles_per_hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_miner_creation() {
        let miner = GpuMiner::new();
        assert!(miner.is_ok());
    }

    #[tokio::test]
    async fn test_nonce_batch_generation() {
        let nonces = create_gpu_nonce_batch(0, 1000);
        assert_eq!(nonces.len(), 1000);
        assert_eq!(nonces[0].len(), 5);
        
        // Verify nonces are within PRIME bounds
        for nonce in &nonces {
            for &val in nonce {
                assert!(val < PRIME);
            }
        }
    }
}