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
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
#[cfg(feature = "opencl")]
use opencl3::kernel::Kernel;
#[cfg(feature = "opencl")]
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
#[cfg(feature = "opencl")]
use opencl3::program::Program;
#[cfg(feature = "opencl")]
use opencl3::command_queue::CommandQueue;
#[cfg(feature = "opencl")]
use opencl3::types::{CL_BLOCKING, cl_device_id};

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

#[cfg(feature = "opencl")]
#[derive(Clone)]
pub struct OpenCLBackend {
    device_id: cl_device_id,
    kernel_source: String,
}

#[cfg(feature = "opencl")]
unsafe impl Send for OpenCLBackend {}
#[cfg(feature = "opencl")]
unsafe impl Sync for OpenCLBackend {}

pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    Cuda(std::sync::Arc<CudaDevice>),
    #[cfg(feature = "opencl")]
    OpenCL(OpenCLBackend),
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
                Ok(opencl_backend) => {
                    info!("OpenCL GPU backend initialized successfully");
                    return Ok((GpuBackend::OpenCL(opencl_backend), None));
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
    fn init_opencl() -> Result<OpenCLBackend, String> {
        info!("Initializing OpenCL device for cross-platform GPU mining");
        
        // Get all GPU devices
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("Failed to get OpenCL GPU devices: {}", e))?;
            
        if device_ids.is_empty() {
            return Err("No OpenCL GPU devices found".into());
        }
        
        let device_id = device_ids[0];
        let device = Device::new(device_id);
        let device_name = device.name()
            .map_err(|e| format!("Failed to get OpenCL device name: {}", e))?;
            
        info!("OpenCL device selected: {}", device_name);
        
        // Test OpenCL setup by creating temporary objects
        let context = Context::from_device(&device)
            .map_err(|e| format!("Failed to create OpenCL context: {}", e))?;
            
        let _queue = CommandQueue::create_default(&context, CL_BLOCKING.into())
            .map_err(|e| format!("Failed to create OpenCL command queue: {}", e))?;
        
        // Load OpenCL kernel source
        let kernel_source = include_str!("kernels/tip5_mining.cl").to_string();
        
        // Test program compilation
        let program = Program::create_from_source(&context, &kernel_source)
            .and_then(|mut p| { p.build(&[], "").map(|_| p) })
            .map_err(|e| format!("Failed to build OpenCL program: {}", e))?;
            
        let _kernel = Kernel::create(&program, "tip5_mine_batch")
            .map_err(|e| format!("Failed to create OpenCL kernel: {}", e))?;
            
        info!("OpenCL TIP5 mining kernel compiled successfully");
        info!("OpenCL backend ready for cross-platform mining");
        
        Ok(OpenCLBackend {
            device_id,
            kernel_source,
        })
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

        match self.backend {
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(ref device) => {
                let device_clone = device.clone();
                self.mine_batch_cuda(&device_clone, &version, &header, &target, pow_len, start_nonce).await
            }
            #[cfg(feature = "opencl")]
            GpuBackend::OpenCL(ref opencl_backend) => {
                self.mine_batch_opencl(opencl_backend.clone(), version, header, target, pow_len, start_nonce).await
            }
            GpuBackend::None => {
                Err("No GPU backend available".into())
            }
        }
    }

    #[cfg(feature = "cuda")]
    async fn mine_batch_cuda(
        self,
        device: &std::sync::Arc<CudaDevice>,
        version: &[u64; 5],
        header: &[u64; 5],
        target: &[u64; 5],
        pow_len: u64,
        start_nonce: u64,
    ) -> Result<GpuMiningResult, String> {
        let batch_size = self.batch_size;
        info!("💻 H100 CUDA kernel launch: processing {} nonces starting from {}", batch_size, start_nonce);
        
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
        
        info!("⚡ H100 kernel execution starting: {} blocks × {} threads = {} parallel operations", 
              blocks_per_grid, threads_per_block, blocks_per_grid * threads_per_block);
              
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
        info!("✅ H100 kernel complete: {:.2}ms", kernel_time.as_millis());
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
            result.hash = Self::calculate_tip5_hash_cpu(version, header, &result.nonce, target, pow_len).map_err(|e| format!("Failed to calculate winning hash: {}", e))?;
            
            info!("🎆 H100 SOLUTION FOUND! nonce={:?}", result.nonce);
            info!("🎆 Winning hash: {:?}", result.hash);
        }
        
        let total_time = start_time.elapsed();
        let hash_rate = batch_size as f64 / total_time.as_secs_f64();
        
        if solution_found {
            info!("🔥 H100 MINING SUCCESS: {:.2} MH/s ({} nonces in {:.2}ms, kernel: {:.2}ms) - SOLUTION FOUND!", 
                  hash_rate / 1_000_000.0, batch_size, total_time.as_millis(), kernel_time.as_millis());
        } else {
            info!("🔍 H100 mining complete: {:.2} MH/s ({} nonces in {:.2}ms, kernel: {:.2}ms) - no solution", 
                  hash_rate / 1_000_000.0, batch_size, total_time.as_millis(), kernel_time.as_millis());
        }
        
        Ok(result)
    }

    #[cfg(feature = "opencl")]
    async fn mine_batch_opencl(
        &self,
        opencl_backend: OpenCLBackend,
        version: [u64; 5],
        header: [u64; 5],  
        target: [u64; 5],
        pow_len: u64,
        start_nonce: u64,
    ) -> Result<GpuMiningResult, String> {
        let batch_size = self.batch_size;
        info!("💻 OpenCL kernel launch: processing {} nonces starting from {}", batch_size, start_nonce);
        
        // This runs in a separate thread to avoid Send trait issues
        let result = tokio::task::spawn_blocking(move || {
            Self::mine_batch_opencl_blocking(opencl_backend, batch_size, version, header, target, pow_len, start_nonce)
        }).await.map_err(|e| format!("OpenCL task panicked: {}", e))??;
        
        Ok(result)
    }

    #[cfg(feature = "opencl")]
    fn mine_batch_opencl_blocking(
        opencl_backend: OpenCLBackend,
        batch_size: usize,
        version: [u64; 5],
        header: [u64; 5],
        target: [u64; 5],
        pow_len: u64,
        start_nonce: u64,
    ) -> Result<GpuMiningResult, String> {
        let start_time = std::time::Instant::now();
        
        // Recreate OpenCL objects from stored data
        let device = Device::new(opencl_backend.device_id);
        let context = Context::from_device(&device)
            .map_err(|e| format!("Failed to recreate OpenCL context: {}", e))?;
            
        let queue = CommandQueue::create_default(&context, CL_BLOCKING.into())
            .map_err(|e| format!("Failed to recreate OpenCL command queue: {}", e))?;
        
        let program = Program::create_from_source(&context, &opencl_backend.kernel_source)
            .and_then(|mut p| { p.build(&[], "").map(|_| p) })
            .map_err(|e| format!("Failed to rebuild OpenCL program: {}", e))?;
            
        let kernel = Kernel::create(&program, "tip5_mine_batch")
            .map_err(|e| format!("Failed to recreate OpenCL kernel: {}", e))?;
        
        // Create OpenCL buffers for inputs
        let mut d_version = unsafe { Buffer::<u64>::create(&context, CL_MEM_READ_ONLY, 5, core::ptr::null_mut()) }
            .map_err(|e| format!("Failed to create version buffer: {}", e))?;
        let mut d_header = unsafe { Buffer::<u64>::create(&context, CL_MEM_READ_ONLY, 5, core::ptr::null_mut()) }
            .map_err(|e| format!("Failed to create header buffer: {}", e))?;
        let mut d_target = unsafe { Buffer::<u64>::create(&context, CL_MEM_READ_ONLY, 5, core::ptr::null_mut()) }
            .map_err(|e| format!("Failed to create target buffer: {}", e))?;
        
        // Create OpenCL buffers for outputs
        let results_size = batch_size * TIP5_HASH_SIZE;
        let d_results = unsafe { Buffer::<u64>::create(&context, CL_MEM_WRITE_ONLY, results_size, core::ptr::null_mut()) }
            .map_err(|e| format!("Failed to create results buffer: {}", e))?;
        let mut d_found = unsafe { Buffer::<u32>::create(&context, CL_MEM_WRITE_ONLY, 1, core::ptr::null_mut()) }
            .map_err(|e| format!("Failed to create found buffer: {}", e))?;
        let mut d_solution_nonce = unsafe { Buffer::<u64>::create(&context, CL_MEM_WRITE_ONLY, 5, core::ptr::null_mut()) }
            .map_err(|e| format!("Failed to create solution nonce buffer: {}", e))?;
        
        // Write input data to OpenCL buffers
        unsafe {
            queue.enqueue_write_buffer(&mut d_version, CL_BLOCKING, 0, &version, &[])
                .map_err(|e| format!("Failed to write version to GPU: {}", e))?;
            queue.enqueue_write_buffer(&mut d_header, CL_BLOCKING, 0, &header, &[])
                .map_err(|e| format!("Failed to write header to GPU: {}", e))?;
            queue.enqueue_write_buffer(&mut d_target, CL_BLOCKING, 0, &target, &[])
                .map_err(|e| format!("Failed to write target to GPU: {}", e))?;
        }
        
        // Set kernel arguments
        unsafe {
            kernel.set_arg(0, &d_version).map_err(|e| format!("Failed to set version arg: {}", e))?;
            kernel.set_arg(1, &d_header).map_err(|e| format!("Failed to set header arg: {}", e))?;
            kernel.set_arg(2, &d_target).map_err(|e| format!("Failed to set target arg: {}", e))?;
            kernel.set_arg(3, &pow_len).map_err(|e| format!("Failed to set pow_len arg: {}", e))?;
            kernel.set_arg(4, &start_nonce).map_err(|e| format!("Failed to set start_nonce arg: {}", e))?;
            kernel.set_arg(5, &(batch_size as u32)).map_err(|e| format!("Failed to set batch_size arg: {}", e))?;
            kernel.set_arg(6, &d_results).map_err(|e| format!("Failed to set results arg: {}", e))?;
            kernel.set_arg(7, &d_found).map_err(|e| format!("Failed to set found arg: {}", e))?;
            kernel.set_arg(8, &d_solution_nonce).map_err(|e| format!("Failed to set solution_nonce arg: {}", e))?;
        }
        
        // Calculate work dimensions
        let global_work_size = [batch_size];
        let local_work_size = [256]; // Optimal local work size for most GPUs
        
        info!("⚡ OpenCL kernel execution starting: {} work items", batch_size);
        
        // Execute the kernel
        let kernel_event = unsafe { 
            queue.enqueue_nd_range_kernel(kernel.get(), 1, std::ptr::null(), global_work_size.as_ptr(), local_work_size.as_ptr(), &[])
                .map_err(|e| format!("Failed to execute OpenCL kernel: {}", e))?
        };
        
        // Wait for kernel completion
        kernel_event.wait()
            .map_err(|e| format!("Failed to wait for OpenCL kernel: {}", e))?;
        
        let kernel_time = start_time.elapsed();
        info!("✅ OpenCL kernel complete: {:.2}ms", kernel_time.as_millis());
        
        // Read results back from GPU
        let mut found_flag = vec![0u32; 1];
        unsafe {
            queue.enqueue_read_buffer(&mut d_found, CL_BLOCKING, 0, &mut found_flag, &[])
                .map_err(|e| format!("Failed to read found flag from GPU: {}", e))?;
        }
        
        let solution_found = found_flag[0] != 0;
        
        let mut result = GpuMiningResult {
            found_solution: solution_found,
            hash: Vec::new(),
            nonce: Vec::new(),
            processed_count: batch_size as u64,
        };
        
        if solution_found {
            let mut solution_nonce = vec![0u64; 5];
            unsafe {
                queue.enqueue_read_buffer(&mut d_solution_nonce, CL_BLOCKING, 0, &mut solution_nonce, &[])
                    .map_err(|e| format!("Failed to read solution nonce from GPU: {}", e))?;
            }
            
            result.nonce = solution_nonce;
            
            // Calculate the winning hash for verification
            result.hash = Self::calculate_tip5_hash_cpu(&version, &header, &result.nonce, &target, pow_len)
                .map_err(|e| format!("Failed to calculate winning hash: {}", e))?;
            
            info!("🎆 OpenCL SOLUTION FOUND! nonce={:?}", result.nonce);
            info!("🎆 Winning hash: {:?}", result.hash);
        }
        
        let total_time = start_time.elapsed();
        let hash_rate = batch_size as f64 / total_time.as_secs_f64();
        
        if solution_found {
            info!("🔥 OpenCL MINING SUCCESS: {:.2} MH/s ({} nonces in {:.2}ms, kernel: {:.2}ms) - SOLUTION FOUND!", 
                  hash_rate / 1_000_000.0, batch_size, total_time.as_millis(), kernel_time.as_millis());
        } else {
            info!("🔍 OpenCL mining complete: {:.2} MH/s ({} nonces in {:.2}ms, kernel: {:.2}ms) - no solution", 
                  hash_rate / 1_000_000.0, batch_size, total_time.as_millis(), kernel_time.as_millis());
        }
        
        Ok(result)
    }

    fn calculate_tip5_hash_cpu(
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
                GpuBackend::OpenCL(opencl_backend) => GpuBackend::OpenCL(opencl_backend.clone()),
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