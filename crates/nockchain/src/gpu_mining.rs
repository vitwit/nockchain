use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::warn;
use rand::Rng;
use zkvm_jetpack::form::PRIME;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

#[cfg(feature = "opencl")]
use opencl3::context::Context;
#[cfg(feature = "opencl")]
use opencl3::device::{get_all_devices, Device, DeviceInfo, CL_DEVICE_TYPE_GPU};
#[cfg(feature = "opencl")]
use opencl3::kernel::{ExecuteKernel, Kernel};
#[cfg(feature = "opencl")]
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
#[cfg(feature = "opencl")]
use opencl3::program::Program;
#[cfg(feature = "opencl")]
use opencl3::queue::CommandQueue;

const GPU_BATCH_SIZE: usize = 1024 * 1024; // Process 1M nonces per batch
const MAX_GPU_THREADS: u32 = 65536;

pub struct GpuMiningResult {
    pub found_solution: bool,
    pub hash: Vec<u64>,
    pub nonce: Vec<u64>,
    pub processed_count: u64,
}

pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),
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
}

impl GpuMiner {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let backend = Self::initialize_gpu_backend()?;
        Ok(Self {
            backend,
            is_running: Arc::new(AtomicBool::new(false)),
            batch_size: GPU_BATCH_SIZE,
        })
    }

    fn initialize_gpu_backend() -> Result<GpuBackend, Box<dyn std::error::Error>> {
        // Try CUDA first
        #[cfg(feature = "cuda")]
        {
            match Self::init_cuda() {
                Ok(device) => {
                    info!("Initialized CUDA GPU mining backend");
                    return Ok(GpuBackend::Cuda(device));
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA: {}", e);
                }
            }
        }

        // Fallback to OpenCL
        #[cfg(feature = "opencl")]
        {
            match Self::init_opencl() {
                Ok((context, queue, program, kernel)) => {
                    info!("Initialized OpenCL GPU mining backend");
                    return Ok(GpuBackend::OpenCL {
                        context,
                        queue,
                        program,
                        kernel,
                    });
                }
                Err(e) => {
                    warn!("Failed to initialize OpenCL: {}", e);
                }
            }
        }

        warn!("No GPU backend available, falling back to CPU mining");
        Ok(GpuBackend::None)
    }

    #[cfg(feature = "cuda")]
    fn init_cuda() -> Result<CudaDevice, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;
        
        // Load TIP5 mining kernel
        let ptx = Ptx::from_src(include_str!("kernels/tip5_mining.cu"));
        device.load_ptx(ptx, "tip5_mining", &["tip5_mine_batch"])?;
        
        Ok(device)
    }

    #[cfg(feature = "opencl")]
    fn init_opencl() -> Result<(Context, CommandQueue, Program, Kernel), Box<dyn std::error::Error>> {
        let device_ids = get_all_devices(CL_DEVICE_TYPE_GPU)?;
        if device_ids.is_empty() {
            return Err("No GPU devices found".into());
        }

        let device = Device::new(device_ids[0]);
        let context = Context::from_device(&device)?;
        let queue = CommandQueue::create_default(&context, device.id(), 0)?;

        let program_source = include_str!("kernels/tip5_mining.cl");
        let program = Program::create_and_build_from_source(&context, program_source, "")?;
        let kernel = Kernel::create(&program, "tip5_mine_batch")?;

        Ok((context, queue, program, kernel))
    }

    pub fn is_available(&self) -> bool {
        !matches!(self.backend, GpuBackend::None)
    }

    pub async fn mine_batch(
        &self,
        _version: &[u64; 5],
        _header: &[u64; 5], 
        _target: &[u64; 5],
        _pow_len: u64,
        _start_nonce: u64,
    ) -> Result<GpuMiningResult, Box<dyn std::error::Error>> {
        if !self.is_available() {
            return Err("GPU backend not available".into());
        }

        match &self.backend {
            #[cfg(feature = "cuda")]
            GpuBackend::Cuda(device) => {
                self.mine_batch_cuda(device, version, header, target, pow_len, start_nonce).await
            }
            #[cfg(feature = "opencl")]
            GpuBackend::OpenCL { context, queue, kernel, .. } => {
                self.mine_batch_opencl(context, queue, kernel, version, header, target, pow_len, start_nonce).await
            }
            GpuBackend::None => Err("GPU backend not available".into()),
        }
    }

    #[cfg(feature = "cuda")]
    async fn mine_batch_cuda(
        &self,
        device: &CudaDevice,
        version: &[u64; 5],
        header: &[u64; 5],
        target: &[u64; 5],
        pow_len: u64,
        start_nonce: u64,
    ) -> Result<GpuMiningResult, Box<dyn std::error::Error>> {
        let batch_size = self.batch_size;
        
        // Allocate GPU memory
        let d_version = device.htod_copy(version.to_vec())?;
        let d_header = device.htod_copy(header.to_vec())?;
        let d_target = device.htod_copy(target.to_vec())?;
        let d_results = device.alloc_zeros::<u64>(batch_size * 5)?; // 5 elements per hash result
        let d_found = device.alloc_zeros::<u32>(1)?;
        let d_solution_nonce = device.alloc_zeros::<u64>(5)?;

        // Launch kernel
        let cfg = LaunchConfig {
            grid_dim: ((batch_size as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let params = (
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
            device.launch_async("tip5_mining", "tip5_mine_batch", cfg, params)?;
        }
        device.synchronize()?;

        // Copy results back
        let found: Vec<u32> = device.dtoh_sync_copy(&d_found)?;
        let solution_found = found[0] != 0;

        let mut result = GpuMiningResult {
            found_solution: solution_found,
            hash: Vec::new(),
            nonce: Vec::new(),
            processed_count: batch_size as u64,
        };

        if solution_found {
            result.nonce = device.dtoh_sync_copy(&d_solution_nonce)?;
            // Calculate hash for the solution nonce
            result.hash = self.calculate_tip5_hash_cpu(version, header, &result.nonce, target, pow_len)?;
        }

        Ok(result)
    }

    #[cfg(feature = "opencl")]
    async fn mine_batch_opencl(
        &self,
        context: &Context,
        queue: &CommandQueue,
        kernel: &Kernel,
        version: &[u64; 5],
        header: &[u64; 5],
        target: &[u64; 5],
        pow_len: u64,
        start_nonce: u64,
    ) -> Result<GpuMiningResult, Box<dyn std::error::Error>> {
        let batch_size = self.batch_size;

        // Create buffers
        let version_buffer = Buffer::<u64>::create(context, CL_MEM_READ_ONLY, 5, std::ptr::null_mut())?;
        let header_buffer = Buffer::<u64>::create(context, CL_MEM_READ_ONLY, 5, std::ptr::null_mut())?;
        let target_buffer = Buffer::<u64>::create(context, CL_MEM_READ_ONLY, 5, std::ptr::null_mut())?;
        let results_buffer = Buffer::<u64>::create(context, CL_MEM_WRITE_ONLY, batch_size * 5, std::ptr::null_mut())?;
        let found_buffer = Buffer::<u32>::create(context, CL_MEM_WRITE_ONLY, 1, std::ptr::null_mut())?;
        let solution_nonce_buffer = Buffer::<u64>::create(context, CL_MEM_WRITE_ONLY, 5, std::ptr::null_mut())?;

        // Write input data
        queue.enqueue_write_buffer(&version_buffer, opencl3::memory::CL_BLOCKING, 0, version, &[])?;
        queue.enqueue_write_buffer(&header_buffer, opencl3::memory::CL_BLOCKING, 0, header, &[])?;
        queue.enqueue_write_buffer(&target_buffer, opencl3::memory::CL_BLOCKING, 0, target, &[])?;

        // Set kernel arguments
        ExecuteKernel::new(kernel)
            .set_arg(&version_buffer)
            .set_arg(&header_buffer)
            .set_arg(&target_buffer)
            .set_arg(&(pow_len as u64))
            .set_arg(&(start_nonce as u64))
            .set_arg(&(batch_size as u32))
            .set_arg(&results_buffer)
            .set_arg(&found_buffer)
            .set_arg(&solution_nonce_buffer)
            .set_global_work_size(batch_size)
            .enqueue_nd_range(queue)?;

        queue.finish()?;

        // Read results
        let mut found = vec![0u32; 1];
        queue.enqueue_read_buffer(&found_buffer, opencl3::memory::CL_BLOCKING, 0, &mut found, &[])?;

        let solution_found = found[0] != 0;
        let mut result = GpuMiningResult {
            found_solution: solution_found,
            hash: Vec::new(),
            nonce: Vec::new(),
            processed_count: batch_size as u64,
        };

        if solution_found {
            let mut solution_nonce = vec![0u64; 5];
            queue.enqueue_read_buffer(&solution_nonce_buffer, opencl3::memory::CL_BLOCKING, 0, &mut solution_nonce, &[])?;
            result.nonce = solution_nonce;
            result.hash = self.calculate_tip5_hash_cpu(version, header, &result.nonce, target, pow_len)?;
        }

        Ok(result)
    }

    fn calculate_tip5_hash_cpu(
        &self,
        version: &[u64; 5],
        header: &[u64; 5],
        nonce: &[u64],
        _target: &[u64; 5],
        pow_len: u64,
    ) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
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
}

pub fn create_gpu_nonce_batch(start_nonce: u64, batch_size: usize) -> Vec<Vec<u64>> {
    let mut nonces = Vec::with_capacity(batch_size);
    let mut rng = rand::thread_rng();
    
    for i in 0..batch_size {
        let mut nonce = Vec::with_capacity(5);
        let base_nonce = start_nonce.wrapping_add(i as u64);
        
        // Generate 5-element nonce tuple
        for j in 0..5 {
            let nonce_val = if j == 0 {
                base_nonce % PRIME
            } else {
                rng.gen::<u64>() % PRIME
            };
            nonce.push(nonce_val);
        }
        nonces.push(nonce);
    }
    
    nonces
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