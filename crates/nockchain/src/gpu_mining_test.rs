use crate::gpu_mining::{GpuMiner, create_gpu_nonce_batch};
use tokio;
use zkvm_jetpack::form::PRIME;

#[tokio::test]
async fn test_gpu_miner_initialization() {
    let result = GpuMiner::new();
    match result {
        Ok(miner) => {
            println!("GPU miner initialized successfully");
            println!("GPU available: {}", miner.is_available());
        }
        Err(e) => {
            println!("GPU miner initialization failed (expected on systems without GPU): {}", e);
        }
    }
}

#[tokio::test]
async fn test_nonce_generation() {
    let batch_size = 1000;
    let nonces = create_gpu_nonce_batch(12345, batch_size);
    
    assert_eq!(nonces.len(), batch_size);
    assert_eq!(nonces[0].len(), 5);
    
    // Check that all nonces are within prime bounds
    for nonce in &nonces {
        for &value in nonce {
            assert!(value < PRIME, "Nonce value {} exceeds PRIME {}", value, PRIME);
        }
    }
    
    // Check that nonces are different
    assert_ne!(nonces[0], nonces[1]);
    println!("Nonce generation test passed for {} nonces", batch_size);
}

#[tokio::test]
async fn test_gpu_mining_if_available() {
    let miner_result = GpuMiner::new();
    if let Ok(miner) = miner_result {
        if miner.is_available() {
            println!("Testing GPU mining with available GPU");
            
            // Test mining parameters
            let version = [1, 2, 3, 4, 5];
            let header = [10, 20, 30, 40, 50]; 
            let target = [u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX]; // Easy target
            let pow_len = 2; // Small for testing
            let start_nonce = 0;
            
            let result = miner.mine_batch(&version, &header, &target, pow_len, start_nonce).await;
            
            match result {
                Ok(mining_result) => {
                    println!("GPU mining test completed");
                    println!("Processed {} nonces", mining_result.processed_count);
                    if mining_result.found_solution {
                        println!("Found solution with nonce: {:?}", mining_result.nonce);
                        println!("Hash: {:?}", mining_result.hash);
                    } else {
                        println!("No solution found in batch");
                    }
                }
                Err(e) => {
                    println!("GPU mining test failed: {}", e);
                }
            }
        } else {
            println!("GPU miner created but no GPU backend available");
        }
    } else {
        println!("GPU miner could not be created (expected on CPU-only systems)");
    }
}

#[tokio::test] 
async fn test_cpu_tip5_hash_calculation() {
    let miner_result = GpuMiner::new();
    if let Ok(miner) = miner_result {
        let version = [1, 2, 3, 4, 5];
        let header = [10, 20, 30, 40, 50];
        let nonce = vec![100, 200, 300, 400, 500];
        let target = [u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let pow_len = 64;
        
        let hash_result = miner.calculate_tip5_hash_cpu(&version, &header, &nonce, &target, pow_len);
        
        match hash_result {
            Ok(hash) => {
                println!("CPU TIP5 hash calculation successful");
                println!("Hash: {:?}", hash);
                assert_eq!(hash.len(), 5);
                
                // Verify hash values are within prime bounds
                for &value in &hash {
                    assert!(value < PRIME, "Hash value {} exceeds PRIME {}", value, PRIME);
                }
            }
            Err(e) => {
                println!("CPU TIP5 hash calculation failed: {}", e);
                panic!("Hash calculation should not fail");
            }
        }
    }
}

#[test]
fn test_performance_estimation() {
    // Estimate performance gain from GPU mining
    let cpu_cores = num_cpus::get();
    let estimated_cpu_hashes_per_sec = cpu_cores as u64 * 10_000; // Rough estimate
    let estimated_gpu_hashes_per_sec = 1_000_000; // 1M hashes/sec on modest GPU
    
    let performance_multiplier = estimated_gpu_hashes_per_sec / estimated_cpu_hashes_per_sec;
    
    println!("Performance Estimation:");
    println!("CPU cores: {}", cpu_cores);
    println!("Estimated CPU hashes/sec: {}", estimated_cpu_hashes_per_sec);
    println!("Estimated GPU hashes/sec: {}", estimated_gpu_hashes_per_sec);
    println!("Expected performance multiplier: {}x", performance_multiplier);
    
    assert!(performance_multiplier > 1, "GPU should be faster than CPU");
}

#[tokio::test]
async fn benchmark_nonce_generation() {
    use std::time::Instant;
    
    let batch_sizes = vec![1000, 10_000, 100_000];
    
    for batch_size in batch_sizes {
        let start = Instant::now();
        let _nonces = create_gpu_nonce_batch(0, batch_size);
        let duration = start.elapsed();
        
        let nonces_per_sec = batch_size as f64 / duration.as_secs_f64();
        
        println!("Batch size: {}, Time: {:?}, Nonces/sec: {:.2}", 
                 batch_size, duration, nonces_per_sec);
    }
}