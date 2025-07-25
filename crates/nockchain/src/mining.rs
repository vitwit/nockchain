use std::str::FromStr;

use kernels::miner::KERNEL;
use nockapp::kernel::form::SerfThread;
use nockapp::nockapp::driver::{IODriverFn, NockAppHandle, PokeResult};
use nockapp::nockapp::wire::Wire;
use nockapp::nockapp::NockAppError;
use nockapp::noun::slab::NounSlab;
use nockapp::noun::{AtomExt, NounExt};
use nockapp::save::SaveableCheckpoint;
use nockapp::utils::NOCK_STACK_SIZE_TINY;
use nockapp::CrownError;
use nockchain_libp2p_io::tip5_util::tip5_hash_to_base58;
use nockvm::interpreter::NockCancelToken;
use nockvm::noun::{Atom, D, NO, T, YES};
use nockvm_macros::tas;
use rand::Rng;
use tokio::sync::Mutex;
use tracing::{debug, info, instrument, warn};
use zkvm_jetpack::form::PRIME;
use zkvm_jetpack::noun::noun_ext::NounExt as OtherNounExt;
use crate::gpu_mining::{GpuMiner, GpuMiningResult, GPU_BATCH_SIZE};

pub enum MiningWire {
    Mined,
    Candidate,
    SetPubKey,
    Enable,
}

impl MiningWire {
    pub fn verb(&self) -> &'static str {
        match self {
            MiningWire::Mined => "mined",
            MiningWire::SetPubKey => "setpubkey",
            MiningWire::Candidate => "candidate",
            MiningWire::Enable => "enable",
        }
    }
}

impl Wire for MiningWire {
    const VERSION: u64 = 1;
    const SOURCE: &'static str = "miner";

    fn to_wire(&self) -> nockapp::wire::WireRepr {
        let tags = vec![self.verb().into()];
        nockapp::wire::WireRepr::new(MiningWire::SOURCE, MiningWire::VERSION, tags)
    }
}

#[derive(Debug, Clone)]
pub struct MiningKeyConfig {
    pub share: u64,
    pub m: u64,
    pub keys: Vec<String>,
}

impl FromStr for MiningKeyConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Expected format: "share,m:key1,key2,key3"
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err("Invalid format. Expected 'share,m:key1,key2,key3'".to_string());
        }

        let share_m: Vec<&str> = parts[0].split(',').collect();
        if share_m.len() != 2 {
            return Err("Invalid share,m format".to_string());
        }

        let share = share_m[0].parse::<u64>().map_err(|e| e.to_string())?;
        let m = share_m[1].parse::<u64>().map_err(|e| e.to_string())?;
        let keys: Vec<String> = parts[1].split(',').map(String::from).collect();

        Ok(MiningKeyConfig { share, m, keys })
    }
}

struct MiningData {
    pub block_header: NounSlab,
    pub version: NounSlab,
    pub target: NounSlab,
    pub pow_len: u64,
}

pub fn create_mining_driver(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    num_threads: u64,
    init_complete_tx: Option<tokio::sync::oneshot::Sender<()>>,
) -> IODriverFn {
    create_mining_driver_with_options(mining_config, mine, num_threads, init_complete_tx, true)
}

pub fn create_mining_driver_with_options(
    mining_config: Option<Vec<MiningKeyConfig>>,
    mine: bool,
    num_threads: u64,
    init_complete_tx: Option<tokio::sync::oneshot::Sender<()>>,
    use_gpu: bool,
) -> IODriverFn {
    Box::new(move |handle| {
        Box::pin(async move {
            let Some(configs) = mining_config else {
                enable_mining(&handle, false).await?;

                if let Some(tx) = init_complete_tx {
                    tx.send(()).map_err(|_| {
                        warn!("Could not send driver initialization for mining driver.");
                        NockAppError::OtherError
                    })?;
                }

                return Ok(());
            };
            if configs.len() == 1
                && configs[0].share == 1
                && configs[0].m == 1
                && configs[0].keys.len() == 1
            {
                set_mining_key(&handle, configs[0].keys[0].clone()).await?;
            } else {
                set_mining_key_advanced(&handle, configs).await?;
            }
            enable_mining(&handle, mine).await?;

            if let Some(tx) = init_complete_tx {
                tx.send(()).map_err(|_| {
                    warn!("Could not send driver initialization for mining driver.");
                    NockAppError::OtherError
                })?;
            }

            if !mine {
                return Ok(());
            }

            info!("Starting mining driver with {} threads", num_threads);

            // Check if GPU mining was requested and initialize H100 CUDA backend
            let gpu_requested = if use_gpu {
                match GpuMiner::new() {
                    Ok(miner) => {
                        if miner.is_available() {
                            info!("H100 GPU mining enabled successfully");
                            true
                        } else {
                            info!("GPU mining framework loaded but no H100 backend available, falling back to CPU mining");
                            false
                        }
                    }
                    Err(e) => {
                        warn!("Failed to initialize H100 GPU miner: {}, falling back to CPU mining", e);
                        false
                    }
                }
            } else {
                info!("GPU mining disabled, using CPU mining only");
                false
            };

            let mut mining_attempts = tokio::task::JoinSet::<(
                SerfThread<SaveableCheckpoint>,
                u64,
                Result<NounSlab, CrownError>,
            )>::new();
            
            // GPU mining tasks
            let mut gpu_mining_attempts = tokio::task::JoinSet::<GpuMiningResult>::new();
            
            let hot_state = zkvm_jetpack::hot::produce_prover_hot_state();
            let test_jets_str = std::env::var("NOCK_TEST_JETS").unwrap_or_default();
            let test_jets = nockapp::kernel::boot::parse_test_jets(test_jets_str.as_str());

            let mining_data: Mutex<Option<MiningData>> = Mutex::new(None);
            let mut cancel_tokens: Vec<NockCancelToken> = Vec::<NockCancelToken>::new();
            let mut gpu_nonce_counter: u64 = 0;

            loop {
                tokio::select! {
                        // Handle GPU mining results
                        gpu_result = gpu_mining_attempts.join_next(), if !gpu_mining_attempts.is_empty() => {
                            let gpu_result = gpu_result.expect("GPU mining task failed");
                            let result = gpu_result.expect("GPU mining result failed");
                            
                            if result.found_solution {
                                info!("GPU found block! Processed {} nonces", result.processed_count);
                                
                                // Convert GPU result to noun format and poke main kernel
                                let mut poke_slab = NounSlab::new();
                                
                                // Create nonce noun from GPU result
                                let mut nonce_cell = Atom::from_value(&mut poke_slab, result.nonce[0])
                                    .expect("Failed to create nonce atom")
                                    .as_noun();
                                    
                                for i in 1..5 {
                                    let nonce_atom = Atom::from_value(&mut poke_slab, result.nonce[i])
                                        .expect("Failed to create nonce atom")
                                        .as_noun();
                                    nonce_cell = T(&mut poke_slab, &[nonce_atom, nonce_cell]);
                                }
                                
                                // Create hash noun from GPU result  
                                let mut hash_cell = Atom::from_value(&mut poke_slab, result.hash[0])
                                    .expect("Failed to create hash atom")
                                    .as_noun();
                                    
                                for i in 1..5 {
                                    let hash_atom = Atom::from_value(&mut poke_slab, result.hash[i])
                                        .expect("Failed to create hash atom")
                                        .as_noun();
                                    hash_cell = T(&mut poke_slab, &[hash_atom, hash_cell]);
                                }
                                
                                // Create the poke data: [hash, nonce]
                                let poke_data = T(&mut poke_slab, &[hash_cell, nonce_cell]);
                                poke_slab.set_root(poke_data);
                                
                                handle.poke(MiningWire::Mined.to_wire(), poke_slab).await
                                    .expect("Could not poke nockchain with GPU mined PoW");
                            } else {
                                debug!("GPU batch completed, processed {} nonces", result.processed_count);
                            }
                            
                            // Restart GPU mining batch if requested
                            start_gpu_mining_batch_if_available(
                                gpu_requested, 
                                &mut gpu_nonce_counter, 
                                &mining_data, 
                                &mut gpu_mining_attempts, 
                                NockAppHandle {
                                    io_sender: handle.io_sender.clone(),
                                    effect_sender: handle.effect_sender.clone(),
                                    effect_receiver: Mutex::new(handle.effect_sender.subscribe()),
                                    metrics: handle.metrics.clone(),
                                    exit: handle.exit.clone(),
                                }
                            ).await;
                        }

                        mining_result = mining_attempts.join_next(), if !mining_attempts.is_empty() => {
                            let mining_result = mining_result.expect("Mining attempt failed");
                            let (serf, id, slab_res) = mining_result.expect("Mining attempt result failed");
                            let slab = slab_res.expect("Mining attempt result failed");
                            let result = unsafe { slab.root() };
                            
                            // Check if result is a cell - if not, it might be an error or unexpected format
                            if !result.is_cell() {
                                // Add detailed debugging for H100 deployment
                                if result.is_atom() {
                                    if let Ok(atom) = result.as_atom() {
                                        let bytes = atom.as_ne_bytes();
                                        warn!("Mining result is atom instead of cell. thread={id}, atom_bytes={:?}", 
                                              String::from_utf8_lossy(&bytes[..std::cmp::min(bytes.len(), 32)]));
                                    } else {
                                        warn!("Mining result is atom but cannot read. thread={id}");
                                    }
                                } else {
                                    warn!("Mining result is neither cell nor atom. thread={id}");
                                }
                                
                                // For H100 deployment, try restarting with a fresh serf if we keep getting bad results
                                let mining_data_guard = mining_data.lock().await;
                                if let Some(ref _mining_data_ref) = *mining_data_guard {
                                    warn!("Restarting mining thread {} due to invalid result format", id);
                                    
                                    // Create a fresh serf for this thread to avoid persistent issues
                                    let kernel = Vec::from(KERNEL);
                                    match SerfThread::<SaveableCheckpoint>::new(
                                        kernel,
                                        None,
                                        hot_state.clone(),
                                        NOCK_STACK_SIZE_TINY,
                                        test_jets.clone(),
                                        false,
                                    ).await {
                                        Ok(fresh_serf) => {
                                            start_mining_attempt(fresh_serf, mining_data_guard, &mut mining_attempts, None, id).await;
                                        }
                                        Err(e) => {
                                            warn!("Failed to create fresh serf for thread {}: {}", id, e);
                                            start_mining_attempt(serf, mining_data_guard, &mut mining_attempts, None, id).await;
                                        }
                                    }
                                } else {
                                    start_mining_attempt(serf, mining_data_guard, &mut mining_attempts, None, id).await;
                                }
                                continue;
                            }
                            
                            // If the mining attempt was cancelled, the goof goes into poke_swap which returns
                            // %poke followed by the cancelled poke. So we check for hed = %poke
                            // to identify a cancelled attempt.
                            let result_cell = result.as_cell().expect("Already checked result is a cell");
                            let hed = result_cell.head();
                            if hed.is_atom() && hed.eq_bytes("poke") {
                                //  mining attempt was cancelled. restart with current block header.
                                debug!("mining attempt cancelled. restarting on new block header. thread={id}");
                                start_mining_attempt(serf, mining_data.lock().await, &mut mining_attempts, None, id).await;
                            } else {
                                // There should only be one effect
                                let effect = result_cell.head();
                                
                                // Check if effect is a cell with the expected structure
                                if !effect.is_cell() {
                                    warn!("Mining effect is not a cell, restarting mining attempt. thread={id}");
                                    start_mining_attempt(serf, mining_data.lock().await, &mut mining_attempts, None, id).await;
                                    continue;
                                }
                                
                                let effect_result = effect.uncell();
                                if effect_result.is_err() {
                                    warn!("Mining effect could not be unpacked, restarting mining attempt. thread={id}");
                                    start_mining_attempt(serf, mining_data.lock().await, &mut mining_attempts, None, id).await;
                                    continue;
                                }
                                
                                let [head, res, tail] = effect_result.expect("Already checked effect structure");
                                if head.is_atom() && head.eq_bytes("mine-result") {
                                    if unsafe { res.raw_equals(&D(0)) } {
                                        // success
                                        // poke main kernel with mined block and start a new attempt
                                        info!("Found block! thread={id}");
                                        
                                        // Check if tail can be unpacked properly
                                        let tail_result = tail.uncell();
                                        if tail_result.is_err() {
                                            warn!("Mining success but tail could not be unpacked, restarting. thread={id}");
                                            start_mining_attempt(serf, mining_data.lock().await, &mut mining_attempts, None, id).await;
                                            continue;
                                        }
                                        let [hash, poke] = tail_result.expect("Already checked tail structure");
                                        let mut poke_slab = NounSlab::new();
                                        poke_slab.copy_into(poke);
                                        handle.poke(MiningWire::Mined.to_wire(), poke_slab).await.expect("Could not poke nockchain with mined PoW");

                                        // launch new attempt
                                        let mut nonce_slab = NounSlab::new();
                                        nonce_slab.copy_into(hash);
                                        start_mining_attempt(serf, mining_data.lock().await, &mut mining_attempts, Some(nonce_slab), id).await;
                                    } else {
                                        // failure
                                        //  launch new attempt, using hash as new nonce
                                        //  nonce is tail
                                        debug!("didn't find block, starting new attempt. thread={id}");
                                        let mut nonce_slab = NounSlab::new();
                                        nonce_slab.copy_into(tail);
                                        start_mining_attempt(serf, mining_data.lock().await, &mut mining_attempts, Some(nonce_slab), id).await;
                                    }
                                }
                            }
                        }

                    effect_res = handle.next_effect() => {
                        let Ok(effect) = effect_res else {
                            warn!("Error receiving effect in mining driver: {effect_res:?}");
                            continue;
                        };
                        let Ok(effect_cell) = (unsafe { effect.root().as_cell() }) else {
                            drop(effect);
                            continue;
                        };

                        if effect_cell.head().eq_bytes("mine") {
                            let (version_slab, header_slab, target_slab, pow_len) = {
                                // Check if effect tail can be unpacked properly
                                let tail_result = effect_cell.tail().uncell();
                                if tail_result.is_err() {
                                    warn!("Mine effect tail could not be unpacked, skipping");
                                    continue;
                                }
                                let [version, commit, target, pow_len_noun] = tail_result.expect("Already checked tail structure");
                                let mut version_slab = NounSlab::new();
                                version_slab.copy_into(version);
                                let mut header_slab = NounSlab::new();
                                header_slab.copy_into(commit);
                                let mut target_slab = NounSlab::new();
                                target_slab.copy_into(target);
                                let pow_len =
                                    pow_len_noun
                                        .as_atom()
                                        .expect("Expected pow-len to be an atom")
                                        .as_u64()
                                        .expect("Expected pow-len to be a u64");
                                (version_slab, header_slab, target_slab, pow_len)
                            };
                            debug!("received new candidate block header: {:?}",
                                tip5_hash_to_base58(*unsafe { header_slab.root() })
                                .expect("Failed to convert header to Base58")
                            );
                            *(mining_data.lock().await) = Some(MiningData {
                                block_header: header_slab,
                                version: version_slab,
                                target: target_slab,
                                pow_len: pow_len
                            });

                            // Mining hasn't started yet, so start it
                            if mining_attempts.is_empty() {
                                info!("starting mining threads");
                                for i in 0..num_threads {
                                    let kernel = Vec::from(KERNEL);
                                    let serf = SerfThread::<SaveableCheckpoint>::new(
                                        kernel,
                                        None,
                                        hot_state.clone(),
                                        NOCK_STACK_SIZE_TINY,
                                        test_jets.clone(),
                                        false,
                                    )
                                    .await
                                    .expect("Could not load mining kernel");

                                    cancel_tokens.push(serf.cancel_token.clone());

                                    start_mining_attempt(serf, mining_data.lock().await, &mut mining_attempts, None, i).await;
                                }
                                info!("mining threads started with {} threads", num_threads);
                                
                                // Start GPU mining if available
                                start_gpu_mining_batch_if_available(
                                    gpu_requested, 
                                    &mut gpu_nonce_counter, 
                                    &mining_data, 
                                    &mut gpu_mining_attempts, 
                                    NockAppHandle {
                                    io_sender: handle.io_sender.clone(),
                                    effect_sender: handle.effect_sender.clone(),
                                    effect_receiver: Mutex::new(handle.effect_sender.subscribe()),
                                    metrics: handle.metrics.clone(),
                                    exit: handle.exit.clone(),
                                }
                                ).await;
                            } else {
                                // Mining is already running so cancel all the running attemps
                                // which are mining on the old block.
                                debug!("restarting mining attempts with new block header.");
                                for token in &cancel_tokens {
                                    token.cancel();
                                }
                                
                                // Stop GPU mining tasks (they will restart automatically)  
                                if gpu_requested {
                                    // Clear GPU tasks - they will be restarted in the next iteration
                                    gpu_mining_attempts.abort_all();
                                }
                            }
                        }
                    }
                }
            }
        })
    })
}

fn create_poke(mining_data: &MiningData, nonce: &NounSlab) -> NounSlab {
    let mut slab = NounSlab::new();
    let header = slab.copy_into(unsafe { *(mining_data.block_header.root()) });
    let version = slab.copy_into(unsafe { *(mining_data.version.root()) });
    let target = slab.copy_into(unsafe { *(mining_data.target.root()) });
    let nonce = slab.copy_into(unsafe { *(nonce.root()) });
    let poke_noun = T(
        &mut slab,
        &[version, header, nonce, target, D(mining_data.pow_len)],
    );
    slab.set_root(poke_noun);
    slab
}

#[instrument(skip(handle, pubkey))]
async fn set_mining_key(
    handle: &NockAppHandle,
    pubkey: String,
) -> Result<PokeResult, NockAppError> {
    let mut set_mining_key_slab = NounSlab::new();
    let set_mining_key = Atom::from_value(&mut set_mining_key_slab, "set-mining-key")
        .expect("Failed to create set-mining-key atom");
    let pubkey_cord =
        Atom::from_value(&mut set_mining_key_slab, pubkey).expect("Failed to create pubkey atom");
    let set_mining_key_poke = T(
        &mut set_mining_key_slab,
        &[D(tas!(b"command")), set_mining_key.as_noun(), pubkey_cord.as_noun()],
    );
    set_mining_key_slab.set_root(set_mining_key_poke);

    handle
        .poke(MiningWire::SetPubKey.to_wire(), set_mining_key_slab)
        .await
}

async fn set_mining_key_advanced(
    handle: &NockAppHandle,
    configs: Vec<MiningKeyConfig>,
) -> Result<PokeResult, NockAppError> {
    let mut set_mining_key_slab = NounSlab::new();
    let set_mining_key_adv = Atom::from_value(&mut set_mining_key_slab, "set-mining-key-advanced")
        .expect("Failed to create set-mining-key-advanced atom");

    // Create the list of configs
    let mut configs_list = D(0);
    for config in configs {
        // Create the list of keys
        let mut keys_noun = D(0);
        for key in config.keys {
            let key_atom =
                Atom::from_value(&mut set_mining_key_slab, key).expect("Failed to create key atom");
            keys_noun = T(&mut set_mining_key_slab, &[key_atom.as_noun(), keys_noun]);
        }

        // Create the config tuple [share m keys]
        let config_tuple = T(
            &mut set_mining_key_slab,
            &[D(config.share), D(config.m), keys_noun],
        );

        configs_list = T(&mut set_mining_key_slab, &[config_tuple, configs_list]);
    }

    let set_mining_key_poke = T(
        &mut set_mining_key_slab,
        &[D(tas!(b"command")), set_mining_key_adv.as_noun(), configs_list],
    );
    set_mining_key_slab.set_root(set_mining_key_poke);

    handle
        .poke(MiningWire::SetPubKey.to_wire(), set_mining_key_slab)
        .await
}

//TODO add %set-mining-key-multisig poke
#[instrument(skip(handle))]
async fn enable_mining(handle: &NockAppHandle, enable: bool) -> Result<PokeResult, NockAppError> {
    let mut enable_mining_slab = NounSlab::new();
    let enable_mining = Atom::from_value(&mut enable_mining_slab, "enable-mining")
        .expect("Failed to create enable-mining atom");
    let enable_mining_poke = T(
        &mut enable_mining_slab,
        &[D(tas!(b"command")), enable_mining.as_noun(), if enable { YES } else { NO }],
    );
    enable_mining_slab.set_root(enable_mining_poke);
    handle
        .poke(MiningWire::Enable.to_wire(), enable_mining_slab)
        .await
}

async fn start_gpu_mining_batch_if_available(
    gpu_requested: bool,
    nonce_counter: &mut u64,
    mining_data: &Mutex<Option<MiningData>>,
    gpu_mining_attempts: &mut tokio::task::JoinSet<GpuMiningResult>,
    handle: NockAppHandle,
) {
    if !gpu_requested {
        return;
    }
    
    let mining_data_guard = mining_data.lock().await;
    let Some(ref mining_data_ref) = *mining_data_guard else {
        debug!("No mining data available for GPU mining");
        return;
    };
    
    // Extract mining parameters for GPU
    let version = extract_5tuple_from_noun(unsafe { *mining_data_ref.version.root() });
    let header = extract_5tuple_from_noun(unsafe { *mining_data_ref.block_header.root() });
    let target = extract_5tuple_from_noun(unsafe { *mining_data_ref.target.root() });
    let pow_len = mining_data_ref.pow_len;
    let start_nonce = *nonce_counter;
    
    // Increment nonce counter for next batch
    *nonce_counter = nonce_counter.wrapping_add(GPU_BATCH_SIZE as u64);
    
    drop(mining_data_guard);
    
    // Clone data for the async task
    let version_owned = version;
    let header_owned = header;
    let target_owned = target;
    
    info!("🔄 Spawning H100 GPU mining batch: start_nonce={}, batch_size={}", start_nonce, GPU_BATCH_SIZE);
    
    gpu_mining_attempts.spawn(async move {
        let batch_start_time = std::time::Instant::now();
        match GpuMiner::new() {
            Ok(miner) => {
                if miner.is_available() {
                    info!("🚀 H100 GPU mining batch initiated: {} nonces from start_nonce={}", miner.get_batch_size(), start_nonce);
                    
                    match miner.mine_batch(version_owned, header_owned, target_owned, pow_len, start_nonce).await {
                        Ok(result) => {
                            let batch_duration = batch_start_time.elapsed();
                            let hash_rate = result.processed_count as f64 / batch_duration.as_secs_f64();
                            
                            if result.found_solution {
                                info!("🎉 H100 GPU FOUND SOLUTION! batch_time={:.2}ms, processed={}, hash_rate={:.2} MH/s", 
                                      batch_duration.as_millis(), result.processed_count, hash_rate / 1_000_000.0);
                                
                                // Create poke data for the solution
                                let mut poke_slab = NounSlab::new();
                                
                                // Convert nonce to noun format
                                let mut nonce_cell = Atom::from_value(&mut poke_slab, result.nonce[0])
                                    .expect("Failed to create nonce atom")
                                    .as_noun();
                                
                                for i in 1..5 {
                                    let nonce_atom = Atom::from_value(&mut poke_slab, result.nonce[i])
                                        .expect("Failed to create nonce atom")
                                        .as_noun();
                                    nonce_cell = T(&mut poke_slab, &[nonce_atom, nonce_cell]);
                                }
                                
                                // Convert hash to noun format
                                let mut hash_cell = Atom::from_value(&mut poke_slab, result.hash[0])
                                    .expect("Failed to create hash atom")
                                    .as_noun();
                                
                                for i in 1..5 {
                                    let hash_atom = Atom::from_value(&mut poke_slab, result.hash[i])
                                        .expect("Failed to create hash atom")
                                        .as_noun();
                                    hash_cell = T(&mut poke_slab, &[hash_atom, hash_cell]);
                                }
                                
                                // Create poke data: [hash, nonce]
                                let poke_data = T(&mut poke_slab, &[hash_cell, nonce_cell]);
                                poke_slab.set_root(poke_data);
                                
                                // Poke the main kernel with the solution
                                if let Err(e) = handle.poke(MiningWire::Mined.to_wire(), poke_slab).await {
                                    warn!("Failed to poke H100 mining solution: {}", e);
                                }
                            }
                            
                            result
                        }
                        Err(e) => {
                            let batch_duration = batch_start_time.elapsed();
                            warn!("❌ H100 GPU mining batch failed after {:.2}ms: {}", batch_duration.as_millis(), e);
                            GpuMiningResult {
                                found_solution: false,
                                hash: Vec::new(),
                                nonce: Vec::new(),
                                processed_count: 0,
                            }
                        }
                    }
                } else {
                    warn!("⚠️ GPU mining requested but H100 backend not available (fallback to CPU)");
                    GpuMiningResult {
                        found_solution: false,
                        hash: Vec::new(),
                        nonce: Vec::new(),
                        processed_count: 0,
                    }
                }
            }
            Err(e) => {
                warn!("❌ Failed to initialize H100 GPU miner: {}", e);
                GpuMiningResult {
                    found_solution: false,
                    hash: Vec::new(),
                    nonce: Vec::new(),
                    processed_count: 0,
                }
            }
        }
    });
}

fn extract_5tuple_from_noun(noun: nockvm::noun::Noun) -> [u64; 5] {
    // Extract 5-tuple from noun format used in TIP5
    // This is a simplified implementation - in practice you'd want more robust parsing
    let mut result = [0u64; 5];
    
    if let Ok(atom) = noun.as_atom() {
        // If it's an atom, treat as single value
        result[0] = atom.as_u64().unwrap_or(0) % PRIME;
        return result;
    }
    
    // If it's a cell, try to extract 5 elements
    let mut current = noun;
    let mut index = 0;
    
    while current.is_cell() && index < 5 {
        if let Ok(cell) = current.as_cell() {
            if let Ok(head_atom) = cell.head().as_atom() {
                result[index] = head_atom.as_u64().unwrap_or(0) % PRIME;
            }
            current = cell.tail();
            index += 1;
        } else {
            break;
        }
    }
    
    result
}

async fn start_mining_attempt(
    serf: SerfThread<SaveableCheckpoint>,
    mining_data: tokio::sync::MutexGuard<'_, Option<MiningData>>,
    mining_attempts: &mut tokio::task::JoinSet<(
        SerfThread<SaveableCheckpoint>,
        u64,
        Result<NounSlab, CrownError>,
    )>,
    nonce: Option<NounSlab>,
    id: u64,
) {
    let nonce = nonce.unwrap_or_else(|| {
        let mut rng = rand::thread_rng();
        let mut nonce_slab = NounSlab::new();
        
        // Generate more conservative nonce values to avoid kernel issues
        let base_nonce = rng.gen::<u32>() as u64; // Use smaller values initially
        let mut nonce_cell = Atom::from_value(&mut nonce_slab, base_nonce % PRIME)
            .expect("Failed to create nonce atom")
            .as_noun();
        
        // Create 4 more nonce elements with controlled values
        for i in 1..5 {
            let nonce_value = (base_nonce + i) % PRIME;
            let nonce_atom = Atom::from_value(&mut nonce_slab, nonce_value)
                .expect("Failed to create nonce atom")
                .as_noun();
            nonce_cell = T(&mut nonce_slab, &[nonce_atom, nonce_cell]);
        }
        
        nonce_slab.set_root(nonce_cell);
        debug!("Generated nonce for thread {}: base={}", id, base_nonce);
        nonce_slab
    });
    let mining_data_ref = mining_data
        .as_ref()
        .expect("Mining data should already be initialized");
    debug!(
        "starting mining attempt on thread {:?} on header {:?}with nonce: {:?}",
        id,
        tip5_hash_to_base58(*unsafe { mining_data_ref.block_header.root() })
            .expect("Failed to convert block header to Base58"),
        tip5_hash_to_base58(*unsafe { nonce.root() }).expect("Failed to convert nonce to Base58"),
    );
    let poke_slab = create_poke(mining_data_ref, &nonce);
    
    // Add validation for H100 deployment
    debug!("Mining poke created for thread {}, poke_root_is_cell: {}", 
           id, unsafe { poke_slab.root().is_cell() });
    
    mining_attempts.spawn(async move {
        let result = serf.poke(MiningWire::Candidate.to_wire(), poke_slab).await;
        
        // Log the result type for debugging H100 issues
        if let Ok(ref slab) = result {
            let root = unsafe { slab.root() };
            debug!("Mining result for thread {}: is_cell={}, is_atom={}", 
                   id, root.is_cell(), root.is_atom());
        } else {
            debug!("Mining result for thread {} was an error: {:?}", id, result);
        }
        
        (serf, id, result)
    });
}
