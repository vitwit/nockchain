use std::str::FromStr;
use std::sync::Arc;

use kernels::miner::KERNEL;
use nockapp::kernel::form::SerfThread;
use nockapp::nockapp::driver::{IODriverFn, NockAppHandle, PokeResult};
use nockapp::nockapp::wire::Wire;
use nockapp::nockapp::NockAppError;
use nockapp::noun::slab::NounSlab;
use nockapp::noun::{AtomExt, NounExt};
use nockapp::save::SaveableCheckpoint;
use nockapp::utils::NOCK_STACK_SIZE_TINY;



use nockvm::noun::{Atom, D, NO, T, YES};
use nockvm_macros::tas;
use rand::Rng;
use tokio::sync::Mutex;
use tracing::{debug, error, info, instrument, warn};
use zkvm_jetpack::form::PRIME;
use zkvm_jetpack::noun::noun_ext::NounExt as OtherNounExt;

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

            let hot_state = zkvm_jetpack::hot::produce_prover_hot_state();
            let test_jets_str = std::env::var("NOCK_TEST_JETS").unwrap_or_default();
            let test_jets = nockapp::kernel::boot::parse_test_jets(test_jets_str.as_str());

            let _mining_data: Arc<Mutex<Option<MiningData>>> = Arc::new(Mutex::new(None));

            let (tx, mut rx) = tokio::sync::mpsc::channel(1);

            let mining_handle = Arc::new(handle);

            loop {
                let mut threads = Vec::new();
                let (version_slab, header_slab, target_slab, pow_len) = {
                    let effect = mining_handle.next_effect().await.expect("Failed to get next effect");
                    let effect_cell = unsafe { effect.root().as_cell() }.expect("Expected effect to be a cell");
                    if !effect_cell.head().eq_bytes("mine") {
                        continue;
                    }
                    let [version, commit, target, pow_len_noun] = effect_cell.tail().uncell().expect(
                        "Expected three elements in %mine effect",
                    );
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

                for i in 0..num_threads {
                    let _mining_handle = mining_handle.clone();
                    let hot_state = hot_state.clone();
                    let test_jets = test_jets.clone();
                    let version_slab = version_slab.clone();
                    let header_slab = header_slab.clone();
                    let target_slab = target_slab.clone();
                    let tx = tx.clone();

                    threads.push(tokio::spawn(async move {
                        let kernel = Vec::from(KERNEL);
                        let serf = SerfThread::<SaveableCheckpoint>::new(
                            kernel,
                            None,
                            hot_state,
                            NOCK_STACK_SIZE_TINY,
                            test_jets,
                            false,
                        )
                        .await
                        .expect("Could not load mining kernel");

                        loop {
                            let nonce = {
                                let mut rng = rand::thread_rng();
                                let mut nonce_slab = NounSlab::new();
                                let mut nonce_cell =
                                    Atom::from_value(&mut nonce_slab, rng.gen::<u64>() % PRIME)
                                        .expect("Failed to create nonce atom")
                                        .as_noun();
                                for _ in 1..5 {
                                    let nonce_atom =
                                        Atom::from_value(&mut nonce_slab, rng.gen::<u64>() % PRIME)
                                            .expect("Failed to create nonce atom")
                                            .as_noun();
                                    nonce_cell = T(&mut nonce_slab, &[nonce_atom, nonce_cell]);
                                }
                                nonce_slab.set_root(nonce_cell);
                                nonce_slab
                            };

                            let poke_slab = create_poke(
                                &MiningData {
                                    block_header: header_slab.clone(),
                                    version: version_slab.clone(),
                                    target: target_slab.clone(),
                                    pow_len,
                                },
                                &nonce,
                            );

                            let slab_res = serf.poke(MiningWire::Candidate.to_wire(), poke_slab).await;
                            let slab = slab_res.expect("Mining attempt result failed");
                            let result = unsafe { slab.root() };
                            let hed = result.as_cell().expect("Expected result to be a cell").head();

                            if hed.is_atom() && hed.eq_bytes("poke") {
                                debug!("mining attempt cancelled. restarting on new block header. thread={i}");
                            } else {
                                let Ok(result_cell) = result.as_cell() else {
                                    error!("Expected result to be a cell but it wasn't");
                                    continue;
                                };
                                let effect = result_cell.head();
                                let Ok([head, res, tail]) = effect.uncell() else {
                                    error!("Expected three elements in mining result but didn't find them");
                                    continue;
                                };
                                if head.eq_bytes("mine-result") {
                                    if unsafe { res.raw_equals(&D(0)) } {
                                        info!("Found block! thread={i}");
                                        let Ok([_hash, poke]) = tail.uncell() else {
                                            error!("Expected two elements in tail but didn't find them");
                                            continue;
                                        };
                                        let mut poke_slab = NounSlab::new();
                                        poke_slab.copy_into(poke);
                                        if let Err(e) = tx.send(poke_slab).await {
                                            error!("Failed to send mined block: {e}");
                                        }
                                        break;
                                    } else {
                                        debug!("didn't find block, starting new attempt. thread={i}");
                                    }
                                }
                            }
                        }
                    }));
                }

                if let Some(poke_slab) = rx.recv().await {
                    mining_handle.poke(MiningWire::Mined.to_wire(), poke_slab).await.expect("Could not poke nockchain with mined PoW");
                }

                for thread in threads {
                    thread.abort();
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

