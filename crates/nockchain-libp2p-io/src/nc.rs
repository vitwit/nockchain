use std::collections::HashMap;
use std::mem::size_of;
use std::str::FromStr;
use std::sync::Arc;

use bytes::Bytes;
use either::{Either, Left, Right};
use futures::{Future, StreamExt};
use libp2p::identify::Event::Received;
use libp2p::identity::Keypair;
use libp2p::kad::NoKnownPeers;
use libp2p::multiaddr::Protocol;
use libp2p::peer_store::Store;
use libp2p::request_response::Event::*;
use libp2p::request_response::Message::*;
use libp2p::request_response::{self};
use libp2p::swarm::{ConnectionId, DialError, ListenError, SwarmEvent};
use libp2p::{
    allow_block_list, connection_limits, memory_connection_limits, Multiaddr, PeerId, Swarm,
};
use nockapp::driver::{IODriverFn, NockAppHandle, PokeResult};
use nockapp::noun::slab::NounSlab;
use nockapp::noun::FromAtom;
use nockapp::utils::make_tas;
use nockapp::utils::scry::*;
use nockapp::wire::{Wire, WireRepr};
use nockapp::{AtomExt, NockAppError, NounExt};
use nockvm::noun::{Atom, Noun, D, T};
use nockvm_macros::tas;
use rand::seq::SliceRandom;
use serde_bytes::ByteBuf;
use tokio::sync::{mpsc, Mutex, MutexGuard};
use tokio::task::{AbortHandle, JoinError, JoinSet};
use tokio::time::{Duration, MissedTickBehavior};
use tracing::{debug, error, info, instrument, trace, warn};

use crate::config::LibP2PConfig;
use crate::metrics::NockchainP2PMetrics;
use crate::p2p::*;
use crate::p2p_util::{
    log_fail2ban_ipv4, log_fail2ban_ipv6, CacheResponse, MessageTracker, NockchainDataRequest,
    NockchainFact, PeerIdExt,
};
use crate::tip5_util::tip5_hash_to_base58;

//TODO This wire is a placeholder for now. The libp2p driver is entangled with the other types of nockchain pokes
//for historical reasons, and should be disentangled in the future.
pub enum NockchainWire {
    Local,
}

impl Wire for NockchainWire {
    const VERSION: u64 = 1;
    const SOURCE: &'static str = "nc";
}

#[derive(Debug)]
pub enum Libp2pWire {
    Gossip(PeerId),
    Response(PeerId),
}

impl Libp2pWire {
    fn verb(&self) -> &'static str {
        match self {
            Libp2pWire::Gossip(_) => "gossip",
            Libp2pWire::Response(_) => "response",
        }
    }

    fn peer_id(&self) -> &PeerId {
        match self {
            Libp2pWire::Gossip(peer_id) => peer_id,
            Libp2pWire::Response(peer_id) => peer_id,
        }
    }
}

impl Wire for Libp2pWire {
    const VERSION: u64 = 1;
    const SOURCE: &'static str = "libp2p";

    fn to_wire(&self) -> WireRepr {
        let tags = vec![self.verb().into(), "peer-id".into(), self.peer_id().to_base58().into()];
        WireRepr::new(Libp2pWire::SOURCE, Libp2pWire::VERSION, tags)
    }
}

enum EffectType {
    Gossip,
    Request,
    LiarPeer,
    LiarBlockId,
    Track,
    Seen,
    Unknown,
}

impl EffectType {
    fn from_noun_slab(noun_slab: &NounSlab) -> Self {
        let Ok(effect_cell) = (unsafe { noun_slab.root().as_cell() }) else {
            return EffectType::Unknown;
        };

        let head = effect_cell.head();
        let Ok(atom) = head.as_atom() else {
            return EffectType::Unknown;
        };
        let Ok(bytes) = atom.to_bytes_until_nul() else {
            warn!("atom was not properly formatted: {:?}", atom);
            return EffectType::Unknown;
        };

        match bytes.as_slice() {
            b"gossip" => EffectType::Gossip,
            b"request" => EffectType::Request,
            b"liar-peer" => EffectType::LiarPeer,
            b"liar-block-id" => EffectType::LiarBlockId,
            b"track" => EffectType::Track,
            b"seen" => EffectType::Seen,
            _ => EffectType::Unknown,
        }
    }
}

struct TrackedJoinSet<T> {
    inner: JoinSet<T>,
    tasks: HashMap<String, AbortHandle>,
}

impl<T: 'static> TrackedJoinSet<T> {
    fn new() -> Self {
        Self {
            inner: JoinSet::new(),
            tasks: HashMap::new(),
        }
    }

    fn spawn(&mut self, name: String, task: impl Future<Output = T> + Send + 'static)
    where
        T: Send + 'static,
    {
        let handle = self.inner.spawn(task);
        self.tasks.insert(name, handle);
    }

    async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
        let result = self.inner.join_next().await;
        if result.is_some() {
            // Remove the completed task from our tracking
            self.tasks.retain(|_, v| !v.is_finished());
        }
        result
    }

    // Keep this around for debugging
    #[allow(dead_code)]
    fn get_running_tasks(&self) -> Vec<String> {
        self.tasks.keys().cloned().collect()
    }
}

#[instrument(skip(keypair, bind, allowed, limits, memory_limits, equix_builder))]
pub fn make_libp2p_driver(
    keypair: Keypair,
    bind: Vec<Multiaddr>,
    allowed: Option<allow_block_list::Behaviour<allow_block_list::AllowedPeers>>,
    limits: connection_limits::ConnectionLimits,
    memory_limits: Option<memory_connection_limits::Behaviour>,
    initial_peers: &[Multiaddr],
    force_peers: &[Multiaddr],
    prune_inbound_size: Option<usize>,
    equix_builder: equix::EquiXBuilder,
    chain_interval: Duration,
    init_complete_tx: Option<tokio::sync::oneshot::Sender<()>>,
) -> IODriverFn {
    let initial_peers = Vec::from(initial_peers);
    let force_peers = Vec::from(force_peers);
    Box::new(move |handle| {
        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Failed to register metrics!"),
        );

        Box::pin(async move {
            let libp2p_config = LibP2PConfig::from_env()?;
            debug!("Libp2p config: {:?}", libp2p_config);
            let kademlia_bootstrap_interval = libp2p_config.kademlia_bootstrap_interval();
            let force_peer_dial_interval = libp2p_config.force_peer_dial_interval();
            let request_high_reset = libp2p_config.request_high_reset();
            let initial_peer_retries = libp2p_config.initial_peer_retries;
            let request_high_threshold = libp2p_config.request_high_threshold;
            let peer_status_interval = libp2p_config.peer_status_interval_secs();
            let elders_debounce_reset = libp2p_config.elders_debounce_reset();
            let seen_tx_clear_interval = libp2p_config.seen_tx_clear_interval();
            let min_peers = libp2p_config.min_peers();
            let poke_timeout = libp2p_config.poke_timeout();
            let mut swarm = match crate::p2p::start_swarm(
                libp2p_config, keypair, bind, allowed, limits, memory_limits,
            ) {
                Ok(swarm) => swarm,
                Err(e) => {
                    error!("Could not create swarm: {}", e);
                    let (_, handle_clone) = handle.dup();
                    tokio::spawn(async move {
                        if let Err(e) = handle_clone.exit.exit(1).await {
                            error!("Failed to send exit signal: {}", e);
                        }
                    });
                    return Err(NockAppError::OtherError);
                }
            };
            let (swarm_tx, mut swarm_rx) = mpsc::channel::<SwarmAction>(1000); // number needs to be high enough to send gossips to peers
            let mut join_set = TrackedJoinSet::<Result<(), NockAppError>>::new();
            let message_tracker = Arc::new(Mutex::new(MessageTracker::new(
                metrics.clone(),
                seen_tx_clear_interval,
            )));
            let mut kad_bootstrap = tokio::time::interval(kademlia_bootstrap_interval);
            kad_bootstrap.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut force_peer_dial = tokio::time::interval(force_peer_dial_interval);
            force_peer_dial.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut reset_request_counts = tokio::time::interval(request_high_reset);
            reset_request_counts.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut reset_elders_debounce = tokio::time::interval(elders_debounce_reset);
            reset_elders_debounce.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut nockchain_timer = tokio::time::interval(chain_interval);
            nockchain_timer.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let nockchain_timer_mutex = Arc::new(Mutex::new(()));
            let (traffic_handle, effect_handle) = handle.dup();
            let traffic_cop =
                traffic_cop::TrafficCop::new(traffic_handle, &mut join_set, poke_timeout);

            let mut initial_peer_retries_remaining = initial_peer_retries;
            dial_peers(&mut swarm, &initial_peers)?;
            if let Some(tx) = init_complete_tx {
                let _ = tx.send(());
                debug!("libp2p driver initialization complete signal sent");
            }
            let mut connectivity_interval = tokio::time::interval(peer_status_interval);
            loop {
                let timer_fut = async {
                    let _ = nockchain_timer.tick().await;
                    nockchain_timer_mutex.clone().lock_owned().await
                };
                tokio::select! {
                    guard = timer_fut => {
                        join_set.spawn("timer".to_string(), send_timer_poke(guard, traffic_cop.clone(), metrics.clone()))
                    }
                    _ = connectivity_interval.tick() => {
                        let peer_count = log_peer_status(&mut swarm, &metrics).await;
                        if peer_count < min_peers {
                            let tracker = message_tracker.lock().await;
                            dial_more_peers(&mut swarm, tracker);
                        }
                    },
                    Ok(noun_slab) = effect_handle.next_effect() => {
                        let _span = tracing::trace_span!("broadcast").entered();
                        let swarm_tx_clone = swarm_tx.clone();
                        let equix_builder_clone = equix_builder.clone();
                        let local_peer_id = *swarm.local_peer_id();
                        let connected_peers: Vec<PeerId> = swarm.connected_peers().cloned().collect();
                        let message_tracker_clone = Arc::clone(&message_tracker); // Clone the Arc, not the MessageTracker
                        let metrics_clone = metrics.clone();
                        join_set.spawn("handle_effect".to_string(), async move {
                            handle_effect(noun_slab, swarm_tx_clone, equix_builder_clone, local_peer_id, connected_peers, message_tracker_clone, metrics_clone).await
                        });
                    },
                    Some(event) = swarm.next() => {
                        match event {
                            SwarmEvent::NewListenAddr { address, .. } => {
                                info!("SEvent: Listening on {address:?}");
                            },
                            SwarmEvent::ListenerError { error, .. } => {
                                error!("SEvent: Listener error: {error:?}");
                            },
                            SwarmEvent::ListenerClosed { addresses, reason, .. } => {
                                if let Err(e) = reason {
                                    error!("SEvent: Listener closed on {addresses:?} because of {e:?}");
                                } else {
                                    info!("SEvent: Listener closed on {addresses:?}");
                                }
                            },
                            SwarmEvent::Behaviour(NockchainEvent::Identify(Received { connection_id: _, peer_id, info })) => {
                                trace!("SEvent: identify_received");
                                identify_received(&mut swarm, peer_id, info)?;
                            },
                            SwarmEvent::ConnectionEstablished { connection_id, peer_id, endpoint, .. } => {
                                message_tracker.lock().await.track_connection(connection_id, peer_id, endpoint.get_remote_address(), endpoint.clone());
                                debug!("SEvent: {peer_id} is new friend via: {endpoint:?}");
                            },
                            SwarmEvent::ConnectionClosed { connection_id, peer_id, endpoint, cause, .. } => {
                                let mut message_tracker_lock = message_tracker.lock().await;
                                let _ = message_tracker_lock.lost_connection(connection_id);
                                if let Some(cause) = cause {
                                    debug!("SEvent: friendship ended with {peer_id} via: {endpoint:?}. cause: {cause:?}");
                                } else {
                                    debug!("SEvent: friendship ended by us with {peer_id} via: {endpoint:?}.");
                                }
                            },
                            SwarmEvent::IncomingConnectionError { local_addr, send_back_addr, error, .. } => {
                               trace!("SEvent: Failed incoming connection from {} to {}: {}",
                               send_back_addr, local_addr, error);

                               // When connection limits are reached, randomly prune inbound connections
                               match error {
                                   ListenError::Denied { cause } => {
                                       metrics.incoming_connections_blocked_by_limits.increment();
                                       if let Some(prune_factor) = prune_inbound_size {
                                           if let Ok(_exceeded) = cause.downcast::<libp2p::connection_limits::Exceeded>() {
                                               message_tracker.lock().await.prune_inbound_connections(metrics.clone(), &mut swarm, prune_factor);
                                           }
                                       }
                                   }
                                   _ => {}
                               }
                            },
                            SwarmEvent::Behaviour(NockchainEvent::RequestResponse(Message { connection_id , peer, message })) => {
                                trace!("SEvent: received RequestResponse");
                                let _span = tracing::debug_span!("SwarmEvent::Behavior(NockchainEvent::RequestResponse(…))").entered();
                                let swarm_tx_clone = swarm_tx.clone();
                                let mut equix_builder_clone = equix_builder.clone();
                                let local_peer_id = *swarm.local_peer_id();
                                // We have to dup and move a handle back into `handle` to propitiate the borrow checker
                                let traffic_clone = traffic_cop.clone();
                                let metrics = metrics.clone();
                                let message_tracker_clone = Arc::clone(&message_tracker); // Clone the Arc, not the MessageTracker
                                join_set.spawn("handle_request_response".to_string(), async move {
                                    handle_request_response(peer, connection_id, message, swarm_tx_clone, &mut equix_builder_clone, local_peer_id, traffic_clone, metrics.clone(), message_tracker_clone, request_high_threshold).await
                                });
                            },
                            SwarmEvent::Behaviour(NockchainEvent::RequestResponse(OutboundFailure { peer, error, ..})) => {
                                log_outbound_failure(peer, error, metrics.clone());
                            }
                            SwarmEvent::Behaviour(NockchainEvent::RequestResponse(InboundFailure { peer, error, .. })) => {
                                log_inbound_failure(peer, error, metrics.clone());
                            }
                            SwarmEvent::OutgoingConnectionError { error, .. } => {
                                log_dial_error(error);
                            },
                            SwarmEvent::IncomingConnection {
                                local_addr,
                                send_back_addr,
                                connection_id,
                                ..
                            } => {
                                debug!("SEvent: Incoming connection from {local_addr:?} to {send_back_addr:?} with {connection_id:?}");
                            },
                            SwarmEvent::Dialing { peer_id, connection_id } => {
                                debug!("SEvent: Dialing {peer_id:?} {connection_id}");
                            },
                            _ => {
                                // Handle other swarm events
                                trace!("SEvent: other swarm event {:?}", event);
                            }
                        }
                    },
                    Some(swarm_action) = swarm_rx.recv() => {
                        // We do this because Swarm doesn't implement Send, and so we can't pass it into the tasks
                        // being spawned in the match cases above.
                        match swarm_action {
                            SwarmAction::SendRequest { peer_id, request } => {
                                trace!("SAction: SendRequest: {peer_id}");
                                let _ = swarm.behaviour_mut().request_response.send_request(&peer_id, request);
                            },
                            SwarmAction::SendResponse { channel, response } => {
                                trace!("SAction: SendResponse");
                                let _ = swarm.behaviour_mut().request_response.send_response(channel, response);
                            },
                            SwarmAction::BlockPeer { peer_id } => {
                                warn!("SAction: Blocking peer {peer_id}");
                                // Block the peer in the allow_block_list
                                swarm.behaviour_mut().allow_block_list.block_peer(peer_id);
                                {
                                    // get peer IP address from the swarm
                                    let peer_addresses = swarm.behaviour_mut().peer_store.store().addresses_of_peer(&peer_id);
                                    if let Some(peer_multi_addrs) = peer_addresses {
                                        for multi_addr in peer_multi_addrs {
                                            for protocol in multi_addr.iter() {

                                                match protocol {
                                                    libp2p::core::multiaddr::Protocol::Ip4(ip) => {
                                                        log_fail2ban_ipv4(&peer_id, &ip);
                                                    },
                                                    libp2p::core::multiaddr::Protocol::Ip6(ip) => {
                                                        log_fail2ban_ipv6(&peer_id, &ip);
                                                    },
                                                    // TODO: Dns?
                                                    _ => {}
                                                }
                                            }
                                        }
                                    } else {
                                        error!("Failed to get peer IP address for peer id: {peer_id}");
                                    };
                                }
                                // Disconnect the peer if they're currently connected
                                let _ = swarm.disconnect_peer_id(peer_id);
                            },
                        }
                    },
                    _ = kad_bootstrap.tick() => {
                        // If we don't have any peers, we should retry dialing our initial peers
                        if let Err(NoKnownPeers())= swarm.behaviour_mut().kad.bootstrap() {
                            if initial_peer_retries_remaining > 0 {
                                info!("Failed to bootstrap: {}", NoKnownPeers());
                                initial_peer_retries_remaining -= 1;
                                dial_peers(&mut swarm, &initial_peers)?;
                            } else {
                                warn!("Failed to bootstrap after {} retries, will not attempt to redial initial peers.", initial_peer_retries);
                            }
                        }
                    },
                    _ = force_peer_dial.tick() => {
                        debug!("Force dialing peers");
                        dial_peers(&mut swarm, &force_peers)?;
                    },
                    _ = reset_request_counts.tick() => {
                        trace!("Resetting request counts");
                        message_tracker.lock().await.reset_requests();
                    },
                    _ = reset_elders_debounce.tick() => {
                        trace!("Resetting elders debounce");
                        let mut tracker = message_tracker.lock().await;
                        tracker.seen_elders.clear();
                    },
                    Some(result) = join_set.join_next() => {
                        if let Err(e) = result {
                            error!("Task error: {:?}", e);
                        }
                    },
                }
            }
        })
    })
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
/// Network struct (in serde/CBOR) for requests
pub enum NockchainRequest {
    /// Request a block or TX from another node, carry PoW
    Request {
        pow: equix::SolutionByteArray,
        nonce: u64,
        message: ByteBuf,
    },
    /// Gossip a block or TX to another node
    Gossip { message: ByteBuf },
}

impl NockchainRequest {
    /// Make a new "request" which gossips a block or a TX
    fn new_gossip(message: &NounSlab) -> NockchainRequest {
        let message_bytes = ByteBuf::from(message.jam().as_ref());
        NockchainRequest::Gossip {
            message: message_bytes,
        }
    }

    /// Make a new request for a block or a TX
    fn new_request(
        builder: &mut equix::EquiXBuilder,
        local_peer_id: &libp2p::PeerId,
        remote_peer_id: &libp2p::PeerId,
        message: &NounSlab,
    ) -> NockchainRequest {
        let message_bytes = ByteBuf::from(message.jam().as_ref());
        let local_peer_bytes = (*local_peer_id).to_bytes();
        let remote_peer_bytes = (*remote_peer_id).to_bytes();
        let mut pow_buf = Vec::with_capacity(
            size_of::<u64>()
                + local_peer_bytes.len()
                + remote_peer_bytes.len()
                + message_bytes.len(),
        );
        pow_buf.extend_from_slice(&[0; size_of::<u64>()][..]);
        pow_buf.extend_from_slice(&local_peer_bytes[..]);
        pow_buf.extend_from_slice(&remote_peer_bytes[..]);
        pow_buf.extend_from_slice(&message_bytes[..]);

        let mut nonce = 0u64;
        let sol_bytes = loop {
            {
                let nonce_buf = &mut pow_buf[0..size_of::<u64>()];
                nonce_buf.copy_from_slice(&nonce.to_le_bytes()[..]);
            }
            if let Ok(sols) = builder.solve(&pow_buf[..]) {
                if !sols.is_empty() {
                    break sols[0].to_bytes();
                }
            }
            nonce += 1;
        };

        NockchainRequest::Request {
            pow: sol_bytes,
            nonce,
            message: message_bytes,
        }
    }

    /// Verify the EquiX PoW attached to a request
    fn verify_pow(
        &self,
        builder: &mut equix::EquiXBuilder,
        local_peer_id: &libp2p::PeerId,
        remote_peer_id: &libp2p::PeerId,
    ) -> Result<(), equix::Error> {
        match self {
            NockchainRequest::Request {
                pow,
                nonce,
                message,
            } => {
                //  This looks backwards, but it's because which node is local and which is remote
                //  is swapped between generation at the sender and verification at the receiver.
                let local_peer_bytes = (*remote_peer_id).to_bytes();
                let remote_peer_bytes = (*local_peer_id).to_bytes();
                let nonce_bytes = nonce.to_le_bytes();
                let mut pow_buf = Vec::with_capacity(
                    size_of::<u64>()
                        + local_peer_bytes.len()
                        + remote_peer_bytes.len()
                        + message.len(),
                );
                pow_buf.extend_from_slice(&nonce_bytes[..]);
                pow_buf.extend_from_slice(&local_peer_bytes[..]);
                pow_buf.extend_from_slice(&remote_peer_bytes[..]);
                pow_buf.extend_from_slice(&message[..]);
                builder.verify_bytes(&pow_buf[..], pow)
            }
            NockchainRequest::Gossip { message: _ } => Ok(()),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
/// Responses to Nockchain requests
pub enum NockchainResponse {
    /// The requested block or raw-tx
    Result { message: ByteBuf },
    /// If the request was a gossip, no actual response is needed
    Ack { acked: bool },
}

impl NockchainResponse {
    fn new_response_result(message: impl AsRef<[u8]>) -> NockchainResponse {
        let message_bytes: &[u8] = message.as_ref();
        let message_bytebuf = ByteBuf::from(message_bytes.to_vec());
        NockchainResponse::Result {
            message: message_bytebuf,
        }
    }
}

// fn emit_fail2ban(peer_ip: u128) -> Result<(), NockAppError> {
//     // get peer ip address
//     let peer_ip = peer_id.to_base58();
// }
//
async fn send_timer_poke(
    guard: tokio::sync::OwnedMutexGuard<()>,
    traffic_cop: traffic_cop::TrafficCop,
    metrics: Arc<NockchainP2PMetrics>,
) -> Result<(), NockAppError> {
    let mut slab = NounSlab::new();
    let timer_noun = T(&mut slab, &[D(tas!(b"command")), D(tas!(b"timer")), D(0)]);
    slab.set_root(timer_noun);
    let wire = nockapp::drivers::timer::TimerWire::Tick.to_wire();
    let (timing, timing_rx) = tokio::sync::oneshot::channel();
    let permit = traffic_cop.poke_high_priority_permit().await?;
    permit.poke_high_priority(wire, slab, Some(timing)).await?;
    let elapsed = timing_rx.await?;
    let _ = metrics.timer_poke_time.add_timing(&elapsed);
    drop(guard);
    Ok(())
}

async fn handle_effect(
    noun_slab: NounSlab,
    swarm_tx: mpsc::Sender<SwarmAction>,
    equix_builder: equix::EquiXBuilder,
    local_peer_id: PeerId,
    connected_peers: Vec<PeerId>,
    message_tracker: Arc<Mutex<MessageTracker>>,
    metrics: Arc<NockchainP2PMetrics>,
) -> Result<(), NockAppError> {
    match EffectType::from_noun_slab(&noun_slab) {
        EffectType::Gossip => {
            // Get the tail of the gossip effect (after %gossip head)
            let mut tail_slab = NounSlab::new();
            let gossip_cell = unsafe { noun_slab.root().as_cell()?.tail() };

            // Skip version number
            // TODO: add version negotiation, reject unknown/incompatible versions
            let data_cell = gossip_cell.as_cell()?.tail();
            tail_slab.copy_into(data_cell);

            // Check if this is a heard-block gossip
            let gossip_noun = unsafe { tail_slab.root() };
            if let Ok(data_cell) = gossip_noun.as_cell() {
                if data_cell.head().eq_bytes(b"heard-block") {
                    trace!("Gossip effect for heard-block, clearing block and elders cache");
                    let mut tracker = message_tracker.lock().await;
                    tracker.block_cache.clear();
                    tracker.elders_cache.clear();
                    tracker.elders_negative_cache.clear();
                }
            }

            let gossip_request = NockchainRequest::new_gossip(&tail_slab);
            debug!("Gossiping to {} peers", connected_peers.len());
            for peer_id in connected_peers.clone() {
                let gossip_request_clone = gossip_request.clone();
                swarm_tx
                    .send(SwarmAction::SendRequest {
                        peer_id,
                        request: gossip_request_clone,
                    })
                    .await
                    .map_err(|_e| NockAppError::OtherError)?;
            }
        }
        EffectType::Request => {
            // Extract request details to check if it's a peer-specific request
            let request_cell = unsafe { noun_slab.root().as_cell()? };
            let request_body = request_cell.tail().as_cell()?;
            let request_type = request_body.head().as_direct()?;

            let target_peers = if request_type.data() == tas!(b"block") {
                let block_cell = request_body.tail().as_cell()?;
                if block_cell.head().eq_bytes(b"elders") {
                    // Extract peer ID from elders request
                    let elders_cell = block_cell.tail().as_cell()?;
                    let peer_id_atom = elders_cell.tail().as_atom()?;
                    if let Ok(bytes) = peer_id_atom.to_bytes_until_nul() {
                        if let Ok(peer_id) = PeerId::from_bytes(&bytes) {
                            vec![peer_id]
                        } else {
                            connected_peers.clone()
                        }
                    } else {
                        connected_peers.clone()
                    }
                } else {
                    connected_peers.clone()
                }
            } else {
                connected_peers.clone()
            };

            if request_type.data() == tas!(b"raw-tx") {
                if let Ok(raw_tx_cell) = request_body.tail().as_cell() {
                    if raw_tx_cell.head().eq_bytes(b"by-id") {
                        trace!("Requesting raw transaction by ID, removing ID from seen set");
                        let tx_id = tip5_hash_to_base58(raw_tx_cell.tail())?;
                        let mut tracker_guard = message_tracker.clone().lock_owned().await;
                        tracker_guard.seen_txs.remove(&tx_id);
                    }
                }
            }

            debug!("Sending request to {} peers", target_peers.len());

            for peer_id in target_peers {
                let local_peer_id_clone = local_peer_id;
                let mut equix_builder_clone = equix_builder.clone();
                let request = NockchainRequest::new_request(
                    &mut equix_builder_clone, &local_peer_id_clone, &peer_id, &noun_slab,
                );
                swarm_tx
                    .send(SwarmAction::SendRequest { peer_id, request })
                    .await
                    .map_err(|_e| NockAppError::OtherError)?;
            }
        }
        EffectType::LiarPeer => {
            let effect_cell = unsafe { noun_slab.root().as_cell()? };
            let peer_id_atom = effect_cell.tail().as_atom().map_err(|_| {
                NockAppError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Expected peer ID atom in liar-peer effect",
                ))
            })?;

            let bytes = peer_id_atom
                .to_bytes_until_nul()
                .expect("failed to strip null bytes");
            let peer_id_str = String::from_utf8(bytes).map_err(|_| {
                NockAppError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Invalid UTF-8 in peer ID",
                ))
            })?;

            let peer_id = PeerId::from_str(&peer_id_str).map_err(|_| {
                NockAppError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Invalid peer ID format",
                ))
            })?;

            swarm_tx
                .send(SwarmAction::BlockPeer { peer_id })
                .await
                .map_err(|_| NockAppError::OtherError)?;
        }
        EffectType::LiarBlockId => {
            let effect_cell = unsafe { noun_slab.root().as_cell()? };
            let block_id = effect_cell.tail();

            // Add the bad block ID
            let mut tracker = message_tracker.lock().await;
            let peers_to_ban = tracker.process_bad_block_id(block_id)?;

            // Ban each peer that sent this block
            for peer_id in peers_to_ban {
                swarm_tx
                    .send(SwarmAction::BlockPeer { peer_id })
                    .await
                    .map_err(|_| NockAppError::OtherError)?;
            }
        }
        EffectType::Track => {
            let effect_cell = unsafe { noun_slab.root().as_cell()? };
            let track_cell = effect_cell.tail().as_cell()?;
            let action = track_cell.head();

            if action.eq_bytes(b"add") {
                // Handle [%track %add block-id peer-id]
                let data_cell = track_cell.tail().as_cell()?;
                let block_id = data_cell.head();
                let peer_id_atom = data_cell.tail().as_atom()?;

                // Convert peer_id from base58 string to PeerId
                let Ok(peer_id) = PeerId::from_noun(peer_id_atom.as_noun()) else {
                    return Err(NockAppError::OtherError);
                };

                // Add to message tracker
                let mut tracker = message_tracker.lock().await;
                tracker.track_block_id_and_peer(block_id, peer_id)?;
            } else if action.eq_bytes(b"remove") {
                // Handle [%track %remove block-id]
                let block_id = track_cell.tail();

                // Remove from message tracker
                let mut tracker = message_tracker.lock().await;
                tracker.remove_block_id(block_id)?;
            } else {
                return Err(NockAppError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Invalid track action",
                )));
            }
        }
        EffectType::Seen => {
            let effect_cell = unsafe { noun_slab.root().as_cell()? };
            let seen_cell = effect_cell.tail().as_cell()?;
            let seen_type = seen_cell.head();

            if seen_type.eq_bytes(b"block") {
                let seen_pq = seen_cell.tail().as_cell()?;
                let block_id = seen_pq.head().as_cell()?;
                let mut tracker = message_tracker.lock().await;
                let block_id_str = tip5_hash_to_base58(block_id.as_noun())
                    .expect("failed to convert block ID to base58");
                trace!("seen block id: {:?}", &block_id_str);
                tracker.seen_blocks.insert(block_id_str);

                if let Ok(block_height_unit_cell) = seen_pq.tail().as_cell() {
                    let block_height = block_height_unit_cell.tail().as_atom()?.as_u64()?;
                    if tracker.first_negative <= block_height {
                        metrics.highest_block_height_seen.swap(block_height as f64);
                        tracker.first_negative = block_height + 1;
                        trace!("Setting tracker.first_negative to {:?}", tracker.first_negative);

                        // Check if we should clear the tx cache
                        if block_height
                            >= tracker.last_tx_cache_clear_height + tracker.seen_tx_clear_interval
                        {
                            debug!("Clearing seen_txs cache at block height {}", block_height);
                            debug!("Cache before clearing: {:?}", tracker.seen_txs);
                            tracker.seen_txs.clear();
                            tracker.last_tx_cache_clear_height = block_height;
                        }
                    }
                }
            } else if seen_type.eq_bytes(b"tx") {
                let tx_id = seen_cell.tail().as_cell()?;
                let mut tracker = message_tracker.lock().await;
                let tx_id_str = tip5_hash_to_base58(tx_id.as_noun())
                    .expect("failed to convert tx ID to base58");
                trace!("seen tx id: {:?}", &tx_id_str);
                tracker.seen_txs.insert(tx_id_str);
            }
        }
        EffectType::Unknown => {
            //  This isn't unexpected - any effect that this driver doesn't handle
            //  will hit this case.
        }
    }
    Ok(())
}

// TODO: Wrap some of this up.
#[allow(clippy::too_many_arguments)]
async fn handle_request_response(
    peer: PeerId,
    connection_id: ConnectionId,
    message: request_response::Message<NockchainRequest, NockchainResponse>,
    swarm_tx: mpsc::Sender<SwarmAction>,
    equix_builder: &mut equix::EquiXBuilder,
    local_peer_id: PeerId,
    traffic: traffic_cop::TrafficCop,
    metrics: Arc<NockchainP2PMetrics>,
    message_tracker: Arc<Mutex<MessageTracker>>,
    request_high_threshold: u64,
) -> Result<(), NockAppError> {
    trace!("handle_request_response peer: {peer}");
    match message {
        Request {
            request, channel, ..
        } => {
            let Ok(()) = request.verify_pow(equix_builder, &local_peer_id, &peer) else {
                warn!("bad libp2p powork from {peer}, blocking!");
                swarm_tx
                    .send(SwarmAction::BlockPeer { peer_id: peer })
                    .await
                    .map_err(|_| NockAppError::OtherError)?;
                return Ok(());
            };
            trace!("handle_request_response: powork verified");
            let addr = {
                message_tracker
                    .lock()
                    .await
                    .connection_address(connection_id)
            };
            if let Some(addr) = addr {
                let addr_str = addr.to_string();
                debug!("Request received from peer at address {addr_str} with id {peer}");
                let mut ip4_addr = None;
                if let Some(proto) = addr.iter().next() {
                    if let Protocol::Ip4(ip4) = proto {
                        ip4_addr = Some(ip4);
                    }
                }
                if let Some(ip4) = ip4_addr {
                    let threshold_exceeded = message_tracker
                        .lock()
                        .await
                        .requested(ip4, request_high_threshold);
                    if let Some(count) = threshold_exceeded {
                        warn!("IP address {ip4} exceeded the request-per-interval threshold with {count} requests");
                    }
                }
            } else {
                debug!("Request received but connection not tracked. Bug?");
            }
            let mut request_slab = NounSlab::new();
            match request {
                NockchainRequest::Request {
                    pow: _,
                    nonce: _,
                    message,
                } => {
                    trace!("handle_request_response: Request received");
                    let message_bytes = Bytes::from(message.to_vec());
                    let request_noun = request_slab.cue_into(message_bytes)?;

                    let data_request = NockchainDataRequest::from_noun(request_noun)?;

                    let cached = {
                        let cache_result = {
                            let mut tracker = message_tracker.lock().await;
                            tracker.check_cache(data_request.clone(), &metrics).await
                        };
                        match cache_result {
                            Ok(CacheResponse::Cached(slab)) => {
                                trace!("Found cached response for request");
                                Some(slab)
                            }
                            Ok(CacheResponse::NegativeCached) => {
                                trace!("Negative-cached response for request");
                                return Ok(()); // short-circuit
                            }
                            Ok(CacheResponse::NotCached) => None,
                            Err(e) => {
                                warn!("Error checking block cache: {e:?}");
                                None
                            }
                        }
                    };

                    let (scry_res_slab, cache_hit) = if let Some(cache_result) = cached {
                        trace!("found cached response for request");
                        (cache_result, true)
                    } else {
                        let scry_slab = request_to_scry_slab(data_request.clone())?;
                        let Some(scry_res_slab) = (match traffic.peek(scry_slab).await {
                            Ok(Some(res_slab)) => {
                                metrics.requests_peeked_some.increment();
                                Some(res_slab)
                            }
                            Ok(None) => {
                                metrics.requests_peeked_none.increment();
                                trace!(
                                "No data found for incoming request from: {}, request type: {:?}",
                                peer,
                                request_noun.as_cell()?.tail().as_cell().map(|c| c.head())
                            );
                                None
                            }
                            Err(NockAppError::MPSCFullError(act)) => {
                                metrics.requests_dropped.increment();
                                trace!(
                                    "handle_request_response: Request dropped due to backpressure"
                                );
                                Err(NockAppError::MPSCFullError(act))?
                            }
                            Err(err) => {
                                match data_request {
                                    NockchainDataRequest::BlockByHeight(height) => {
                                        debug!("Peek error getting block at height: {:?}", height);
                                        metrics.requests_erred_block_by_height.increment();
                                    }
                                    NockchainDataRequest::EldersById(ref id, _, _) => {
                                        debug!("Peek error getting elders of id: {:?}", id);
                                        metrics.requests_erred_elders_by_id.increment();
                                    }
                                    NockchainDataRequest::RawTransactionById(ref id, _) => {
                                        debug!("Peek error getting raw tx with id: {:?}", &id);
                                        metrics.requests_erred_raw_tx_by_id.increment();
                                    }
                                }
                                trace!("handle_request_response: Error getting response");
                                Err(err)?
                            }
                        }) else {
                            return Ok(());
                        };
                        (scry_res_slab, false)
                    };

                    let mut res_slab = NounSlab::new();
                    let response = match data_request {
                        NockchainDataRequest::BlockByHeight(height) => {
                            let scry_res = unsafe { scry_res_slab.root() };
                            match create_scry_response(scry_res, "heard-block", &mut res_slab) {
                                Left(()) => {
                                    trace!("No data found for incoming block by-height request");
                                    return Ok(());
                                }
                                Right(result) => {
                                    if !cache_hit {
                                        let mut tracker = message_tracker.lock().await;
                                        tracker.block_cache.insert(height, scry_res_slab.clone());
                                    }
                                    result?
                                }
                            }
                        }
                        NockchainDataRequest::EldersById(id, _, _) => {
                            let scry_res = unsafe { scry_res_slab.root() };
                            match create_scry_response(scry_res, "heard-elders", &mut res_slab) {
                                Left(()) => {
                                    trace!("No data found for incoming elders request");
                                    let mut tracker = message_tracker.lock().await;
                                    tracker.elders_negative_cache.insert(id.clone());
                                    return Ok(());
                                }
                                Right(result) => {
                                    if !cache_hit {
                                        let mut tracker = message_tracker.lock().await;
                                        tracker.elders_cache.insert(id, scry_res_slab.clone());
                                    }
                                    result?
                                }
                            }
                        }
                        NockchainDataRequest::RawTransactionById(ref id, _) => {
                            let scry_res = unsafe { scry_res_slab.root() };
                            match create_scry_response(scry_res, "heard-tx", &mut res_slab) {
                                Left(()) => {
                                    trace!("No data found for incoming raw-tx request");
                                    return Ok(());
                                }
                                Right(result) => {
                                    if !cache_hit {
                                        let mut tracker = message_tracker.lock().await;
                                        trace!("cacheing tx request by id={:?}", id);
                                        tracker.tx_cache.insert(id.clone(), scry_res_slab.clone());
                                    }
                                    result?
                                }
                            }
                        }
                    };
                    swarm_tx
                        .send(SwarmAction::SendResponse { channel, response })
                        .await
                        .map_err(|_| NockAppError::OtherError)?;
                }
                NockchainRequest::Gossip { message } => {
                    trace!("handle_request_response: Gossip received");
                    let message_bytes = Bytes::from(message.to_vec());
                    let request_noun = request_slab.cue_into(message_bytes)?;
                    request_slab.set_root(request_noun);
                    trace!("handle_request_response: Gossip noun parsed");

                    let send_response: tokio::task::JoinHandle<Result<(), NockAppError>> =
                        tokio::spawn(async move {
                            let response = NockchainResponse::Ack { acked: true };
                            swarm_tx
                                .send(SwarmAction::SendResponse { channel, response })
                                .await
                                .map_err(|_| NockAppError::OtherError)?;
                            Ok(())
                        });

                    let poke_kernel = tokio::task::spawn(async move {
                        let permit = traffic.poke_high_priority_permit().await?;
                        let gossip = NockchainFact::from_noun_slab(&request_slab)?;
                        match gossip {
                            NockchainFact::HeardBlock(ref id, _) => {
                                let tracker = message_tracker.lock().await;
                                if tracker.seen_blocks.contains(id) {
                                    trace!("Block already seen, not processing: {:?}", id);
                                    metrics.block_seen_cache_hits.increment();
                                    return Ok(());
                                } else {
                                    trace!("block not seen, processing: {:?}", id);
                                    metrics.block_seen_cache_misses.increment();
                                }
                            }
                            NockchainFact::HeardTx(ref id, _) => {
                                let tracker = message_tracker.lock().await;
                                if tracker.seen_txs.contains(id) {
                                    trace!("Tx already seen, not processing: {:?}", id);
                                    metrics.tx_seen_cache_hits.increment();
                                    return Ok(());
                                } else {
                                    trace!("tx not seen, processing: {:?}", id);
                                    metrics.tx_seen_cache_misses.increment();
                                }
                            }
                            NockchainFact::HeardElders(..) => {
                                warn!("Heard elders over gossip, should not happen!");
                                return Ok(());
                            }
                        }

                        let wire = Libp2pWire::Gossip(peer);

                        trace!(
                            "Poking kernel with wire: {:?} noun: {:?}",
                            wire,
                            nockvm::noun::FullDebugCell(unsafe { &request_slab.root().as_cell()? })
                        );

                        let poke = gossip.fact_poke();
                        let (timing, timing_rx) = tokio::sync::oneshot::channel();
                        let poke_result = permit
                            .poke_high_priority(wire.to_wire(), poke.clone(), Some(timing))
                            .await;
                        let elapsed = timing_rx.await?;
                        match gossip {
                            NockchainFact::HeardBlock(_, _) => {
                                metrics.heard_block_poke_time.add_timing(&elapsed);
                            }
                            NockchainFact::HeardTx(_, _) => {
                                metrics.heard_tx_poke_time.add_timing(&elapsed);
                            }
                            _ => {}
                        }
                        match poke_result {
                            Ok(PokeResult::Ack) => match gossip {
                                NockchainFact::HeardBlock(..) => {
                                    metrics.gossip_acked_heard_block.increment();
                                }
                                NockchainFact::HeardTx(..) => {
                                    metrics.gossip_acked_heard_tx.increment();
                                }
                                NockchainFact::HeardElders(..) => {
                                    metrics.gossip_acked_heard_elders.increment();
                                }
                            },
                            Ok(PokeResult::Nack) => {
                                match gossip {
                                    NockchainFact::HeardBlock(height, _) => {
                                        debug!(
                                            "Poke gossip nacked for heard-block at height: {:?}",
                                            height
                                        );
                                        metrics.gossip_nacked_heard_block.increment();
                                    }
                                    NockchainFact::HeardTx(id, _) => {
                                        debug!("Poke gossip nacked for heard-tx id: {:?}", id);
                                        metrics.gossip_nacked_heard_tx.increment();
                                    }
                                    NockchainFact::HeardElders(oldest, block_ids, _) => {
                                        debug!(
                                            "Poke heard-elders nacked for block height {:?} with ancestors {:?}",
                                            oldest, block_ids
                                        );
                                        metrics.gossip_nacked_heard_elders.increment();
                                    }
                                };
                                trace!("handle_request_response: gossip poke nacked");
                                return Ok(());
                            }
                            Err(NockAppError::MPSCFullError(act)) => {
                                metrics.gossip_dropped.increment();
                                trace!(
                                "handle_request_response: gossip poke dropped due to backpressure"
                            );
                                return Err(NockAppError::MPSCFullError(act));
                            }
                            Err(err) => {
                                match gossip {
                                    NockchainFact::HeardBlock(height, _) => {
                                        debug!(
                                            "Poke gossip erred for heard-block at height: {:?}",
                                            height
                                        );
                                        metrics.gossip_erred_heard_block.increment();
                                    }
                                    NockchainFact::HeardTx(id, _) => {
                                        debug!("Poke gossip erred for heard-tx id: {:?}", id);
                                        metrics.gossip_erred_heard_tx.increment();
                                    }
                                    NockchainFact::HeardElders(oldest, block_ids, _) => {
                                        debug!(
                                            "Poke heard-elders erred for block height {:?} with ancestors {:?}",
                                            oldest, block_ids
                                        );
                                        metrics.gossip_erred_heard_elders.increment();
                                    }
                                };
                                trace!("handle_request_response: Poke errored");
                                return Err(err);
                            }
                        };
                        trace!("handle_request_response: Poke successful");
                        Ok(())
                    });
                    send_response.await??;
                    poke_kernel.await??;
                }
            }
        }
        Response { response, .. } => match response {
            NockchainResponse::Result { message } => {
                let permit = traffic.poke_high_priority_permit().await?;
                trace!("handle_request_response: Response result received");
                let mut response_slab = NounSlab::new();
                let message_bytes = Bytes::from(message.to_vec());
                let response_noun = response_slab.cue_into(message_bytes)?;
                response_slab.set_root(response_noun);

                trace!("Received response from peer");

                let response_cell = response_noun.as_cell()?;
                if response_cell.head().eq_bytes(b"heard-elders") {
                    let elders_p_cell = response_cell.tail().as_cell()?;
                    let elders_list_cell = elders_p_cell.tail().as_cell()?;
                    let top_block_id = tip5_hash_to_base58(elders_list_cell.head())?;

                    let mut tracker = message_tracker.lock().await;
                    if tracker.seen_elders.contains(&top_block_id) {
                        return Ok(());
                    } else {
                        tracker.seen_elders.insert(top_block_id);
                    }
                }

                trace!(
                    "Response noun: {:?}",
                    nockvm::noun::FullDebugCell(&response_noun.as_cell()?)
                );

                let response = NockchainFact::from_noun_slab(&response_slab)?;
                match response {
                    NockchainFact::HeardBlock(ref id, _) => {
                        let tracker = message_tracker.lock().await;
                        if tracker.seen_blocks.contains(id) {
                            trace!("Block already seen, not processing: {:?}", id);
                            metrics.block_seen_cache_hits.increment();
                            return Ok(());
                        } else {
                            trace!("block not seen, processing: {:?}", id);
                            metrics.block_seen_cache_misses.increment();
                        }
                    }
                    NockchainFact::HeardTx(..) => {}
                    NockchainFact::HeardElders(..) => {}
                }

                let wire = Libp2pWire::Response(peer);
                let poke_slab = response.fact_poke();

                let (timing, timing_rx) = tokio::sync::oneshot::channel();
                let poke_result = permit
                    .poke_high_priority(wire.to_wire(), poke_slab.clone(), Some(timing))
                    .await;
                let elapsed = timing_rx.await?;

                if response_cell.head().eq_bytes(b"heard-block") {
                    metrics.heard_block_poke_time.add_timing(&elapsed);
                } else if response_cell.head().eq_bytes(b"heard-tx") {
                    metrics.heard_tx_poke_time.add_timing(&elapsed);
                } else if response_cell.head().eq_bytes(b"heard-elders") {
                    metrics.heard_elders_poke_time.add_timing(&elapsed);
                }

                match poke_result {
                    Ok(PokeResult::Ack) => match response {
                        NockchainFact::HeardBlock(..) => {
                            metrics.responses_acked_heard_block.increment();
                        }
                        NockchainFact::HeardTx(..) => {
                            metrics.responses_acked_heard_tx.increment();
                        }
                        NockchainFact::HeardElders(..) => {
                            metrics.responses_acked_heard_elders.increment();
                        }
                    },
                    Ok(PokeResult::Nack) => {
                        match response {
                            NockchainFact::HeardBlock(..) => {
                                metrics.responses_nacked_heard_block.increment();
                            }
                            NockchainFact::HeardTx(id, _) => {
                                debug!("Poke response nacked for heard-tx id: {:?}", id);
                                metrics.responses_nacked_heard_tx.increment();
                            }
                            NockchainFact::HeardElders(oldest, block_ids, _) => {
                                debug!(
                                    "Poke response heard-elders nacked for block height {:?} with ancestors {:?}",
                                    oldest, block_ids
                                );
                                metrics.responses_nacked_heard_elders.increment();
                            }
                        }
                        trace!("handle_request_response: Poke failed");
                        return Ok(());
                    }
                    Err(NockAppError::MPSCFullError(act)) => {
                        trace!("handle_request_response: Response dropped due to backpressure.");
                        metrics.responses_dropped.increment();
                        return Err(NockAppError::MPSCFullError(act));
                    }
                    Err(_) => {
                        trace!("handle_request_response: Error poking with response");
                        match response {
                            NockchainFact::HeardBlock(height, _) => {
                                debug!(
                                    "Poke response error for heard-block at height: {:?}",
                                    height
                                );
                                metrics.responses_erred_heard_block.increment();
                            }
                            NockchainFact::HeardTx(id, _) => {
                                debug!("Poke response error for heard-tx id: {:?}", id);
                                metrics.responses_erred_heard_tx.increment();
                            }
                            NockchainFact::HeardElders(oldest, block_ids, _) => {
                                debug!(
                                    "Poke response error for heard-elders for block height {:?} with ancestors {:?}",
                                    oldest, block_ids
                                );
                                metrics.responses_erred_heard_elders.increment();
                            }
                        }
                        trace!("Error sending poke")
                    }
                }
                trace!("handle_request_response: Poke successful");
            }
            NockchainResponse::Ack { acked } => {
                trace!("Received acknowledgement from peer {}", peer);
                if !acked {
                    warn!("Peer {} did not acknowledge the response", peer);
                }
            }
        },
    }
    Ok(())
}

async fn log_peer_status(
    swarm: &mut Swarm<NockchainBehaviour>,
    metrics: &NockchainP2PMetrics,
) -> usize {
    let connected_peer_count = {
        info!("Logging current peer status...");
        let connected_peers: Vec<_> = swarm.connected_peers().cloned().collect();
        let peer_count = connected_peers.len();

        if peer_count == 0 {
            warn!(
                connected_peers = peer_count,
                peers = ?connected_peers.iter().map(|p| p.to_base58()).collect::<Vec<_>>(),
                "No current peers connected!"
            );
        } else {
            info!(
                connected_peers = peer_count,
                peers = ?connected_peers.iter().map(|p| p.to_base58()).collect::<Vec<_>>(),
                "Current peer status"
            );
        }

        let _ = metrics.active_peer_connections.swap(peer_count as f64);
        peer_count
    };

    // Count peers in the routing table by iterating through k-buckets
    let mut routing_table_size = 0;
    for bucket in swarm.behaviour_mut().kad.kbuckets() {
        routing_table_size += bucket.num_entries();
    }

    if routing_table_size == 0 {
        warn!(
            routing_table_size = routing_table_size,
            "Routing table is empty!"
        );
    } else {
        info!(
            routing_table_size = routing_table_size,
            "Routing table has {} entries", routing_table_size
        );
    };
    connected_peer_count
}

/// Converts a request noun into a scry path that can be used to query the Nockchain state.
///
/// The request noun is expected to be in the format:
/// `[%request [type data]]` where:
/// - `type` can be either "block" or "raw-tx"
/// - For "block" type:
///   - `data` can be either `[%by-height height]` or `[%elders block-id peer-id]`
/// - For "raw-tx" type:
///   - `data` must be `[%by-id id]`
///
/// # Arguments
/// * `noun` - The request noun to convert
///
/// # Returns
/// * `Ok(NounSlab)` - A noun slab containing the constructed scry path
/// * `Err(NockAppError)` - If the request format is invalid or unknown
///
/// # Examples
/// For a block by height request:
/// [%request [%block [%by-height 123]]] -> [%heavy-n 123 0]
/// For a block by id request:
/// [%request [%block [%elders [1 2 3 4 5] abcDEF]]] -> [%elders base58-block-id peer-id 0]
/// For a raw transaction request:
/// [%request [%raw-tx [%by-id [1 2 3 4 5]]]] -> [%raw-transaction base58-tx-id 0]
fn request_to_scry_slab(request: NockchainDataRequest) -> Result<NounSlab, NockAppError> {
    match request {
        NockchainDataRequest::BlockByHeight(height) => {
            debug!("Requesting block by height: {}", height);
            let mut slab = NounSlab::new();
            let height_atom = Noun::from_atom(Atom::new(&mut slab, height));
            let noun = T(&mut slab, &[D(tas!(b"heavy-n")), height_atom, D(0)]);
            slab.set_root(noun);
            Ok(slab)
        }
        NockchainDataRequest::EldersById(str, _, _) => {
            debug!("Requesting elders by ID: {}", str);
            let mut slab = NounSlab::new();
            let id_atom = Atom::from_value(&mut slab, str)?;
            let noun = T(&mut slab, &[D(tas!(b"elders")), id_atom.as_noun(), D(0)]);
            slab.set_root(noun);
            Ok(slab)
        }
        NockchainDataRequest::RawTransactionById(str, _) => {
            debug!("Requesting raw transaction by ID: {}", str);
            let mut slab = NounSlab::new();
            let raw_tx_tag = make_tas(&mut slab, "raw-transaction").as_noun();
            let id_atom = Atom::from_value(&mut slab, str)?;
            let noun = T(&mut slab, &[raw_tx_tag, id_atom.as_noun(), D(0)]);
            slab.set_root(noun);
            Ok(slab)
        }
    }
}

/// Creates a response to a scry request by processing the scry result noun
///
/// # Arguments
/// * `scry_res` - The noun containing the scry result to process
/// * `heard_type` - The type of request that was heard (as a string)
/// * `res_slab` - Mutable reference to the noun slab for storing the response
///
/// # Returns
/// Either:
/// - `Left(())` if the scry path was bad or nothing was found
/// - `Right(Ok(NockchainResponse))` containing the successful response
/// - `Right(Err(NockAppError))` if there was an error processing the result
fn create_scry_response(
    scry_res: &Noun,
    heard_type: &str,
    res_slab: &mut NounSlab,
) -> Either<(), Result<NockchainResponse, NockAppError>> {
    match ScryResult::from(scry_res) {
        ScryResult::BadPath => {
            warn!("Bad scry path");
            Left(())
        }
        ScryResult::Nothing => {
            trace!("Nothing found at scry path");
            Left(())
        }
        ScryResult::Some(x) => {
            let nouns = vec![x];
            if let Ok(response_noun) = prepend_tas(res_slab, heard_type, nouns) {
                res_slab.set_root(response_noun);
                Right(Ok(NockchainResponse::new_response_result(res_slab.jam())))
            } else {
                error!("Failed to prepend tas to response noun");
                Right(Err(NockAppError::OtherError))
            }
        }
        ScryResult::Invalid => {
            error!("Invalid scry result");
            Right(Err(NockAppError::OtherError))
        }
    }
}

/// Prepends a @tas to one or more Nouns.
///
/// # Arguments
/// * `slab` - The NounSlab containing the noun
/// * `tas_str` - The tag string to prepend
/// * `nouns` - The Nouns to include
///
/// # Returns
/// The noun with @tas prepended
fn prepend_tas(slab: &mut NounSlab, tas_str: &str, nouns: Vec<Noun>) -> Result<Noun, NockAppError> {
    let tas_atom = Atom::from_value(slab, tas_str)?;

    // Create a cell with the tag and all provided nouns
    let mut cell_elements = Vec::with_capacity(nouns.len() + 1);
    cell_elements.push(tas_atom.as_noun());
    cell_elements.extend(nouns);

    Ok(T(slab, &cell_elements))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use nockapp::noun::slab::NounSlab;
    use nockvm::noun::{D, T};
    use nockvm_macros::tas;

    use super::*;

    pub static LIBP2P_CONFIG: LazyLock<LibP2PConfig> = LazyLock::new(|| LibP2PConfig::default());

    #[test]
    #[cfg_attr(miri, ignore)] // ibig has a memory leak so miri fails this test
    fn test_request_to_scry_slab() {
        // Test block by-height request
        {
            let mut slab: NounSlab = NounSlab::new();
            let height = 123u64;
            let by_height_tas = make_tas(&mut slab, "by-height");
            let by_height = T(&mut slab, &[by_height_tas.as_noun(), D(height)]);
            let block_cell = T(&mut slab, &[D(tas!(b"block")), by_height]);
            let request = T(&mut slab, &[D(tas!(b"request")), block_cell]);
            slab.set_root(request);

            let data_request = NockchainDataRequest::from_noun(request)
                .expect("Failed to create request from noun");

            let result_slab = request_to_scry_slab(data_request).unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            let result = unsafe { result_slab.root() };

            assert!(result.is_cell());
            let result_cell = result.as_cell().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            assert!(result_cell.head().eq_bytes(b"heavy-n"));

            // Get the tail cell and check its components
            let tail_cell = result_cell.tail().as_cell().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            let height_atom = tail_cell.head().as_atom().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            assert_eq!(
                height_atom.as_u64().unwrap_or_else(|err| {
                    panic!(
                        "Panicked with {err:?} at {}:{} (git sha: {:?})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA")
                    )
                }),
                height
            );
            let tail_atom = tail_cell.tail().as_atom().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            assert_eq!(
                tail_atom.as_u64().unwrap_or_else(|err| {
                    panic!(
                        "Panicked with {err:?} at {}:{} (git sha: {:?})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA")
                    )
                }),
                0
            );
        }

        // Test invalid request (not a cell)
        {
            let mut slab: NounSlab = NounSlab::new();
            slab.set_root(D(123));
            let result = NockchainDataRequest::from_noun(*unsafe { slab.root() })
                .and_then(|r| request_to_scry_slab(r));
            assert!(result.is_err());
        }

        // Test elders request
        {
            let mut slab: NounSlab = NounSlab::new();
            // Create a 5-tuple [1 2 3 4 5] for the block ID
            let five_tuple = T(&mut slab, &[D(1), D(2), D(3), D(4), D(5)]);

            // Create a random peer ID and store its bytes
            let peer_id = PeerId::random();
            let peer_id_atom = Atom::from_value(&mut slab, peer_id.to_base58()).unwrap();

            let elders_cell = T(&mut slab, &[five_tuple, peer_id_atom.as_noun()]);
            let elders_tas = D(tas!(b"elders"));
            let inner_cell = T(&mut slab, &[elders_tas, elders_cell]);
            let block_cell = T(&mut slab, &[D(tas!(b"block")), inner_cell]);
            let request = T(&mut slab, &[D(tas!(b"request")), block_cell]);
            slab.set_root(request);

            let data_request = NockchainDataRequest::from_noun(request)
                .expect("Could not create request from noun");

            let result_slab = request_to_scry_slab(data_request).unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            let result = unsafe { result_slab.root() };

            // Verify the structure: [%elders block_id_b58 0]
            assert!(result.is_cell());
            let result_cell = result.as_cell().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });

            // Check %elders tag
            assert!(result_cell.head().eq_bytes(b"elders"));

            // Get the tail cell
            let tail_cell = result_cell.tail().as_cell().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });

            // Check block ID (should be base58 encoded)
            let block_id_atom = tail_cell.head().as_atom().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            let block_id_bytes = block_id_atom.to_bytes_until_nul().unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            let block_id_str = String::from_utf8(block_id_bytes).unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });

            // Get the expected base58 string
            let expected_b58 = tip5_hash_to_base58(five_tuple).unwrap_or_else(|_| {
                panic!(
                    "Called `expect()` at {}:{} (git sha: {})",
                    file!(),
                    line!(),
                    option_env!("GIT_SHA").unwrap_or("unknown")
                )
            });
            assert_eq!(block_id_str, expected_b58);

            // Check final 0
            assert_eq!(
                tail_cell
                    .tail()
                    .as_direct()
                    .unwrap_or_else(|err| {
                        panic!(
                            "Panicked with {err:?} at {}:{} (git sha: {:?})",
                            file!(),
                            line!(),
                            option_env!("GIT_SHA")
                        )
                    })
                    .data(),
                0
            );
        }

        // Test invalid elders request (not a cell)
        {
            let mut slab: NounSlab = NounSlab::new();
            let invalid_request = T(
                &mut slab,
                &[D(tas!(b"request")), D(tas!(b"block")), D(tas!(b"elders"))],
            );
            slab.set_root(invalid_request);

            let result = NockchainDataRequest::from_noun(invalid_request)
                .and_then(|r| request_to_scry_slab(r));
            assert!(result.is_err());
            drop(slab);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // equix uses a foreign function so miri fails this tes
    fn test_equix_pow_verification() {
        use equix::EquiXBuilder;
        // Create EquiX builder - new() doesn't return Result
        let mut builder = EquiXBuilder::new();

        // Create test peer IDs
        let local_peer_id = PeerId::random();
        let remote_peer_id = PeerId::random();

        // Create test message
        let message = ByteBuf::from(vec![1, 2, 3, 4, 5]);

        // Create valid request with correct PoW
        let valid_request =
            NockchainRequest::new_request(&mut builder, &local_peer_id, &remote_peer_id, &{
                let mut slab = NounSlab::new();
                let message_noun = Atom::from_value(&mut slab, &message[..])
                    .expect("Failed to create message atom")
                    .as_noun();
                slab.set_root(message_noun);
                slab
            });

        // Verify the valid request
        match &valid_request {
            NockchainRequest::Request {
                pow,
                nonce,
                message: _,
            } => {
                // Test successful verification
                let result = valid_request.verify_pow(
                    &mut builder, &remote_peer_id, // Note: peers are swapped for verification
                    &local_peer_id,
                );
                assert!(result.is_ok(), "Valid PoW should verify successfully");

                // Test failed verification with tampered nonce
                let tampered_request = NockchainRequest::Request {
                    pow: *pow,
                    nonce: nonce + 1, // Tamper with the nonce
                    message: message.clone(),
                };
                let result =
                    tampered_request.verify_pow(&mut builder, &remote_peer_id, &local_peer_id);
                assert!(result.is_err(), "Tampered nonce should fail verification");

                // Test failed verification with wrong peer order
                let result = valid_request.verify_pow(
                    &mut builder, &local_peer_id, // Wrong order - not swapped
                    &remote_peer_id,
                );
                assert!(result.is_err(), "Wrong peer order should fail verification");
            }
            _ => panic!("Expected Request variant"),
        }

        // Test that gossip requests always verify successfully
        let gossip_request = NockchainRequest::Gossip {
            message: message.clone(),
        };
        let result = gossip_request.verify_pow(&mut builder, &remote_peer_id, &local_peer_id);
        assert!(
            result.is_ok(),
            "Gossip requests should always verify successfully"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_liar_peer_effect() {
        use equix::EquiXBuilder;
        use tokio::sync::mpsc;

        // Create a test peer ID and convert to base58
        let peer_id = PeerId::random();
        let peer_id_base58 = peer_id.to_base58();

        // Create the liar-peer effect noun
        let mut effect_slab = NounSlab::new();
        let liar_peer_atom = Atom::from_value(&mut effect_slab, "liar-peer")
            .expect("Failed to create liar-peer atom");
        let peer_id_atom = Atom::from_value(&mut effect_slab, peer_id_base58)
            .expect("Failed to create peer ID atom");
        let effect = T(
            &mut effect_slab,
            &[liar_peer_atom.as_noun(), peer_id_atom.as_noun()],
        );
        effect_slab.set_root(effect);
        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );

        // Create channel to receive SwarmAction
        let (swarm_tx, mut swarm_rx) = mpsc::channel(1);

        // Call handle_effect with the liar-peer effect
        let result = handle_effect(
            effect_slab,
            swarm_tx,
            EquiXBuilder::new(),
            PeerId::random(), // local peer ID (not relevant for this test)
            vec![],           // connected peers (not relevant for this test)
            Arc::new(Mutex::new(MessageTracker::new(
                metrics.clone(),
                LIBP2P_CONFIG.seen_tx_clear_interval,
            ))),
            metrics,
        )
        .await;

        // Verify the function succeeded
        assert!(result.is_ok(), "handle_effect should succeed");

        // Verify that a BlockPeer action was sent with the correct peer ID
        match swarm_rx.recv().await {
            Some(SwarmAction::BlockPeer {
                peer_id: blocked_peer,
            }) => {
                assert_eq!(blocked_peer, peer_id, "Wrong peer ID was blocked");
            }
            other => panic!("Expected BlockPeer action, got {:?}", other),
        }

        // Verify no more actions were sent
        assert!(swarm_rx.try_recv().is_err(), "Should only send one action");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // ibig has a memory leak so miri fails this test
    async fn test_track_add_effect() {
        use equix::EquiXBuilder;
        use tokio::sync::mpsc;

        // Create test peer ID
        let peer_id = PeerId::random();
        let peer_id_base58 = peer_id.to_base58();

        // Create the track add effect noun
        let mut effect_slab = NounSlab::new();
        let track_atom = make_tas(&mut effect_slab, "track");
        let add_atom = make_tas(&mut effect_slab, "add");

        // Create block ID as [1 2 3 4 5]
        let block_id_tuple = T(&mut effect_slab, &[D(1), D(2), D(3), D(4), D(5)]);
        let peer_id_atom = Atom::from_value(&mut effect_slab, peer_id_base58)
            .expect("Failed to create peer ID atom");

        // Build the noun structure: [%track %add block-id peer-id]
        let data_cell = T(&mut effect_slab, &[block_id_tuple, peer_id_atom.as_noun()]);
        let add_cell = T(&mut effect_slab, &[add_atom.as_noun(), data_cell]);
        let track_cell = T(&mut effect_slab, &[track_atom.as_noun(), add_cell]);
        effect_slab.set_root(track_cell);

        // Create message tracker and other required components
        let (swarm_tx, _swarm_rx) = mpsc::channel(1);

        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );
        let message_tracker = Arc::new(Mutex::new(MessageTracker::new(
            metrics.clone(),
            LIBP2P_CONFIG.seen_tx_clear_interval,
        )));

        // Call handle_effect with the track add effect
        let result = handle_effect(
            effect_slab.clone(), // test fails if we don't clone
            swarm_tx,
            EquiXBuilder::new(),
            PeerId::random(), // local peer ID (not relevant for this test)
            vec![],           // connected peers (not relevant for this test)
            message_tracker.clone(),
            metrics,
        )
        .await;

        // Verify the function succeeded
        assert!(result.is_ok(), "handle_effect should succeed");

        // Get the expected block ID string (base58 of [1 2 3 4 5])
        let block_id_str = tip5_hash_to_base58(block_id_tuple).unwrap_or_else(|_| {
            panic!(
                "Called `expect()` at {}:{} (git sha: {})",
                file!(),
                line!(),
                option_env!("GIT_SHA").unwrap_or("unknown")
            )
        });

        // Verify the message tracker state
        let tracker = message_tracker.lock().await;

        // Check block_id_to_peers mapping
        let peers = tracker.get_peers_for_block_id(block_id_tuple);
        assert!(
            peers.contains(&peer_id),
            "Peer ID should be associated with block ID"
        );

        // Check peer_to_block_ids mapping
        let block_ids = tracker.get_block_ids_for_peer(peer_id);
        assert!(
            block_ids.contains(&block_id_str),
            "Block ID should be associated with peer ID"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // ibig has a memory leak so miri fails this test
    async fn test_track_remove_effect() {
        use equix::EquiXBuilder;
        use tokio::sync::mpsc;

        // Create test peer ID
        let peer_id = PeerId::random();

        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );
        // Create a message tracker and add an entry that we'll later remove
        let message_tracker = Arc::new(Mutex::new(MessageTracker::new(
            metrics.clone(),
            LIBP2P_CONFIG.seen_tx_clear_interval,
        )));

        // Create block ID as [1 2 3 4 5]
        let mut setup_slab: NounSlab = NounSlab::new();
        let block_id_tuple = T(&mut setup_slab, &[D(1), D(2), D(3), D(4), D(5)]);

        {
            let mut tracker = message_tracker.lock().await;
            tracker
                .track_block_id_and_peer(block_id_tuple, peer_id)
                .unwrap_or_else(|_| {
                    panic!(
                        "Called `expect()` at {}:{} (git sha: {})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA").unwrap_or("unknown")
                    )
                });

            // Verify it was added correctly
            assert!(tracker.is_tracking_block_id(block_id_tuple));
            assert!(tracker.is_tracking_peer(peer_id));
        }

        // Now create the track remove effect noun
        let mut effect_slab = NounSlab::new();
        let track_atom = make_tas(&mut effect_slab, "track");
        let remove_atom = make_tas(&mut effect_slab, "remove");

        // Copy the block ID tuple to the effect slab
        let block_id_tuple_in_effect = T(&mut effect_slab, &[D(1), D(2), D(3), D(4), D(5)]);

        // Build the noun structure: [%track %remove block-id]
        let remove_cell = T(
            &mut effect_slab,
            &[remove_atom.as_noun(), block_id_tuple_in_effect],
        );
        let track_cell = T(&mut effect_slab, &[track_atom.as_noun(), remove_cell]);
        effect_slab.set_root(track_cell);

        // Create channel for SwarmAction (not used in this test)
        let (swarm_tx, _swarm_rx) = mpsc::channel(1);

        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );

        // Call handle_effect with the track remove effect
        let result = handle_effect(
            effect_slab,
            swarm_tx,
            EquiXBuilder::new(),
            PeerId::random(), // local peer ID (not relevant for this test)
            vec![],           // connected peers (not relevant for this test)
            message_tracker.clone(),
            metrics,
        )
        .await;

        // Verify the function succeeded
        assert!(result.is_ok(), "handle_effect should succeed");

        // Verify the message tracker state after removal
        let tracker = message_tracker.lock().await;

        // Check that the block ID was removed from block_id_to_peers
        assert!(
            !tracker.is_tracking_block_id(block_id_tuple),
            "Block ID should be removed"
        );

        // Check that the peer's entry in peer_to_block_ids is also removed
        // (since this was the only block ID associated with the peer)
        assert!(
            !tracker.is_tracking_peer(peer_id),
            "Peer ID should be removed since it has no more block IDs"
        );
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // ibig has a memory leak so miri fails this test
    async fn test_liar_block_id_effect() {
        use equix::EquiXBuilder;
        use tokio::sync::mpsc;

        println!("Starting test_liar_block_id_effect");

        // Create test peer IDs
        let bad_peer1 = PeerId::random();
        let bad_peer2 = PeerId::random();
        let good_peer = PeerId::random();
        println!(
            "Created peer_ids: bad1={}, bad2={}, good={}",
            bad_peer1, bad_peer2, good_peer
        );

        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );
        // Create a message tracker and add entries
        let message_tracker = Arc::new(Mutex::new(MessageTracker::new(
            metrics.clone(),
            LIBP2P_CONFIG.seen_tx_clear_interval,
        )));

        // Create block IDs
        let mut setup_slab: NounSlab = NounSlab::new();
        // Bad block ID as [1 2 3 4 5]
        let bad_block_id = T(&mut setup_slab, &[D(1), D(2), D(3), D(4), D(5)]);
        // Good block ID as [6 7 8 9 10]
        let good_block_id = T(&mut setup_slab, &[D(6), D(7), D(8), D(9), D(10)]);
        println!("Created block_ids");

        {
            let mut tracker = message_tracker.lock().await;
            println!("Tracking block_ids and peers");

            // Associate bad_peer1 with the bad block
            tracker
                .track_block_id_and_peer(bad_block_id, bad_peer1)
                .unwrap_or_else(|_| {
                    panic!(
                        "Called `expect()` at {}:{} (git sha: {})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA").unwrap_or("unknown")
                    )
                });

            // Associate bad_peer2 with the bad block
            tracker
                .add_peer_if_tracking_block_id(bad_block_id, bad_peer2)
                .unwrap_or_else(|_| {
                    panic!(
                        "Called `expect()` at {}:{} (git sha: {})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA").unwrap_or("unknown")
                    )
                });

            // Associate good_peer with a different block
            tracker
                .track_block_id_and_peer(good_block_id, good_peer)
                .unwrap_or_else(|_| {
                    panic!(
                        "Called `expect()` at {}:{} (git sha: {})",
                        file!(),
                        line!(),
                        option_env!("GIT_SHA").unwrap_or("unknown")
                    )
                });

            // Verify tracking is working
            assert!(tracker.is_tracking_block_id(bad_block_id));
            assert!(tracker.is_tracking_block_id(good_block_id));
            assert!(tracker.is_tracking_peer(bad_peer1));
            assert!(tracker.is_tracking_peer(bad_peer2));
            assert!(tracker.is_tracking_peer(good_peer));
            println!("Verified tracking is working");
        }

        // Now create the liar-block-id effect noun for the bad block
        let mut effect_slab = NounSlab::new();
        let liar_block_id_atom = Atom::from_value(&mut effect_slab, "liar-block-id")
            .expect("Failed to create liar-block-id atom");

        // Copy the bad block ID tuple to the effect slab
        let bad_block_id_in_effect = T(&mut effect_slab, &[D(1), D(2), D(3), D(4), D(5)]);

        // Build the noun structure: [%liar-block-id bad-block-id]
        let effect = T(
            &mut effect_slab,
            &[liar_block_id_atom.as_noun(), bad_block_id_in_effect],
        );
        effect_slab.set_root(effect);
        println!("Created liar-block-id effect");

        // Create channel for SwarmAction
        let (swarm_tx, mut swarm_rx) = mpsc::channel(10); // Increased capacity for multiple actions
        println!("Created swarm channel");

        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );

        // Call handle_effect with the liar-block-id effect
        println!("Calling handle_effect");
        let result = handle_effect(
            effect_slab,
            swarm_tx,
            EquiXBuilder::new(),
            PeerId::random(), // local peer ID (not relevant for this test)
            vec![],           // connected peers (not relevant for this test)
            message_tracker.clone(),
            metrics,
        )
        .await;

        println!("handle_effect result: {:?}", result);

        // Verify the function succeeded
        assert!(result.is_ok(), "handle_effect should succeed");
        println!("Verified handle_effect succeeded");

        // Collect all the block actions
        let mut blocked_peers = Vec::new();
        while let Ok(action) = swarm_rx.try_recv() {
            match action {
                SwarmAction::BlockPeer { peer_id } => {
                    println!("Received BlockPeer action for peer: {}", peer_id);
                    blocked_peers.push(peer_id);
                }
                other => {
                    println!("Unexpected action received: {:?}", other);
                    panic!("Expected BlockPeer action, got {:?}", other);
                }
            }
        }

        // Verify both bad peers were blocked
        assert_eq!(
            blocked_peers.len(),
            2,
            "Should have blocked exactly 2 peers"
        );
        assert!(
            blocked_peers.contains(&bad_peer1),
            "bad_peer1 should be blocked"
        );
        assert!(
            blocked_peers.contains(&bad_peer2),
            "bad_peer2 should be blocked"
        );
        assert!(
            !blocked_peers.contains(&good_peer),
            "good_peer should not be blocked"
        );
        println!("Verified correct peers were blocked");

        // Verify the bad block ID was removed from the tracker
        {
            let tracker = message_tracker.lock().await;

            // Bad block should be removed
            assert!(
                !tracker.is_tracking_block_id(bad_block_id),
                "Bad block ID should be removed"
            );

            // Good block should still be tracked
            assert!(
                tracker.is_tracking_block_id(good_block_id),
                "Good block ID should still be tracked"
            );

            // Bad peers should be removed
            assert!(
                !tracker.is_tracking_peer(bad_peer1),
                "bad_peer1 should be removed from tracker"
            );
            assert!(
                !tracker.is_tracking_peer(bad_peer2),
                "bad_peer2 should be removed from tracker"
            );

            // Good peer should still be tracked
            assert!(
                tracker.is_tracking_peer(good_peer),
                "good_peer should still be tracked"
            );

            println!("Verified tracker state is correct after processing effect");
        }
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // ibig has a memory leak so miri fails this test

    async fn test_seen_block_effect() {
        use equix::EquiXBuilder;
        use tokio::sync::mpsc;

        let mut effect_slab = NounSlab::new();
        let block_id = T(&mut effect_slab, &[D(1), D(2), D(3), D(4), D(5)]);
        let block_id_str = tip5_hash_to_base58(block_id).unwrap_or_else(|_| {
            panic!(
                "Called `expect()` at {}:{} (git sha: {})",
                file!(),
                line!(),
                option_env!("GIT_SHA").unwrap_or("unknown")
            )
        });
        let effect = T(
            &mut effect_slab,
            &[D(tas!(b"seen")), D(tas!(b"block")), block_id, D(0)],
        );
        effect_slab.set_root(effect);

        let (swarm_tx, _) = mpsc::channel(1);

        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );

        let message_tracker = Arc::new(Mutex::new(MessageTracker::new(
            metrics.clone(),
            LIBP2P_CONFIG.seen_tx_clear_interval,
        )));
        let message_tracker_clone = Arc::clone(&message_tracker); // Clone the Arc, not the MessageTracker
        let result = handle_effect(
            effect_slab,
            swarm_tx,
            EquiXBuilder::new(),
            PeerId::random(), // local peer ID (not relevant for this test)
            vec![],           // connected peers (not relevant for this test)
            message_tracker_clone,
            metrics,
        )
        .await;

        assert!(result.is_ok(), "handle_effect should succeed");

        // Verify that the block id was added to the seen_blocks set
        let tracker = message_tracker.lock().await;
        let contains = tracker.seen_blocks.contains(&block_id_str);
        assert!(contains, "Block ID should be marked as seen");
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)] // ibig has a memory leak so miri fails this test
    async fn test_seen_tx_effect() {
        use equix::EquiXBuilder;
        use tokio::sync::mpsc;

        let mut effect_slab = NounSlab::new();
        let tx_id = T(&mut effect_slab, &[D(1), D(2), D(3), D(4), D(5)]);
        let tx_id_str = tip5_hash_to_base58(tx_id).unwrap_or_else(|_| {
            panic!(
                "Called `expect()` at {}:{} (git sha: {})",
                file!(),
                line!(),
                option_env!("GIT_SHA").unwrap_or("unknown")
            )
        });
        let effect = T(&mut effect_slab, &[D(tas!(b"seen")), D(tas!(b"tx")), tx_id]);

        effect_slab.set_root(effect);

        let (swarm_tx, _) = mpsc::channel(1);
        let metrics = Arc::new(
            NockchainP2PMetrics::register(gnort::global_metrics_registry())
                .expect("Could not register metrics"),
        );

        let message_tracker = Arc::new(Mutex::new(MessageTracker::new(
            metrics.clone(),
            LIBP2P_CONFIG.seen_tx_clear_interval,
        )));
        let message_tracker_clone = Arc::clone(&message_tracker); // Clone the Arc, not the MessageTracker
        let result = handle_effect(
            effect_slab,
            swarm_tx,
            EquiXBuilder::new(),
            PeerId::random(), // local peer ID (not relevant for this test)
            vec![],           // connected peers (not relevant for this test)
            message_tracker_clone,
            metrics,
        )
        .await;

        assert!(result.is_ok(), "handle_effect should succeed");

        // Verify that the tx id was added to the seen_blocks set
        let tracker = message_tracker.lock().await;
        let contains = tracker.seen_txs.contains(&tx_id_str);
        assert!(contains, "tx ID should be marked as seen");
    }
}

fn dial_peers(
    swarm: &mut Swarm<NockchainBehaviour>,
    peers: &[Multiaddr],
) -> Result<(), NockAppError> {
    let mut rng = rand::thread_rng();

    let cloned_peers: &mut [libp2p::Multiaddr] = &mut peers.to_vec();
    cloned_peers.shuffle(&mut rng);

    for peer in cloned_peers {
        let peer = peer.clone();
        debug!("Dialing peer: {}", peer);
        let _ = swarm.dial(peer.clone()).map_err(log_dial_error);
    }
    Ok(())
}

mod traffic_cop {
    use std::time::Instant;

    use tokio::select;
    use tokio::sync::oneshot;

    use super::*;

    const TRAFFIC_COP_BUFFER_SIZE: usize = 50;

    pub(crate) struct HighPriorityPermit<'a> {
        permit: mpsc::Permit<'a, TrafficCopPoke>,
    }

    impl<'a> HighPriorityPermit<'a> {
        pub(crate) async fn poke_high_priority(
            self,
            wire: WireRepr,
            cause: NounSlab,
            timing: Option<oneshot::Sender<Duration>>,
        ) -> Result<PokeResult, NockAppError> {
            let (result_tx, result_rx) = oneshot::channel();
            let poke = TrafficCopPoke {
                wire,
                cause,
                timing,
                result: result_tx,
            };
            self.permit.send(poke);
            result_rx.await?
        }
    }

    enum TrafficCopAction {
        Poke(TrafficCopPoke),
        Peek {
            path: NounSlab,
            result: oneshot::Sender<Result<Option<NounSlab>, NockAppError>>,
        },
    }

    struct TrafficCopPoke {
        wire: WireRepr,
        cause: NounSlab,
        timing: Option<oneshot::Sender<Duration>>,
        result: oneshot::Sender<Result<PokeResult, NockAppError>>,
    }

    #[derive(Clone)]
    pub(crate) struct TrafficCop {
        high_priority_pokes: mpsc::Sender<TrafficCopPoke>,
        low_priority: mpsc::Sender<TrafficCopAction>,
    }

    impl TrafficCop {
        pub(crate) fn new(
            handle: NockAppHandle,
            join_set: &mut TrackedJoinSet<Result<(), NockAppError>>,
            poke_timeout: Duration,
        ) -> Self {
            let (high_priority_pokes, high) = mpsc::channel(1);
            let (low_priority, low) = mpsc::channel(TRAFFIC_COP_BUFFER_SIZE);
            join_set.spawn(
                "traffic_cop".to_string(),
                traffic_cop_task(handle, high, low, poke_timeout),
            );
            Self {
                high_priority_pokes,
                low_priority,
            }
        }

        pub(crate) async fn poke_high_priority_permit(
            &self,
        ) -> Result<HighPriorityPermit, NockAppError> {
            let permit = self
                .high_priority_pokes
                .reserve()
                .await
                .map_err(|_| NockAppError::ChannelClosedError)?;
            Ok(HighPriorityPermit { permit })
        }

        #[allow(dead_code)]
        pub(crate) async fn poke_low_priority(
            &self,
            wire: WireRepr,
            cause: NounSlab,
            timing: Option<oneshot::Sender<std::time::Duration>>,
        ) -> Result<PokeResult, NockAppError> {
            let (result_tx, result_rx) = oneshot::channel();
            let action = TrafficCopAction::Poke(TrafficCopPoke {
                wire,
                cause,
                timing,
                result: result_tx,
            });
            self.low_priority.try_send(action).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => NockAppError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "dropping poke: queue full",
                )),
                mpsc::error::TrySendError::Closed(_) => NockAppError::ChannelClosedError,
            })?;
            result_rx.await?
        }

        pub(crate) async fn peek(&self, path: NounSlab) -> Result<Option<NounSlab>, NockAppError> {
            let (result_tx, result_rx) = oneshot::channel();
            let action = TrafficCopAction::Peek {
                path,
                result: result_tx,
            };
            self.low_priority.try_send(action).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => NockAppError::IoError(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "dropping peek: queue full",
                )),
                mpsc::error::TrySendError::Closed(_) => NockAppError::ChannelClosedError,
            })?;
            result_rx.await?
        }
    }

    async fn traffic_cop_task(
        handle: NockAppHandle,
        mut high: mpsc::Receiver<TrafficCopPoke>,
        mut low: mpsc::Receiver<TrafficCopAction>,
        poke_timeout: Duration,
    ) -> Result<(), NockAppError> {
        loop {
            select! { biased;
                high_priority_poke = high.recv() => match high_priority_poke {
                    Some(TrafficCopPoke { wire, cause, result, timing }) => {
                        let now = Instant::now();
                        let res = handle.poke_timeout(wire, cause, poke_timeout).await;
                        timing.map(|c| c.send(now.elapsed()));
                        let _ = result.send(res).map_err(|e| {
                            error!("Failed to send high priority poke result");
                            e
                        });
                    }
                    None => {
                        error!("High priority channel closed");
                        break Err(NockAppError::ChannelClosedError);
                    }
                },
                low_priority_action = low.recv() => match low_priority_action {
                    Some(TrafficCopAction::Poke(TrafficCopPoke { wire, cause, result, timing })) => {
                        let now = Instant::now();
                        let res = handle.poke_timeout(wire, cause, poke_timeout).await;
                        let elapsed = now.elapsed();
                        timing.map(|c| c.send(elapsed));
                        let _ = result.send(res).map_err(|e| {
                            error!("Failed to send low priority poke result");
                            e
                        });
                    }
                    Some(TrafficCopAction::Peek { path, result }) => {
                        let res = handle.peek(path).await;
                        let _ = result.send(res).map_err(|e| {
                            error!("Failed to send low priority peek result");
                            e
                        });
                    }
                    None => {
                        error!("Low priority channel closed");
                        break Err(NockAppError::ChannelClosedError);
                    }
                },
                _ = handle.next_effect() => {
                    // We have to do this to prevent the broadcast channel from lagging
                }
            }
        }
    }
}

fn log_dial_error(error: DialError) {
    match error {
        DialError::NoAddresses => debug!("No addresses to dial"),
        DialError::LocalPeerId { address } => {
            debug!("Tried to dial ourselves at {}", address.to_string())
        }

        DialError::Aborted => trace!("Dial aborted"),
        DialError::WrongPeerId { obtained, address } => {
            warn!(
                "Wrong peer id {} from address {}",
                obtained,
                address.to_string()
            )
        }
        DialError::Denied { cause } => debug!("Outgoing connection denied: {}", cause),
        DialError::DialPeerConditionFalse(_) => debug!("Dial peer condition false"),
        DialError::Transport(addr_errs) => {
            for (addr, error) in addr_errs {
                trace!("Failed to dial address {}: {}", addr.to_string(), error);
            }
        }
    }
}

fn log_outbound_failure(
    peer: PeerId,
    error: request_response::OutboundFailure,
    metrics: Arc<NockchainP2PMetrics>,
) {
    metrics.request_failed.increment();
    match error {
        request_response::OutboundFailure::DialFailure => {
            debug!("Failed to dial peer {} for request", peer)
        }
        request_response::OutboundFailure::Timeout => debug!("Request to peer {} timed out", peer),
        request_response::OutboundFailure::ConnectionClosed => {
            debug!("Connection to peer {} closed with request pending", peer)
        }
        request_response::OutboundFailure::Io(err) => {
            debug!("Error making request to peer {}: {}", peer, err)
        }
        request_response::OutboundFailure::UnsupportedProtocols => {
            debug!("Unsupported protocol when making request to peer {}", peer)
        }
    }
}

fn log_inbound_failure(
    peer: PeerId,
    error: request_response::InboundFailure,
    metrics: Arc<NockchainP2PMetrics>,
) {
    if let request_response::InboundFailure::ResponseOmission = error {
        metrics.response_dropped.increment();
    } else {
        metrics.response_failed_not_dropped.increment();
    }
    match error {
        request_response::InboundFailure::ResponseOmission => trace!(
            "Response to peer {} refused, likely load shedding or simply no data for request", peer
        ),
        request_response::InboundFailure::Timeout => warn!("Response to peer {} timed out", peer),
        request_response::InboundFailure::Io(err) => {
            warn!("Error responding to peer {}: {}", peer, err)
        }
        request_response::InboundFailure::ConnectionClosed => {
            debug!("Connection to peer {} closed with response pending", peer)
        }
        request_response::InboundFailure::UnsupportedProtocols => {
            debug!("Unsupported protocol when responding to peer {}", peer)
        }
    };
}

fn dial_more_peers(swarm: &mut Swarm<NockchainBehaviour>, tracker: MutexGuard<MessageTracker>) {
    let mut addresses_to_dial = Vec::new();
    for bucket in swarm.behaviour_mut().kad.kbuckets() {
        for peer in bucket.iter() {
            if tracker
                .peer_connections
                .contains_key(&peer.node.key.into_preimage())
            {
                continue;
            }
            for address in peer.node.value.iter() {
                let mut address = address.clone();

                if let Ok(address_with_peer_id) =
                    address.clone().with_p2p(peer.node.key.into_preimage())
                {
                    address = address_with_peer_id;
                }
                addresses_to_dial.push(address);
            }
        }
    }
    addresses_to_dial.shuffle(&mut rand::thread_rng());
    for address in addresses_to_dial {
        info!("Redialing {}", address);
        if let Err(err) = swarm.dial(address) {
            log_dial_error(err);
        };
    }
}
