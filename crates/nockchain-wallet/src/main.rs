#![allow(clippy::doc_overindented_list_items)]

use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use getrandom::getrandom;
use nockapp::utils::bytes::Byts;
use nockapp::{system_data_dir, CrownError, NockApp, NockAppError, ToBytesExt};
use nockvm::jets::cold::Nounable;
use nockvm::noun::{Atom, Cell, IndirectAtom, Noun, D, NO, SIG, T, YES};
use tokio::fs as tokio_fs;
use tokio::net::UnixStream;
use tracing::{error, info};
use zkvm_jetpack::hot::produce_prover_hot_state;

mod error;

use kernels::wallet::KERNEL;
use nockapp::driver::*;
use nockapp::kernel::boot::{self, Cli as BootCli};
use nockapp::noun::slab::NounSlab;
use nockapp::utils::make_tas;
use nockapp::wire::{Wire, WireRepr};
use nockapp::{exit_driver, file_driver, markdown_driver, one_punch_driver};

/// Represents a timelock range with optional min and max page numbers
#[derive(Debug, Clone)]
pub struct TimelockRange {
    pub min: Option<u64>,
    pub max: Option<u64>,
}

impl TimelockRange {
    pub fn new(min: Option<u64>, max: Option<u64>) -> Self {
        Self { min, max }
    }

    pub fn none() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    /// Convert to noun representation: [min=(unit page-number) max=(unit page-number)]
    pub fn to_noun(&self, slab: &mut NounSlab) -> Noun {
        let min_noun = match self.min {
            Some(val) => T(slab, &[D(0), D(val)]), // unit: [~ value]
            None => D(0),                          // unit: ~
        };
        let max_noun = match self.max {
            Some(val) => T(slab, &[D(0), D(val)]), // unit: [~ value]
            None => D(0),                          // unit: ~
        };
        T(slab, &[min_noun, max_noun])
    }
}

/// Represents a timelock intent - optional constraint for output notes
#[derive(Debug, Clone)]
pub struct TimelockIntent {
    pub absolute: Option<TimelockRange>,
    pub relative: Option<TimelockRange>,
}

impl TimelockIntent {
    pub fn new(absolute: Option<TimelockRange>, relative: Option<TimelockRange>) -> Self {
        Self { absolute, relative }
    }

    pub fn none() -> Self {
        Self {
            absolute: None,
            relative: None,
        }
    }

    pub fn absolute_only(range: TimelockRange) -> Self {
        Self {
            absolute: Some(range),
            relative: Some(TimelockRange::none()),
        }
    }

    pub fn relative_only(range: TimelockRange) -> Self {
        Self {
            absolute: Some(TimelockRange::none()),
            relative: Some(range),
        }
    }

    /// Convert to noun representation: (unit [absolute=timelock-range relative=timelock-range])
    pub fn to_noun(&self, slab: &mut NounSlab) -> Noun {
        match (&self.absolute, &self.relative) {
            (None, None) => D(0), // unit: ~ (no intent)
            (abs, rel) => {
                let default_abs_range = TimelockRange::none();
                let default_rel_range = TimelockRange::none();
                let abs_range = abs.as_ref().unwrap_or(&default_abs_range);
                let rel_range = rel.as_ref().unwrap_or(&default_rel_range);

                let abs_noun = abs_range.to_noun(slab);
                let rel_noun = rel_range.to_noun(slab);
                let content = T(slab, &[abs_noun, rel_noun]);
                T(slab, &[D(0), content]) // unit: [~ content]
            }
        }
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct WalletCli {
    #[command(flatten)]
    boot: BootCli,

    #[command(subcommand)]
    command: Commands,

    #[arg(long, value_name = "PATH")]
    nockchain_socket: Option<PathBuf>,
}

#[derive(Debug)]
pub enum WalletWire {
    ListNotes,
    UpdateBalance,
    UpdateBlock,
    Exit,
    Command(Commands),
}

impl Wire for WalletWire {
    const VERSION: u64 = 1;
    const SOURCE: &str = "wallet";

    fn to_wire(&self) -> WireRepr {
        let tags = match self {
            WalletWire::ListNotes => vec!["list-notes".into()],
            WalletWire::UpdateBalance => vec!["update-balance".into()],
            WalletWire::UpdateBlock => vec!["update-block".into()],
            WalletWire::Exit => vec!["exit".into()],
            WalletWire::Command(command) => {
                vec!["command".into(), command.as_wire_tag().into()]
            }
        };
        WireRepr::new(WalletWire::SOURCE, WalletWire::VERSION, tags)
    }
}

/// Represents a Noun that the wallet kernel can handle
type CommandNoun<T> = Result<(T, Operation), NockAppError>;

fn validate_label(s: &str) -> Result<String, String> {
    if s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        Ok(s.to_string())
    } else {
        Err("Label must contain only lowercase letters, numbers, and hyphens".to_string())
    }
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands {
    /// Generate a new key pair
    Keygen,

    /// Derive child key (pub, private or both) from the current master key
    DeriveChild {
        /// Index of the child key to derive, should be in range [0, 2^31)
        #[arg(short, long, value_parser = clap::value_parser!(u64).range(0..2 << 31))]
        index: u64,

        /// Hardened or unhardened child key
        #[arg(short, long)]
        hardened: bool,

        /// Label for the child key
        #[arg(short, long, value_parser = validate_label, default_value = None)]
        label: Option<String>,
    },

    /// Import keys from a file or extended key
    #[command(group = clap::ArgGroup::new("import_source").required(true).args(&["file", "key"]))]
    ImportKeys {
        /// Path to the jammed keys file
        #[arg(short = 'f', long = "file", value_name = "FILE")]
        file: Option<String>,

        /// Extended key string (e.g., "zprv..." or "zpub...")
        #[arg(short = 'k', long = "key", value_name = "EXTENDED_KEY")]
        key: Option<String>,
    },

    /// Export keys to a file
    ExportKeys,

    /// Signs a transaction
    SignTx {
        /// Path to input bundle file
        #[arg(short, long)]
        draft: String,

        /// Optional key index to use for signing [0, 2^31)
        #[arg(short, long, value_parser = clap::value_parser!(u64).range(0..2 << 31))]
        index: Option<u64>,
    },

    /// Generate a master private key from a seed phrase
    GenMasterPrivkey {
        /// Seed phrase to generate master private key
        #[arg(short, long)]
        seedphrase: String,
    },

    /// Generate a master public key from a master private key
    GenMasterPubkey {
        /// Master private key (base58-encoded)
        #[arg(short, long)]
        master_privkey: String,
        /// Chain code (base58-encoded)
        #[arg(short, long)]
        chain_code: String,
    },

    /// Perform a simple scan of the blockchain
    Scan {
        /// Master public key to scan for
        #[arg(short, long)]
        master_pubkey: String,
        /// Optional search depth (default 100)
        #[arg(short, long, default_value = "100")]
        search_depth: u64,
        /// Include timelocks in scan
        #[arg(long, default_value = "false")]
        include_timelocks: bool,
        /// Include multisig in scan
        #[arg(long, default_value = "false")]
        include_multisig: bool,
    },

    /// List all notes in the wallet
    ListNotes,

    /// List notes by public key
    ListNotesByPubkey {
        /// Optional public key to filter notes
        #[arg(short, long)]
        pubkey: Option<String>,
    },

    /// List notes by public key in CSV format
    ListNotesByPubkeyCsv {
        /// Public key to filter notes
        #[arg(short, long)]
        pubkey: String,
    },

    /// Perform a simple spend operation
    SimpleSpend {
        /// Names of notes to spend (comma-separated)
        #[arg(long)]
        names: String,
        /// Recipient addresses (comma-separated)
        #[arg(long)]
        recipients: String,
        /// Amounts to send (comma-separated)
        #[arg(long)]
        gifts: String,
        /// Transaction fee
        #[arg(long)]
        fee: u64,
        /// Optional key index to use for signing [0, 2^32)
        /// if hardened index must have 2^31 already added to the index
        #[arg(short, long, value_parser = clap::value_parser!(u64).range(0..2 << 32))]
        index: Option<u64>,
        /// Type of timelock intent: "absolute", "relative", or "none"
        #[arg(long, default_value = "none")]
        timelock_intent: String,
        /// Minimum block height for absolute timelock or relative delay in blocks
        #[arg(long)]
        timelock_min: Option<u64>,
    },

    /// Create a transaction from a draft file
    SendTx {
        /// Draft file to create transaction from
        #[arg(short, long)]
        draft: String,
    },

    /// Update the wallet balance
    UpdateBalance,

    /// Export a master public key
    ExportMasterPubkey,

    /// Import a master public key
    ImportMasterPubkey {
        // Path to keys file generated from export-master-pubkey
        #[arg(short, long)]
        key_path: String,
    },

    /// Lists all public keys in the wallet
    ListPubkeys,

    /// Show the seed phrase for the current master key
    ShowSeedphrase,

    /// Show the master public key
    ShowMasterPubkey,

    /// Show the master private key
    ShowMasterPrivkey,
}

impl Commands {
    fn as_wire_tag(&self) -> &'static str {
        match self {
            Commands::Keygen => "keygen",
            Commands::DeriveChild { .. } => "derive-child",
            Commands::ImportKeys { .. } => "import-keys",
            Commands::ExportKeys => "export-keys",
            Commands::SignTx { .. } => "sign-tx",
            Commands::GenMasterPrivkey { .. } => "gen-master-privkey",
            Commands::GenMasterPubkey { .. } => "gen-master-pubkey",
            Commands::Scan { .. } => "scan",
            Commands::ListNotes => "list-notes",
            Commands::ListNotesByPubkey { .. } => "list-notes-by-pubkey",
            Commands::ListNotesByPubkeyCsv { .. } => "list-notes-by-pubkey-csv",
            Commands::SimpleSpend { .. } => "simple-spend",
            Commands::SendTx { .. } => "send-tx",
            Commands::UpdateBalance => "update-balance",
            Commands::ExportMasterPubkey => "export-master-pubkey",
            Commands::ImportMasterPubkey { .. } => "import-master-pubkey",
            Commands::ListPubkeys => "list-pubkeys",
            Commands::ShowSeedphrase => "show-seedphrase",
            Commands::ShowMasterPubkey => "show-master-pubkey",
            Commands::ShowMasterPrivkey => "show-master-privkey",
        }
    }
}

pub struct Wallet {
    app: NockApp,
}

#[derive(Debug, Clone)]
pub enum KeyType {
    Pub,
    Prv,
}

impl KeyType {
    pub fn to_string(&self) -> &'static str {
        match self {
            KeyType::Pub => "pub",
            KeyType::Prv => "prv",
        }
    }
}

impl Wallet {
    /// Creates a new `Wallet` instance with the given kernel.
    ///
    /// This wraps the kernel in a NockApp, which exposes a substrate
    /// for kernel interaction with IO driver semantics.
    ///
    /// # Arguments
    ///
    /// * `kernel` - The kernel to initialize the wallet with.
    ///
    /// # Returns
    ///
    /// A new `Wallet` instance with the kernel initialized
    /// as a NockApp.
    fn new(nockapp: NockApp) -> Self {
        Wallet { app: nockapp }
    }

    /// Wraps a command with sync-run to ensure it runs after block and balance updates
    ///
    /// # Arguments
    ///
    /// * `command_noun_slab` - The command noun to wrap
    /// * `operation` - The operation type (Poke or Peek)
    ///
    /// # Returns
    ///
    /// A result containing the wrapped command noun and operation, or an error
    fn wrap_with_sync_run(
        command_noun_slab: NounSlab,
        operation: Operation,
    ) -> Result<(NounSlab, Operation), NockAppError> {
        let mut sync_slab = command_noun_slab.clone();

        let sync_tag = make_tas(&mut sync_slab, "sync-run");
        let tag_noun = sync_tag.as_noun();

        sync_slab.modify(move |original_root| vec![tag_noun, original_root]);

        Ok((sync_slab, operation))
    }

    /// Prepares a wallet command for execution.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute.
    /// * `args` - The arguments for the command.
    /// * `operation` - The operation type (Poke or Peek).
    /// * `slab` - The NounSlab to use for the command.
    ///
    /// # Returns
    ///
    /// A `CommandNoun` containing the prepared NounSlab and operation.
    fn wallet(
        command: &str,
        args: &[Noun],
        operation: Operation,
        slab: &mut NounSlab,
    ) -> CommandNoun<NounSlab> {
        let head = make_tas(slab, command).as_noun();

        let tail = match args.len() {
            0 => D(0),
            1 => args[0],
            _ => T(slab, args),
        };

        let full = T(slab, &[head, tail]);

        slab.set_root(full);
        Ok((slab.clone(), operation))
    }

    /// Generates a new key pair.
    ///
    /// # Arguments
    ///
    /// * `entropy` - The entropy to use for key generation.
    fn keygen(entropy: &[u8; 32], sal: &[u8; 16]) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let ent: Byts = Byts::new(entropy.to_vec());
        let ent_noun = ent.into_noun(&mut slab);
        let sal: Byts = Byts::new(sal.to_vec());
        let sal_noun = sal.into_noun(&mut slab);
        Self::wallet("keygen", &[ent_noun, sal_noun], Operation::Poke, &mut slab)
    }

    // Derives a child key from current master key.
    //
    // # Arguments
    //
    // * `index` - The index of the child key to derive
    // * `hardened` - Whether the child key should be hardened
    // * `label` - Optional label for the child key
    fn derive_child(index: u64, hardened: bool, label: &Option<String>) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let index_noun = D(index);
        let hardened_noun = if hardened { YES } else { NO };
        let label_noun = label.as_ref().map_or(SIG, |l| {
            let label_noun = l.into_noun(&mut slab);
            T(&mut slab, &[SIG, label_noun])
        });

        Self::wallet(
            "derive-child",
            &[index_noun, hardened_noun, label_noun],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Signs a transaction.
    ///
    /// # Arguments
    ///
    /// * `draft_path` - Path to the draft file
    /// * `index` - Optional index of the key to use for signing
    fn sign_tx(draft_path: &str, index: Option<u64>) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();

        // Validate index is within range (though clap should prevent this)
        if let Some(idx) = index {
            if idx > 255 {
                return Err(CrownError::Unknown("Key index must not exceed 255".into()).into());
            }
        }

        // Read and decode the input bundle
        let draft_data = fs::read(draft_path)
            .map_err(|e| CrownError::Unknown(format!("Failed to read draft: {}", e)))?;

        // Convert the bundle data into a noun using cue
        let draft_noun = slab
            .cue_into(draft_data.as_bytes()?)
            .map_err(|e| CrownError::Unknown(format!("Failed to decode draft: {}", e)))?;

        let index_noun = match index {
            Some(i) => T(&mut slab, &[D(0), D(i)]),
            None => D(0),
        };

        // Generate random entropy
        let mut entropy_bytes = [0u8; 32];
        getrandom(&mut entropy_bytes).map_err(|e| CrownError::Unknown(e.to_string()))?;
        let entropy = from_bytes(&mut slab, &entropy_bytes).as_noun();

        Self::wallet(
            "sign-tx",
            &[draft_noun, index_noun, entropy],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Generates a master private key from a seed phrase.
    ///
    /// # Arguments
    ///
    /// * `seedphrase` - The seed phrase to generate the master private key from.
    fn gen_master_privkey(seedphrase: &str) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let seedphrase_noun = make_tas(&mut slab, seedphrase).as_noun();
        Self::wallet(
            "gen-master-privkey",
            &[seedphrase_noun],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Generates a master public key from a master private key and chain code.
    ///
    /// # Arguments
    ///
    /// * `master_privkey` - The master private key (base58-encoded)
    /// * `chain_code` - The chain code (base58-encoded)
    fn gen_master_pubkey(master_privkey: &str, chain_code: &str) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let key_noun = make_tas(&mut slab, master_privkey).as_noun();
        let chain_code_noun = make_tas(&mut slab, chain_code).as_noun();
        Self::wallet(
            "gen-master-pubkey",
            &[key_noun, chain_code_noun],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Imports keys.
    ///
    /// # Arguments
    ///
    /// * `input_path` - Path to jammed keys file
    fn import_keys(input_path: &str) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();

        let key_data = fs::read(input_path)
            .map_err(|e| CrownError::Unknown(format!("Failed to read master pubkeys: {}", e)))?;

        let pubkey_noun = slab
            .cue_into(key_data.as_bytes()?)
            .map_err(|e| CrownError::Unknown(format!("Failed to decode master pubkeys: {}", e)))?;

        Self::wallet("import-keys", &[pubkey_noun], Operation::Poke, &mut slab)
    }

    /// Imports an extended key.
    ///
    /// # Arguments
    ///
    /// * `extended_key` - Extended key string (e.g., "zprv..." or "zpub...")
    fn import_extended(extended_key: &str) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let key_noun = make_tas(&mut slab, extended_key).as_noun();
        Self::wallet("import-extended", &[key_noun], Operation::Poke, &mut slab)
    }

    /// Exports keys to a file.
    fn export_keys() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("export-keys", &[], Operation::Poke, &mut slab)
    }

    /// Performs a simple scan of the blockchain.
    ///
    /// # Arguments
    ///
    /// * `master_pubkey` - The master public key to scan for.
    /// * `search_depth` - How many addresses to scan (default 100)
    fn scan(
        master_pubkey: &str,
        search_depth: u64,
        include_timelocks: bool,
        include_multisig: bool,
    ) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let master_pubkey_noun = make_tas(&mut slab, master_pubkey).as_noun();
        let search_depth_noun = D(search_depth);
        let include_timelocks_noun = D(include_timelocks as u64);
        let include_multisig_noun = D(include_multisig as u64);

        Self::wallet(
            "scan",
            &[
                master_pubkey_noun, search_depth_noun, include_timelocks_noun,
                include_multisig_noun,
            ],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Performs a simple spend operation by creating transaction inputs from notes.
    ///
    /// Takes a list of note names, recipient addresses, and gift amounts to create
    /// transaction inputs. The fee is subtracted from the first note that has sufficient
    /// assets to cover both the fee and its corresponding gift amount.
    ///
    /// # Arguments
    ///
    /// * `names` - Comma-separated list of note name pairs in format "[first last]"
    ///             Example: "[first1 last1],[first2 last2]"
    ///
    /// * `recipients` - Comma-separated list of recipient $locks
    ///                 Example: "[1 pk1],[2 pk2,pk3,pk4]"
    ///                 A simple comma-separated list is also supported: "pk1,pk2,pk3",
    ///                 where it is presumed that all recipients are single-signature,
    ///                 that is to say, it is the same as "[1 pk1],[1 pk2],[1 pk3]"
    ///
    /// * `gifts` - Comma-separated list of amounts to send to each recipient
    ///             Example: "100,200"
    ///
    /// * `fee` - Transaction fee to be subtracted from one of the input notes
    ///
    /// # Returns
    ///
    /// Returns a `CommandNoun` containing:
    /// - A `NounSlab` with the encoded simple-spend command
    /// - The `Operation` type (Poke)
    ///
    /// # Errors
    ///
    /// Returns `NockAppError` if:
    /// - Name pairs are not properly formatted as "[first last]"
    /// - Number of names, recipients, and gifts don't match
    /// - Any input parsing fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// let names = "[first1 last1],[first2 last2]";
    /// let recipients = "[1 pk1],[2 pk2,pk3,pk4]";
    /// let gifts = "100,200";
    /// let fee = 10;
    /// wallet.simple_spend(names.to_string(), recipients.to_string(), gifts.to_string(), fee)?;
    /// ```
    fn simple_spend(
        names: String,
        recipients: String,
        gifts: String,
        fee: u64,
        index: Option<u64>,
        timelock_intents: Vec<TimelockIntent>,
    ) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();

        // Split the comma-separated inputs
        // Each name should be in format "[first last]"
        let names_vec: Vec<(String, String)> = names
            .split(',')
            .filter_map(|pair| {
                let pair = pair.trim();
                if pair.starts_with('[') && pair.ends_with(']') {
                    let inner = &pair[1..pair.len() - 1];
                    let parts: Vec<&str> = inner.split_whitespace().collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        // Convert recipients to list of [number pubkeys] pairs
        let recipients_vec: Vec<(u64, Vec<String>)> = if recipients.contains('[') {
            // Parse complex format: "[1 pk1],[2 pk2,pk3,pk4]"
            recipients
                .split(',')
                .filter_map(|pair| {
                    let pair = pair.trim();
                    if pair.starts_with('[') && pair.ends_with(']') {
                        let inner = &pair[1..pair.len() - 1];
                        let mut parts = inner.splitn(2, ' ');

                        // Parse the number
                        let number = parts.next()?.parse().ok()?;

                        // Parse the pubkeys
                        let pubkeys = parts
                            .next()?
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();

                        Some((number, pubkeys))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            // Parse simple format: "pk1,pk2,pk3"
            recipients
                .split(',')
                .map(|addr| (1, vec![addr.trim().to_string()]))
                .collect()
        };

        let gifts_vec: Vec<u64> = gifts.split(',').filter_map(|s| s.parse().ok()).collect();

        // Verify equal lengths
        if names_vec.len() != recipients_vec.len() || names_vec.len() != gifts_vec.len() {
            return Err(CrownError::Unknown(
                "Invalid input - names, recipients, and gifts must have the same length"
                    .to_string(),
            )
            .into());
        }

        // Use the first timelock intent if provided, or a default one
        let timelock_intent = if timelock_intents.is_empty() {
            TimelockIntent::none()
        } else {
            timelock_intents
                .into_iter()
                .next()
                .unwrap_or(TimelockIntent::none())
        };

        // Convert names to list of pairs
        let names_noun = names_vec
            .into_iter()
            .rev()
            .fold(D(0), |acc, (first, last)| {
                // Create a tuple [first_name last_name] for each name pair
                let first_noun = make_tas(&mut slab, &first).as_noun();
                let last_noun = make_tas(&mut slab, &last).as_noun();
                let name_pair = T(&mut slab, &[first_noun, last_noun]);
                Cell::new(&mut slab, name_pair, acc).as_noun()
            });

        // Convert recipients to list
        let recipients_noun = recipients_vec
            .into_iter()
            .rev()
            .fold(D(0), |acc, (num, pubkeys)| {
                // Create the inner list of pubkeys
                let pubkeys_noun = pubkeys.into_iter().rev().fold(D(0), |acc, pubkey| {
                    let pubkey_noun = make_tas(&mut slab, &pubkey).as_noun();
                    Cell::new(&mut slab, pubkey_noun, acc).as_noun()
                });

                // Create the pair of [number pubkeys_list]
                let pair = T(&mut slab, &[D(num), pubkeys_noun]);
                Cell::new(&mut slab, pair, acc).as_noun()
            });

        // Convert gifts to list
        let gifts_noun = gifts_vec.into_iter().rev().fold(D(0), |acc, amount| {
            Cell::new(&mut slab, D(amount), acc).as_noun()
        });

        let fee_noun = D(fee);
        let index_noun = match index {
            Some(i) => {
                let inner = D(i);
                T(&mut slab, &[D(0), inner])
            }
            None => D(0),
        };

        // Convert timelock intent to noun
        let timelock_intent_noun = timelock_intent.to_noun(&mut slab);

        Self::wallet(
            "simple-spend",
            &[
                names_noun, recipients_noun, gifts_noun, fee_noun, index_noun, timelock_intent_noun,
            ],
            Operation::Poke,
            &mut slab,
        )
    }

    fn update_balance() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("update-balance", &[], Operation::Poke, &mut slab)
    }

    /// Lists all notes in the wallet.
    ///
    /// Retrieves and displays all notes from the wallet's balance, sorted by assets.
    fn list_notes() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("list-notes", &[], Operation::Poke, &mut slab)
    }

    /// Exports the master public key.
    ///
    /// # Returns
    ///
    /// Retrieves and displays master public key and chaincode.
    fn export_master_pubkey() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("export-master-pubkey", &[], Operation::Poke, &mut slab)
    }

    /// Imports a master public key.
    ///
    /// # Arguments
    ///
    /// * `key` - Base58-encoded public key
    /// * `chain_code` - Base58-encoded chain code
    fn import_master_pubkey(input_path: &str) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();

        let key_data = fs::read(input_path)
            .map_err(|e| CrownError::Unknown(format!("Failed to read master pubkeys: {}", e)))?;

        let pubkey_noun = slab
            .cue_into(key_data.as_bytes()?)
            .map_err(|e| CrownError::Unknown(format!("Failed to decode master pubkeys: {}", e)))?;

        Self::wallet(
            "import-master-pubkey",
            &[pubkey_noun],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Creates a transaction from a draft file.
    ///
    /// # Arguments
    ///
    /// * `draft_path` - Path to the draft file to create transaction from
    fn send_tx(draft_path: &str) -> CommandNoun<NounSlab> {
        // Read and decode the draft file
        let draft_data = fs::read(draft_path)
            .map_err(|e| CrownError::Unknown(format!("Failed to read draft file: {}", e)))?;

        let mut slab = NounSlab::new();
        let draft_noun = slab
            .cue_into(draft_data.as_bytes()?)
            .map_err(|e| CrownError::Unknown(format!("Failed to decode draft data: {}", e)))?;

        Self::wallet("send-tx", &[draft_noun], Operation::Poke, &mut slab)
    }

    /// Lists all public keys in the wallet.
    fn list_pubkeys() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("list-pubkeys", &[], Operation::Poke, &mut slab)
    }

    /// Lists notes by public key
    fn list_notes_by_pubkey(pubkey: &str) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let pubkey_noun = make_tas(&mut slab, pubkey).as_noun();
        Self::wallet(
            "list-notes-by-pubkey",
            &[pubkey_noun],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Lists notes by public key in CSV format
    fn list_notes_by_pubkey_csv(pubkey: &str) -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        let pubkey_noun = make_tas(&mut slab, pubkey).as_noun();
        Self::wallet(
            "list-notes-by-pubkey-csv",
            &[pubkey_noun],
            Operation::Poke,
            &mut slab,
        )
    }

    /// Shows the seed phrase for the current master key.
    fn show_seedphrase() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("show-seedphrase", &[], Operation::Poke, &mut slab)
    }

    /// Shows the master public key.
    fn show_master_pubkey() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("show-master-pubkey", &[], Operation::Poke, &mut slab)
    }

    /// Shows the master private key.
    fn show_master_privkey() -> CommandNoun<NounSlab> {
        let mut slab = NounSlab::new();
        Self::wallet("show-master-privkey", &[], Operation::Poke, &mut slab)
    }
}

pub async fn wallet_data_dir() -> Result<PathBuf, NockAppError> {
    let wallet_data_dir = system_data_dir().join("wallet");
    if !wallet_data_dir.exists() {
        tokio_fs::create_dir_all(&wallet_data_dir)
            .await
            .map_err(|e| {
                CrownError::Unknown(format!("Failed to create wallet data directory: {}", e))
            })?;
    }
    Ok(wallet_data_dir)
}

#[tokio::main]
async fn main() -> Result<(), NockAppError> {
    let cli = WalletCli::parse();
    boot::init_default_tracing(&cli.boot.clone()); // Init tracing early

    let prover_hot_state = produce_prover_hot_state();
    let data_dir = wallet_data_dir().await?;

    let kernel = boot::setup(
        KERNEL,
        Some(cli.boot.clone()),
        prover_hot_state.as_slice(),
        "wallet",
        Some(data_dir),
    )
    .await
    .map_err(|e| CrownError::Unknown(format!("Kernel setup failed: {}", e)))?;

    let mut wallet = Wallet::new(kernel);

    // Determine if this command requires chain synchronization

    let requires_socket = match &cli.command {
        // Commands that DON'T need socket either because they don't sync
        // or they don't interact with the chain
        Commands::Keygen
        | Commands::DeriveChild { .. }
        | Commands::ImportKeys { .. }
        | Commands::ExportKeys
        | Commands::SignTx { .. }
        | Commands::GenMasterPrivkey { .. }
        | Commands::GenMasterPubkey { .. }
        | Commands::ExportMasterPubkey
        | Commands::ImportMasterPubkey { .. }
        | Commands::ListPubkeys
        | Commands::ShowSeedphrase
        | Commands::ShowMasterPubkey
        | Commands::ShowMasterPrivkey
        | Commands::SimpleSpend { .. } => false,

        // All other commands DO need sync
        _ => true,
    };
    // Check if we need sync but don't have a socket
    if requires_socket && cli.nockchain_socket.is_none() {
        return Err(CrownError::Unknown(
            "This command requires connection to a nockchain node. Please provide --nockchain-socket"
            .to_string()
        ).into());
    }

    // Generate the command noun and operation
    let poke = match &cli.command {
        Commands::Keygen => {
            let mut entropy = [0u8; 32];
            let mut salt = [0u8; 16];
            getrandom(&mut entropy).map_err(|e| CrownError::Unknown(e.to_string()))?;
            getrandom(&mut salt).map_err(|e| CrownError::Unknown(e.to_string()))?;
            Wallet::keygen(&entropy, &salt)
        }
        Commands::DeriveChild {
            index,
            hardened,
            label,
        } => Wallet::derive_child(*index, *hardened, label),

        Commands::SignTx { draft, index } => Wallet::sign_tx(draft, *index),
        Commands::ImportKeys { file, key } => {
            if let Some(file_path) = file {
                Wallet::import_keys(file_path)
            } else if let Some(extended_key) = key {
                Wallet::import_extended(extended_key)
            } else {
                return Err(CrownError::Unknown(
                    "Either --file or --key must be provided for import-keys".to_string(),
                )
                .into());
            }
        }
        Commands::ExportKeys => Wallet::export_keys(),
        Commands::GenMasterPrivkey { seedphrase } => Wallet::gen_master_privkey(seedphrase),
        Commands::GenMasterPubkey {
            master_privkey,
            chain_code,
        } => Wallet::gen_master_pubkey(master_privkey, chain_code),
        Commands::Scan {
            master_pubkey,
            search_depth,
            include_timelocks,
            include_multisig,
        } => Wallet::scan(
            master_pubkey, *search_depth, *include_timelocks, *include_multisig,
        ),
        Commands::ListNotes => Wallet::list_notes(),
        Commands::ListNotesByPubkey { pubkey } => {
            if let Some(pk) = pubkey {
                Wallet::list_notes_by_pubkey(pk)
            } else {
                return Err(CrownError::Unknown("Public key is required".into()).into());
            }
        }
        Commands::ListNotesByPubkeyCsv { pubkey } => Wallet::list_notes_by_pubkey_csv(pubkey),
        Commands::SimpleSpend {
            names,
            recipients,
            gifts,
            fee,
            index,
            timelock_intent,
            timelock_min,
        } => {
            let parsed_timelock_intent = match timelock_intent.as_str() {
                "absolute" => {
                    TimelockIntent::absolute_only(TimelockRange::new(*timelock_min, None))
                }
                "relative" => {
                    TimelockIntent::relative_only(TimelockRange::new(*timelock_min, None))
                }
                "none" => TimelockIntent::none(),
                _ => {
                    return Err(CrownError::Unknown(format!(
                        "Unknown timelock intent: {}",
                        timelock_intent
                    ))
                    .into())
                }
            };

            Wallet::simple_spend(
                names.clone(),
                recipients.clone(),
                gifts.clone(),
                *fee,
                *index,
                vec![parsed_timelock_intent],
            )
        }
        Commands::SendTx { draft } => Wallet::send_tx(draft),
        Commands::UpdateBalance => Wallet::update_balance(),
        Commands::ExportMasterPubkey => Wallet::export_master_pubkey(),
        Commands::ImportMasterPubkey { key_path } => Wallet::import_master_pubkey(key_path),
        Commands::ListPubkeys => Wallet::list_pubkeys(),
        Commands::ShowSeedphrase => Wallet::show_seedphrase(),
        Commands::ShowMasterPubkey => Wallet::show_master_pubkey(),
        Commands::ShowMasterPrivkey => Wallet::show_master_privkey(),
    }?;

    // If this command requires sync and we have a socket, wrap it with sync-run
    let final_poke = if requires_socket && cli.nockchain_socket.is_some() {
        Wallet::wrap_with_sync_run(poke.0, poke.1)?
    } else {
        poke
    };

    wallet
        .app
        .add_io_driver(one_punch_driver(final_poke.0, final_poke.1))
        .await;

    {
        if let Some(socket_path) = cli.nockchain_socket {
            match UnixStream::connect(&socket_path).await {
                Ok(stream) => {
                    info!("Connected to nockchain NPC socket at {:?}", socket_path);
                    wallet
                        .app
                        .add_io_driver(nockapp::npc_client_driver(stream))
                        .await;
                }
                Err(e) => {
                    error!(
                        "Failed to connect to nockchain NPC socket at {:?}: {}\n\
                         This could mean:\n\
                         1. Nockchain is not running\n\
                         2. The socket path is incorrect\n\
                         3. The socket file exists but is stale (try removing it)\n\
                         4. Insufficient permissions to access the socket",
                        socket_path, e
                    );
                }
            }
        }

        wallet.app.add_io_driver(file_driver()).await;
        wallet.app.add_io_driver(markdown_driver()).await;
        wallet.app.add_io_driver(exit_driver()).await;

        wallet.app.run().await?;
        Ok(())
    }
}

pub fn from_bytes(stack: &mut NounSlab, bytes: &[u8]) -> Atom {
    unsafe {
        let mut tas_atom = IndirectAtom::new_raw_bytes(stack, bytes.len(), bytes.as_ptr());
        tas_atom.normalize_as_atom()
    }
}

// TODO: all these tests need to also validate the results and not
// just ensure that the wallet can be poked with the expected noun.
#[allow(warnings)]
#[cfg(test)]
mod tests {
    use std::sync::Once;

    use nockapp::kernel::boot::{self, Cli as BootCli};
    use nockapp::wire::SystemWire;
    use nockapp::{exit_driver, AtomExt, Bytes};
    use tokio::sync::mpsc;

    use super::*;

    static INIT: Once = Once::new();

    fn init_tracing() {
        INIT.call_once(|| {
            let cli = boot::default_boot_cli(true);
            boot::init_default_tracing(&cli);
        });
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_keygen() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&["--new"]);

        let prover_hot_state = produce_prover_hot_state();
        let nockapp = boot::setup(
            KERNEL,
            Some(cli.clone()),
            prover_hot_state.as_slice(),
            "wallet",
            None,
        )
        .await
        .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);
        let mut entropy = [0u8; 32];
        let mut salt = [0u8; 16];
        getrandom(&mut entropy).map_err(|e| CrownError::Unknown(e.to_string()))?;
        getrandom(&mut salt).map_err(|e| CrownError::Unknown(e.to_string()))?;
        let (noun, op) = Wallet::keygen(&entropy, &salt)?;

        let wire = WalletWire::Command(Commands::Keygen).to_wire();

        let keygen_result = wallet.app.poke(wire, noun.clone()).await?;

        println!("keygen result: {:?}", keygen_result);
        assert!(
            keygen_result.len() == 2,
            "Expected keygen result to be a list of 2 noun slabs - markdown and exit"
        );
        let exit_cause = unsafe { keygen_result[1].root() };
        let code = exit_cause.as_cell()?.tail();
        assert!(unsafe { code.raw_equals(&D(0)) }, "Expected exit code 0");

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_derive_child() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&["--new"]);

        let prover_hot_state = produce_prover_hot_state();
        let nockapp = boot::setup(
            KERNEL,
            Some(cli.clone()),
            prover_hot_state.as_slice(),
            "wallet",
            None,
        )
        .await
        .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);
        let key_type = KeyType::Prv;

        // Generate a new key pair
        let mut entropy = [0u8; 32];
        let mut salt = [0u8; 16];
        let (noun, op) = Wallet::keygen(&entropy, &salt)?;
        let wire = WalletWire::Command(Commands::Keygen).to_wire();
        let _ = wallet.app.poke(wire, noun.clone()).await?;

        // Derive a child key
        let index = 0;
        let hardened = true;
        let label = None;
        let (noun, op) = Wallet::derive_child(index, hardened, &label)?;

        let wire = WalletWire::Command(Commands::DeriveChild {
            index,
            hardened,
            label,
        })
        .to_wire();

        let derive_result = wallet.app.poke(wire, noun.clone()).await?;

        assert!(
            derive_result.len() == 1,
            "Expected derive result to be a list of 1 noun slab"
        );

        let exit_cause = unsafe { derive_result[0].root() };
        let code = exit_cause.as_cell()?.tail();
        assert!(unsafe { code.raw_equals(&D(0)) }, "Expected exit code 0");

        Ok(())
    }

    // TODO make this a real test by creating and signing a real draft
    #[tokio::test]
    #[ignore]
    async fn test_sign_tx() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&[""]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);

        // Create a temporary input bundle file
        let bundle_path = "test_bundle.jam";
        let test_data = vec![0u8; 32]; // TODO make this a real input bundle
        fs::write(bundle_path, &test_data).map_err(|e| NockAppError::IoError(e))?;

        let wire = WalletWire::Command(Commands::SignTx {
            draft: bundle_path.to_string(),
            index: None,
        })
        .to_wire();

        // Test signing with valid indices
        let (noun, op) = Wallet::sign_tx(bundle_path, None)?;
        let sign_result = wallet.app.poke(wire, noun.clone()).await?;

        println!("sign_result: {:?}", sign_result);

        let wire = WalletWire::Command(Commands::SignTx {
            draft: bundle_path.to_string(),
            index: Some(1),
        })
        .to_wire();

        let (noun, op) = Wallet::sign_tx(bundle_path, Some(1))?;
        let sign_result = wallet.app.poke(wire, noun.clone()).await?;

        println!("sign_result: {:?}", sign_result);

        let wire = WalletWire::Command(Commands::SignTx {
            draft: bundle_path.to_string(),
            index: Some(255),
        })
        .to_wire();

        let (noun, op) = Wallet::sign_tx(bundle_path, Some(255))?;
        let sign_result = wallet.app.poke(wire, noun.clone()).await?;

        println!("sign_result: {:?}", sign_result);

        // Cleanup
        fs::remove_file(bundle_path).map_err(|e| NockAppError::IoError(e))?;
        Ok(())
    }

    // Tests for Cold Side Commands
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_gen_master_privkey() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&[""]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);
        let seedphrase = "correct horse battery staple";
        let (noun, op) = Wallet::gen_master_privkey(seedphrase)?;
        println!("privkey_slab: {:?}", noun);
        let wire = WalletWire::Command(Commands::GenMasterPrivkey {
            seedphrase: seedphrase.to_string(),
        })
        .to_wire();
        let privkey_result = wallet.app.poke(wire, noun.clone()).await?;
        println!("privkey_result: {:?}", privkey_result);
        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_gen_master_pubkey() -> Result<(), NockAppError> {
        init_tracing();
        // Start with a fresh wallet (--new flag)
        let cli = BootCli::parse_from(&["--new"]);
        let prover_hot_state = produce_prover_hot_state();
        let nockapp = boot::setup(
            KERNEL,
            Some(cli.clone()),
            prover_hot_state.as_slice(),
            "wallet",
            None,
        )
        .await
        .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);

        let master_privkey = "5crSXzcevKKieL7VTZW2hy3guFgT6sserEXm3pWHZAnQ";
        let chain_code = "yr3PWpcne3t6ByqtHSmybAJkGqyHB41WNifc5qwNfWA";

        // Generate master public key from the private key and chain code
        let (noun, op) = Wallet::gen_master_pubkey(master_privkey, chain_code)?;
        let wire = WalletWire::Command(Commands::GenMasterPubkey {
            master_privkey: master_privkey.to_string(),
            chain_code: chain_code.to_string(),
        })
        .to_wire();
        let pubkey_result = wallet.app.poke(wire, noun.clone()).await?;
        println!("pubkey_result: {:?}", pubkey_result);

        assert!(
            pubkey_result.len() == 2,
            "Expected pubkey result to be a list of 2 noun slabs - markdown and exit"
        );
        let exit_cause = unsafe { pubkey_result[1].root() };
        let code = exit_cause.as_cell()?.tail();
        assert!(unsafe { code.raw_equals(&D(0)) }, "Expected exit code 0");

        // Now show the master private key to verify it matches our input
        let (show_noun, show_op) = Wallet::show_master_privkey()?;
        let show_wire = WalletWire::Command(Commands::ShowMasterPrivkey).to_wire();
        let show_result = wallet.app.poke(show_wire, show_noun).await?;
        println!("show_master_privkey result: {:?}", show_result);

        // Verify the show command succeeded
        assert!(
            show_result.len() == 2,
            "Expected show result to be a list of 2 noun slabs - markdown and exit"
        );
        let show_exit_cause = unsafe { show_result[1].root() };
        let show_code = show_exit_cause.as_cell()?.tail();
        assert!(
            unsafe { show_code.raw_equals(&D(0)) },
            "Expected show exit code 0"
        );

        // Parse the markdown output to extract the private key
        let markdown_slab = &show_result[0];
        let markdown_root = unsafe { markdown_slab.root() };
        let markdown_cell = markdown_root.as_cell()?;
        let markdown_content_atom = markdown_cell.tail().as_atom()?;

        let markdown_text =
            String::from_utf8_lossy(&markdown_content_atom.to_bytes_until_nul()?).to_string();

        println!("Markdown content: {}", markdown_text);

        // Extract the private key from the markdown - it should be on a line with "- private key: "
        

        // remove the "- private key: " prefix and get the base58 value directly
        let extracted_privkey_b58 = markdown_text
            .lines()
            .find(|line| line.trim().starts_with("- private key: "))
            .ok_or_else(|| {
                CrownError::Unknown("Private key not found in markdown output".to_string())
            })?
            .trim_start_matches("- private key: ")
            .trim()
            .to_string();

        // Verify the extracted private key matches our input
        assert_eq!(
            extracted_privkey_b58, master_privkey,
            "Extracted private key '{}' does not match input private key '{}'",
            extracted_privkey_b58, master_privkey
        );

        println!("✓ Verification successful: Private key in output matches input");
        println!("  Input:     {}", master_privkey);
        println!("  Retrieved: {}", extracted_privkey_b58);

        Ok(())
    }

    // Tests for Hot Side Commands
    // TODO: fix this test by adding a real key file
    #[tokio::test]
    #[ignore]
    async fn test_import_keys() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&["--new"]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);

        // Create test key file
        let test_path = "test_keys.jam";
        let test_data = vec![0u8; 32]; // TODO: Use real jammed key data
        fs::write(test_path, &test_data).expect(&format!(
            "Called `expect()` at {}:{} (git sha: {})",
            file!(),
            line!(),
            option_env!("GIT_SHA").unwrap_or("unknown")
        ));

        let (noun, op) = Wallet::import_keys(test_path)?;
        let wire = SystemWire.to_wire();
        let import_result = wallet.app.poke(wire, noun.clone()).await?;

        fs::remove_file(test_path).expect(&format!(
            "Called `expect()` at {}:{} (git sha: {})",
            file!(),
            line!(),
            option_env!("GIT_SHA").unwrap_or("unknown")
        ));

        println!("import result: {:?}", import_result);
        assert!(
            !import_result.is_empty(),
            "Expected non-empty import result"
        );

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_simple_scan() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&[""]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);
        let master_pubkey = "pubkey123";
        let (noun, op) = Wallet::scan(master_pubkey, 100, false, false)?;
        let wire = WalletWire::Command(Commands::Scan {
            master_pubkey: master_pubkey.to_string(),
            search_depth: 100,
            include_timelocks: false,
            include_multisig: false,
        })
        .to_wire();
        let scan_result = wallet.app.poke(wire, noun.clone()).await?;
        println!("scan_result: {:?}", scan_result);
        Ok(())
    }

    // TODO: fix this test
    #[tokio::test]
    #[ignore]
    async fn test_simple_spend_multisig_format() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&[""]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);

        let names = "[first1 last1],[first2 last2]".to_string();
        let recipients = "[1 pk1],[2 pk2,pk3,pk4]".to_string();
        let gifts = "1,2".to_string();
        let fee = 1;

        let (noun, op) = Wallet::simple_spend(
            names.clone(),
            recipients.clone(),
            gifts.clone(),
            fee,
            None,
            vec![TimelockIntent::none()],
        )?;
        let wire = WalletWire::Command(Commands::SimpleSpend {
            names: names.clone(),
            recipients: recipients.clone(),
            gifts: gifts.clone(),
            fee: fee.clone(),
            index: None,
            timelock_intent: "none".to_string(),
            timelock_min: None,
        })
        .to_wire();
        let spend_result = wallet.app.poke(wire, noun.clone()).await?;
        println!("spend_result: {:?}", spend_result);

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_simple_spend_single_sig_format() -> Result<(), NockAppError> {
        let cli = BootCli::parse_from(&[""]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        init_tracing();
        let mut wallet = Wallet::new(nockapp);

        // these should be valid names of notes in the wallet balance
        let names = "[Amt4GcpYievY4PXHfffiWriJ1sYfTXFkyQsGzbzwMVzewECWDV3Ad8Q BJnaDB3koU7ruYVdWCQqkFYQ9e3GXhFsDYjJ1vSmKFdxzf6Y87DzP4n]".to_string();
        let recipients = "3HKKp7xZgCw1mhzk4iw735S2ZTavCLHc8YDGRP6G9sSTrRGsaPBu1AqJ8cBDiw2LwhRFnQG7S3N9N9okc28uBda6oSAUCBfMSg5uC9cefhrFrvXVGomoGcRvcFZTWuJzm3ch".to_string();

        let gifts = "0".to_string();
        let fee = 0;

        // generate keys
        let (genkey_noun, genkey_op) = Wallet::gen_master_privkey("correct horse battery staple")?;
        let (spend_noun, spend_op) = Wallet::simple_spend(
            names.clone(),
            recipients.clone(),
            gifts.clone(),
            fee,
            None,
            vec![TimelockIntent::none()],
        )?;

        let wire1 = WalletWire::Command(Commands::GenMasterPrivkey {
            seedphrase: "correct horse battery staple".to_string(),
        })
        .to_wire();
        let genkey_result = wallet.app.poke(wire1, genkey_noun.clone()).await?;
        println!("genkey_result: {:?}", genkey_result);

        let wire2 = WalletWire::Command(Commands::SimpleSpend {
            names: names.clone(),
            recipients: recipients.clone(),
            gifts: gifts.clone(),
            fee: fee.clone(),
            index: None,
            timelock_intent: "none".to_string(),
            timelock_min: None,
        })
        .to_wire();
        let spend_result = wallet.app.poke(wire2, spend_noun.clone()).await?;
        println!("spend_result: {:?}", spend_result);

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_update_balance() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&["--new"]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);

        let (noun, _) = Wallet::update_balance()?;

        let wire = WalletWire::Command(Commands::UpdateBalance {}).to_wire();
        let update_result = wallet.app.poke(wire, noun.clone()).await?;
        println!("update_result: {:?}", update_result);

        Ok(())
    }

    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn test_list_notes() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&[""]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);

        // Test listing notes
        let (noun, op) = Wallet::list_notes()?;
        let wire = WalletWire::Command(Commands::ListNotes {}).to_wire();
        let list_result = wallet.app.poke(wire, noun.clone()).await?;
        println!("list_result: {:?}", list_result);

        Ok(())
    }

    // TODO: fix this test by adding a real draft
    #[tokio::test]
    #[ignore]
    async fn test_make_tx_from_draft() -> Result<(), NockAppError> {
        init_tracing();
        let cli = BootCli::parse_from(&[""]);
        let nockapp = boot::setup(KERNEL, Some(cli.clone()), &[], "wallet", None)
            .await
            .map_err(|e| CrownError::Unknown(e.to_string()))?;
        let mut wallet = Wallet::new(nockapp);

        // use the draft in .drafts/
        let draft_path = ".drafts/test_draft.draft";
        let test_data = vec![0u8; 32]; // TODO: Use real draft data
        fs::write(draft_path, &test_data).expect(&format!(
            "Called `expect()` at {}:{} (git sha: {})",
            file!(),
            line!(),
            option_env!("GIT_SHA").unwrap_or("unknown")
        ));

        let (noun, op) = Wallet::send_tx(draft_path)?;
        let wire = WalletWire::Command(Commands::SendTx {
            draft: draft_path.to_string(),
        })
        .to_wire();
        let tx_result = wallet.app.poke(wire, noun.clone()).await?;

        fs::remove_file(draft_path).expect(&format!(
            "Called `expect()` at {}:{} (git sha: {})",
            file!(),
            line!(),
            option_env!("GIT_SHA").unwrap_or("unknown")
        ));

        println!("transaction result: {:?}", tx_result);
        assert!(
            !tx_result.is_empty(),
            "Expected non-empty transaction result"
        );

        Ok(())
    }
}
