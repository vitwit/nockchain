# Create .env file if it doesn't exist
$(shell [ ! -f .env ] && touch .env)

# Load environment variables from .env file
include .env

# Set default env variables if not set in .env
export RUST_BACKTRACE ?= full
export RUST_LOG ?= info,nockchain=info,nockchain_libp2p_io=info,libp2p=info,libp2p_quic=info
export MINIMAL_LOG_FORMAT ?= true
export MINING_PUBKEY ?= 2qwq9dQRZfpFx8BDicghpMRnYGKZsZGxxhh9m362pzpM9aeo276pR1yHZPS41y3CW3vPKxeYM8p8fzZS8GXmDGzmNNCnVNekjrSYogqfEFMqwhHh5iCjaKPaDTwhupWqiXj6

# GPU build configuration
export GPU_SUPPORT ?= true
export CUDA_SUPPORT ?= true 
export OPENCL_SUPPORT ?= true

# GPU feature flags for Cargo
GPU_FEATURES := 
ifeq ($(GPU_SUPPORT),true)
    ifeq ($(CUDA_SUPPORT),true)
        ifeq ($(OPENCL_SUPPORT),true)
            GPU_FEATURES := --features gpu
        else
            GPU_FEATURES := --features cuda
        endif
    else
        ifeq ($(OPENCL_SUPPORT),true)
            GPU_FEATURES := --features opencl
        else
            GPU_FEATURES := 
        endif
    endif
else
    GPU_FEATURES := 
endif

export

.PHONY: build
build: build-hoon-all build-rust
	$(call show_env_vars)

## Build all rust
.PHONY: build-rust
build-rust:
	cargo build --release $(GPU_FEATURES)

.PHONY: build-nockchain-jemalloc
build-nockchain-jemalloc:
	cargo build --release --features jemalloc $(GPU_FEATURES) --bin nockchain

.PHONY: build-nockchain-gpu
build-nockchain-gpu:
	cargo build --release $(GPU_FEATURES) --bin nockchain

.PHONY: build-nockchain-cpu
build-nockchain-cpu:
	cargo build --release --bin nockchain

## Run all tests
.PHONY: test
test:
	cargo test --release $(GPU_FEATURES)

.PHONY: test-gpu
test-gpu:
	cargo test --release $(GPU_FEATURES) gpu_mining

.PHONY: test-cpu
test-cpu:
	cargo test --release

.PHONY: fmt
fmt:
	cargo fmt

.PHONY: install-hoonc
install-hoonc: nuke-hoonc-data ## Install hoonc from this repo
	$(call show_env_vars)
	cargo install --locked --force --path crates/hoonc --bin hoonc

.PHONY: update-hoonc
update-hoonc:
	$(call show_env_vars)
	cargo install --locked --path crates/hoonc --bin hoonc

.PHONY: install-nockchain
install-nockchain: assets/dumb.jam assets/miner.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain --bin nockchain $(GPU_FEATURES)

.PHONY: install-nockchain-gpu
install-nockchain-gpu: assets/dumb.jam assets/miner.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain --bin nockchain --features gpu

.PHONY: install-nockchain-cuda
install-nockchain-cuda: assets/dumb.jam assets/miner.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain --bin nockchain --features cuda

.PHONY: install-nockchain-opencl
install-nockchain-opencl: assets/dumb.jam assets/miner.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain --bin nockchain --features opencl

.PHONY: install-nockchain-cpu
install-nockchain-cpu: assets/dumb.jam assets/miner.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain --bin nockchain

.PHONY: install-nockchain-wallet
install-nockchain-wallet: assets/wal.jam
	$(call show_env_vars)
	cargo install --locked --force --path crates/nockchain-wallet --bin nockchain-wallet

.PHONY: ensure-dirs
ensure-dirs:
	mkdir -p hoon
	mkdir -p assets

.PHONY: build-trivial
build-trivial: ensure-dirs
	$(call show_env_vars)
	echo '%trivial' > hoon/trivial.hoon
	hoonc --arbitrary hoon/trivial.hoon

HOON_TARGETS=assets/dumb.jam assets/wal.jam assets/miner.jam

.PHONY: nuke-hoonc-data
nuke-hoonc-data:
	rm -rf .data.hoonc
	rm -rf ~/.nockapp/hoonc

.PHONY: nuke-assets
nuke-assets:
	rm -f assets/*.jam

.PHONY: build-hoon-all
build-hoon-all: nuke-assets update-hoonc ensure-dirs build-trivial $(HOON_TARGETS)
	$(call show_env_vars)

.PHONY: build-hoon
build-hoon: ensure-dirs update-hoonc $(HOON_TARGETS)
	$(call show_env_vars)

.PHONY: build-assets
build-assets: ensure-dirs $(HOON_TARGETS)
	$(call show_env_vars)

HOON_SRCS := $(find hoon -type file -name '*.hoon')

## Build dumb.jam with hoonc
assets/dumb.jam: ensure-dirs hoon/apps/dumbnet/outer.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/dumb.jam
	RUST_LOG=trace hoonc hoon/apps/dumbnet/outer.hoon hoon
	mv out.jam assets/dumb.jam

## Build wal.jam with hoonc
assets/wal.jam: ensure-dirs hoon/apps/wallet/wallet.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/wal.jam
	RUST_LOG=trace hoonc hoon/apps/wallet/wallet.hoon hoon
	mv out.jam assets/wal.jam

## Build mining.jam with hoonc
assets/miner.jam: ensure-dirs hoon/apps/dumbnet/miner.hoon $(HOON_SRCS)
	$(call show_env_vars)
	rm -f assets/miner.jam
	RUST_LOG=trace hoonc hoon/apps/dumbnet/miner.hoon hoon
	mv out.jam assets/miner.jam

# GPU-specific convenience targets
.PHONY: gpu-check
gpu-check:
	@echo "GPU Configuration:"
	@echo "  GPU_SUPPORT: $(GPU_SUPPORT)"
	@echo "  CUDA_SUPPORT: $(CUDA_SUPPORT)"
	@echo "  OPENCL_SUPPORT: $(OPENCL_SUPPORT)"
	@echo "  GPU_FEATURES: $(GPU_FEATURES)"
	@echo ""
	@echo "To disable GPU support, set: make GPU_SUPPORT=false"
	@echo "To use CUDA only: make CUDA_SUPPORT=true OPENCL_SUPPORT=false"
	@echo "To use OpenCL only: make CUDA_SUPPORT=false OPENCL_SUPPORT=true"

.PHONY: gpu-deps-check
gpu-deps-check:
	@echo "Checking GPU dependencies..."
	@command -v nvidia-smi >/dev/null 2>&1 && echo "✓ NVIDIA GPU detected" || echo "✗ NVIDIA GPU not found"
	@command -v nvcc >/dev/null 2>&1 && echo "✓ CUDA toolkit found" || echo "✗ CUDA toolkit not found"
	@command -v clinfo >/dev/null 2>&1 && echo "✓ OpenCL tools found" || echo "✗ OpenCL tools not found"
	@echo ""
	@echo "For CUDA support, install: nvidia-cuda-toolkit"
	@echo "For OpenCL support, install: opencl-headers opencl-dev"

.PHONY: help-gpu
help-gpu:
	@echo "GPU Mining Build Targets:"
	@echo ""
	@echo "  Build targets:"
	@echo "    make build               - Build with GPU support (default)"
	@echo "    make build-nockchain-gpu - Build nockchain with GPU support"
	@echo "    make build-nockchain-cpu - Build nockchain with CPU only"
	@echo ""
	@echo "  Install targets:"
	@echo "    make install-nockchain        - Install with GPU support (default)"
	@echo "    make install-nockchain-gpu    - Install with both CUDA and OpenCL"
	@echo "    make install-nockchain-cuda   - Install with CUDA only"
	@echo "    make install-nockchain-opencl - Install with OpenCL only"
	@echo "    make install-nockchain-cpu    - Install with CPU only"
	@echo ""
	@echo "  Test targets:"
	@echo "    make test                - Test with GPU support"
	@echo "    make test-gpu            - Test GPU mining specifically"
	@echo "    make test-cpu            - Test CPU mining only"
	@echo ""
	@echo "  Configuration targets:"
	@echo "    make gpu-check           - Show current GPU configuration"
	@echo "    make gpu-deps-check      - Check for GPU dependencies"
	@echo ""
	@echo "  Environment variables:"
	@echo "    GPU_SUPPORT=true/false   - Enable/disable GPU support"
	@echo "    CUDA_SUPPORT=true/false  - Enable/disable CUDA backend"
	@echo "    OPENCL_SUPPORT=true/false - Enable/disable OpenCL backend"
	@echo ""
	@echo "  Usage examples:"
	@echo "    make GPU_SUPPORT=false build"
	@echo "    make CUDA_SUPPORT=false install-nockchain"
	@echo "    make OPENCL_SUPPORT=false test"

define show_env_vars
	@echo "Environment Variables:"
	@echo "  GPU_SUPPORT: $(GPU_SUPPORT)"
	@echo "  CUDA_SUPPORT: $(CUDA_SUPPORT)" 
	@echo "  OPENCL_SUPPORT: $(OPENCL_SUPPORT)"
	@echo "  GPU_FEATURES: $(GPU_FEATURES)"
	@echo "  RUST_BACKTRACE: $(RUST_BACKTRACE)"
	@echo "  RUST_LOG: $(RUST_LOG)"
endef
