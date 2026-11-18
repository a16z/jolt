# Recipes read their arguments as "$@" or "$1": interpolating {{ args }} instead would re-split a quoted
# filter such as -E 'test(/a b/)' into separate words.
set positional-arguments

[private]
default:
    @{{ just_executable() }} --list

# Format the workspace's Rust sources
fmt:
    cargo fmt -q

# Check formatting, as the rust.yml fmt job does
fmt-check:
    cargo fmt --all --check

# Check TOML formatting, as the rust.yml taplo job does
taplo:
    taplo fmt --check

# Spell-check the repository, as the spell-check.yml typos job does
typos:
    typos .

# Style invariants against origin/main, as the rust.yml style job checks them
style:
    python3 scripts/check_style_invariants.py --base origin/main

# Find unused dependencies, as the rust.yml machete job does
machete:
    cargo-machete
    cargo machete --with-metadata

# Workspace Clippy under the three feature sets the rust.yml clippy job starts with
clippy:
    RUSTFLAGS='-D warnings' cargo clippy --all --all-targets --features allocative,host -q --message-format=short
    RUSTFLAGS='-D warnings' cargo clippy --all --all-targets --features allocative,host,zk -q --message-format=short
    RUSTFLAGS='-D warnings' cargo clippy --all --all-targets --no-default-features -q --message-format=short

# Clippy on one selection with warnings denied, e.g. -p jolt-prover --all-targets --features akita
clippy-with *args:
    RUSTFLAGS='-D warnings' cargo clippy "$@" -q --message-format=short

# cargo nextest on a selection, e.g. -p jolt-poly -E 'test(/bind/)'; no arguments tests only the root jolt package
test *args:
    cargo nextest run --cargo-quiet "$@"

# Guest x mode acceptance matrix under one feature set (CLAUDE.md lists them), e.g. prover-fixtures,zk
e2e-matrix features="prover-fixtures":
    cargo nextest run -p jolt-prover --features "$1" -E 'binary(e2e_matrix)' --cargo-quiet

# Install the jolt CLI that guest builds call; rerun after pulling changes
install-cli:
    cargo install --path . --locked

# Build the prover for wasm32, as the rust.yml build-wasm job does
build-wasm:
    RUSTFLAGS='--cfg getrandom_backend="wasm_js"' cargo build -p jolt-prover -p jolt-platform --features jolt-platform/random --release --target wasm32-unknown-unknown
