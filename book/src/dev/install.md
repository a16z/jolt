# Development Installation
## Rust
- `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- `rustup target add riscv32i-unknown-none-elf`

## Circom
Install from source:
- `git clone https://github.com/iden3/circom.git`
- `cd circom && cargo build --release`
- `cargo install --path circom`


## mdBook
`cargo install mdbook`