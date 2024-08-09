# Development Installation

## Rust

- `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Rustup should automatically install Rust toolchain and necessary targets on
the first `cargo` invocation.  If you need to add the RISC-V target for building
guest programs manually use `rustup target add riscv32im-unknown-none-elf`.

## mdBook

For building this book:

`cargo install mdbook mdbook-katex`

For watching the changes in your local browser:

`mdbook watch book --open`
