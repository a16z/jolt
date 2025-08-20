# Installing
Start by installing the `jolt` command line tool.
```
cargo +nightly-2024-09-30 install --git https://github.com/a16z/jolt --force --bins jolt
```

## Development

### Rust

To build Jolt from source, you will need Rust:

- `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Rustup should automatically install the Rust toolchain and necessary targets on
the first `cargo` invocation.
If you need to add the RISC-V target for building
guest programs manually use `rustup target add riscv32im-unknown-none-elf`.

### mdBook

To build this book from source:

`cargo install mdbook mdbook-katex`

For watching the changes in your local browser:

`mdbook watch book --open`
