# Development Tools
## Tracing
Jolt is instrumented using [tokio-rs/tracing](https://github.com/tokio-rs/tracing). These traces can be displayed using the `--format chrome` flag, for example:
`cargo run -p jolt-core --release -- trace --name sha3 --format chrome`

After tracing, files can be found in the workspace root with a name `trace-<timestamp>.json`. Load these traces into `chrome://tracing` or `https://ui.perfetto.dev/`.

![Tracing in Jolt](../imgs/tracing.png)


## Objdump
Debugging the emulator / tracer can be hard. Use `riscv64-unknown-elf-objdump` to compare the actual ELF to the `.bytecode` / `.jolttrace` files.