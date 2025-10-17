# Compressed Instructions Example

This example shows that Jolt generates non-compressed RISC-V instructions.

## What it does

RISC-V has an extension called "C" that creates shorter 16-bit instructions to save space. .

This example:
1. Writes Rust code that would normally create compressed instructions
2. Jolt generate non-compressed instructions
3. Shows the program still works perfectly

## Check the results

After running the example, verify no compressed instructions exist (should print 0):

```bash
riscv64-unknown-elf-objdump -d /tmp/jolt-guest-targets/compressed-guest-demo/riscv64imac-unknown-none-elf/release/compressed-guest | awk '$1 ~ /^[0-9a-f]+:$/ && $2 ~ /^[0-9a-f]{4}$/ {count++} END {print count}'
```

## What code patterns are tested

The program uses common Rust patterns that typically create compressed instructions:
- **Immediate operations** - `C.LI`, `C.ADDI`
- **Register operations** - `C.MV`, `C.ADD`, `C.SUB` 
- **Small constants** - `C.LI` with various small values
- **Memory operations** - `C.LW`, `C.SW` with small offsets
- **Shift operations** - `C.SLLI`, `C.SRLI`, `C.SRAI`
- **Control flow** - `C.BEQZ`, `C.BNEZ`

## Run the example

```bash
cd examples/compressed
cargo run --release
```
