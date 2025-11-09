# Base64 Encoder Example

A minimal, self-contained example that proves the execution of a Base64 encoder inside Jolt.

## What it does
- Encodes an input string (`"hello jolt base64!"`) into Base64.
- Returns the result as two fixed-size arrays (`[u8; 32]`) to comply with Jolt’s serialization limits.
- Generates and verifies a zero-knowledge proof that the encoding was executed correctly.

## Run
```bash
cargo run --release -p base64-ex
```

### Expected output
```
Prover runtime: 0.XX s
output: aGVsbG8gam9sdCBiYXNlNjQh
valid: true
```

You can double-check the encoding:
```bash
echo -n "hello jolt base64!" | base64
```
Should print:
```
aGVsbG8gam9sdCBiYXNlNjQh
```

## Structure
```
base64-ex/
├── guest/          # RISC-V guest code (encoder)
├── src/            # Host runner (prove & verify)
└── Cargo.toml      # Host manifest
```

## Notes
- The encoder pads the output with `=` to a multiple of 4 characters.
- Unused bytes are zero-filled; the host trims them for display.
