# Base64 Encoder Example

A minimal, self-contained example that proves the execution of a Base64 encoder inside Jolt.

## What it does
- Encodes an input string (`"hello jolt base64!"`) into Base64  
- Returns the result as a **single `[u8; 64]`** array  
- Uses a tiny new-type wrapper (`B64Array`) to work around postcard's 32-element array limit  
- Generates and verifies a zero-knowledge proof that the encoding was executed correctly

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
│   └── src/lib.rs  # B64Array new-type with manual serde impl
├── src/            # Host runner (prove & verify)
└── Cargo.toml      # Host manifest
```

## Notes
- The encoder pads the output with = to a multiple of 4 characters
- Unused bytes are zero-filled; the host trims them for display
- The new-type wrapper is zero-cost and only used to satisfy serde limits
