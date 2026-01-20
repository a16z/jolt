
## Generate proof binary
```bash
cargo run -p recursion generate --example fibonacci        
```

## Verify proof binary
Use `--embed` to embed the proof bytes directly into the recursion-guest program binary, otherwise the proof bytes are passed as input to the recursion-guest program at runtime.
```bash
cargo run -p recursion verify --example fibonacci --embed
```
