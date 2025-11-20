# ECALL Handlers to Add to Jolt Tracer

## File to Edit
`/Users/sdhawan/Work/jolt/tracer/src/emulator/cpu.rs`

## Find This Location

Search for `JOLT_PRINT_ECALL_NUM` - you should find code around line 550-561 that looks like:

```rust
} else if call_id == JOLT_PRINT_ECALL_NUM {
    let string_ptr = self.x[11] as u32;
    let string_len = self.x[12] as u32;
    let event_type = self.x[13] as u32;

    let _ = self.handle_jolt_print(string_ptr, string_len, event_type as u8);

    return false;
}
```

## Add These Handlers RIGHT AFTER the print handler

```rust
} else if call_id == 0x435352 { // "CSR" in ASCII - CSR operations
    let op = self.x[11] as u32;      // a1: operation type
    let addr = self.x[12] as u16;    // a2: CSR address
    let val = self.x[13] as u64;     // a3: value

    match op {
        1 => { // CSR_OP_READ
            let old = self.read_csr_raw(addr);
            self.x[10] = old as i64;  // Return value in a0
        }
        2 => { // CSR_OP_WRITE
            self.write_csr_raw(addr, val);
        }
        3 => { // CSR_OP_SET (read-modify-write: set bits)
            let old = self.read_csr_raw(addr);
            self.x[10] = old as i64;  // Return old value in a0
            self.write_csr_raw(addr, old | val);
        }
        4 => { // CSR_OP_CLEAR (read-modify-write: clear bits)
            let old = self.read_csr_raw(addr);
            self.x[10] = old as i64;  // Return old value in a0
            self.write_csr_raw(addr, old & !val);
        }
        _ => {
            // Unknown CSR operation - just ignore
        }
    }
    return false; // Don't take the trap

} else if call_id == 0x455849 { // "EXI" in ASCII - exit program
    let exit_code = self.x[11] as u32; // a1: exit code

    // Use existing tohost mechanism to signal exit
    self.get_mut_mmu().store_doubleword_raw(
        self.tohost_addr,
        ((exit_code as u64) << 1) | 1
    );

    return false; // Don't take the trap
}
```

## What This Does

**CSR ECALL (0x435352):**
- Handles CSR read/write operations via ECALL
- Uses existing `read_csr_raw()` and `write_csr_raw()` methods
- Returns values in a0 for READ/SET/CLEAR operations

**Exit ECALL (0x455849):**
- Handles program exit
- Uses existing tohost mechanism (same as riscv-tests)
- Signals emulator to stop

## Verify It Works

After adding, compile Jolt to check for errors:

```bash
cd /Users/sdhawan/Work/jolt
cargo check --bin jolt-emu
```

Should compile without errors!

## Optional: Add Debug Logging

If you want to see what's happening, add logging:

```rust
} else if call_id == 0x435352 {
    tracing::info!("CSR ECALL: op={}, addr={:#x}, val={:#x}",
                   self.x[11], self.x[12], self.x[13]);
    // ... rest of handler ...
}

} else if call_id == 0x455849 {
    tracing::info!("Exit ECALL: code={}", self.x[11]);
    // ... rest of handler ...
}
```

Then run with: `RUST_LOG=info cargo run --bin jolt-emu -- <binary>`
