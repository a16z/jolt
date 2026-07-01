# What we did — expansion dump tests

Scaffolding to read the Rust bytecode expander and print it in a form we can
compare to the hand-written Lean in `JoltBytecode/JoltISA/Expansions/`. All of it
is test code; the live prover is untouched.

## How to run

```
# one instruction
cargo nextest run -p jolt-program dump_srl_symbolic --cargo-quiet --no-capture

# all of them
cargo nextest run -p jolt-program dump_ --cargo-quiet --no-capture
```

`--no-capture` is required or nextest hides the `println!` output.

There are two kinds of dump:

- **recipe** dump — the raw expansion *before* materialization: symbolic temps,
  nested `Expand` placeholders, no `rd==x0` handling.
- **symbolic** dump — the *fully expanded* rows through the real path (includes
  `rd==x0` handling), with registers relabeled back to `rd`/`rs1`/`rs2`. This is
  the one that lines up with the Lean.

## crates/jolt-program/src/expand/grammar.rs

- Added `Debug` to `ExpansionOp` so a recipe's ops can be printed.
- Expanded the doc comment on `TempId` to explain the number is a per-recipe
  allocation counter (first `allocate()` is `TempId(0)`), not a register number.

## crates/jolt-program/src/expand/tests.rs

Recipe-level dumps (before materialization):

- `dump_recipe(kind)` — expand one instruction with `expand_source_only_instruction`
  and print every op. `Emit` = a native row kept as-is, `Expand` = a non-native
  helper expanded again later, `Allocate`/`Release` = scratch temps.
- `dump_srl_recipe`, `dump_lw_recipe`, `dump_div_recipe`, `dump_amoaddd_recipe` —
  call `dump_recipe` for those instructions.

Symbolic full-expansion dumps (what matches the Lean):

- `label(reg)` — turn a register number back into a symbol: `0`→`x0`, our
  sentinels `1/2/3`→`rd/rs1/rs2`, anything `>= RISCV_REGISTER_COUNT`→`v{n}` (a
  virtual register; Lean's `inlineTmp0` is `v40`), else `x{n}`.
- `fmt_row(row)` — format one expanded row with symbolic operands, clean
  instruction name, and the immediate as signed 64-bit (so `-8` shows as `-8`).
- `dump_symbolic(kind)` — feed sentinel registers (`rd=1, rs1=2, rs2=3`) through
  the real `expand_instruction` path, once with `rd==x0` and once with `rd!=x0`,
  and print both arms. The two arms are how the Lean's `pureWritebackTraceProgram`
  / side-effecting `rd==x0` handling shows up.
- `dump_srl_symbolic`, `dump_lw_symbolic`, `dump_div_symbolic`,
  `dump_amoaddd_symbolic` — call `dump_symbolic` for those instructions. No
  per-instruction code; the same function handles all of them.

Reused (pre-existing) helpers: `source_row`, `instruction`, `rows`.

## Where to trace back from

Both dumps reach the per-instruction expander through the same dispatch:

- `expand_instruction` (`mod.rs:93`) — full path used by the *symbolic* dumps;
  applies the `rd==x0` handling in `materialize.rs::dispatch_source`.
- `expand_source_only_instruction` (`mod.rs:195`) — the `match` on instruction
  kind used by the *recipe* dumps. Each arm calls one `expand_*` function.

The `expand_*` function is the hand-written Rust that builds the sequence:

- **SRL** → `expand_srl` in `crates/jolt-program/src/expand/shifts/srl.rs`
- **LW** → `expand_lw` in `crates/jolt-program/src/expand/memory/lw.rs`

## Alignment with Lean (as of now)

- SRL → `srlProgram`, LW → `lwProgram`, AMOADDD → `amoadddProgram`,
  DIV → `divProgram` all line up row-for-row, including the `rd==x0` behaviour
  (SRL/DIV are pure-writeback → `ADDI x0,x0,0`; LW/AMOADDD are side-effecting →
  keep the sequence, rewrite dst to `v40`, temps shift up by one).
- Register numbers match (`v40` = `inlineTmp0`), so the eventual equivalence is
  nearly syntactic per program.
- One residual: shift/mask immediates. DIV's `VirtualSRAI` shows
  `imm=-9223372036854775808`, which is `2^63` = Lean's `sraiBitmask(63)`. The Lean
  keeps the symbolic form; our dump shows the concrete number. This is a
  representation gap to resolve on the Lean side, not in the prover.
