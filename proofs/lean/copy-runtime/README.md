# Lean proofs for the RV64 copy runtime

This project proves the correctness of the machine code that
`jolt-platform/src/memory.S` assembles to: the wrappers `__wrap_memcpy` and
`__wrap_memmove` and their shared leaf `__jolt_copy_misaligned`. It uses Lean 4
(`leanprover/lean4:v4.34.0`, see `lean-toolchain`) with core and `Std` only.

The Lean kernel checks every proof. There is no `sorry`, no `native_decide`,
and no `bv_decide`, which in this Lean version checks its certificates natively
and adds an axiom. `lake build` runs `CopyRuntime/Audit.lean`, which fails the
build if a main theorem or a key path lemma depends on an axiom other than
`propext`, `Classical.choice`, and `Quot.sound`. `check.sh` also rejects any
`sorry` in the sources.

## What is proved

The statements are in `CopyRuntime/Theorems.lean`, in the vocabulary of
`CopyRuntime/Spec.lean`.

`Exits cfg region post s` (in `Machine/Semantics.lean`) means that from state
`s`, the machine executes only instructions inside `region` and never faults.
After finitely many steps it reaches the first state outside `region`, and
that state satisfies `post`. The semantics is deterministic, so this is the
behavior of the program (`Theorems.exit_state_unique`).

`Theorems.wrap_memcpy_correct` and `Theorems.wrap_memmove_correct` assume the
following.

* The wrapper section is loaded at `B` (`Config.Loaded`), and the linker
  accepted its two relocations (`RelocsInRange`). `__jolt_copy_misaligned`
  resolves to `L` and `__real_memcpy` or `__real_memmove` to `F`, both outside
  the wrapper, with `F` even (`ExitTargetsOk`).
* `B` is even, `R` is not in the wrapper, and no byte of `[dst, dst + n)` is
  in the wrapper (`CodeRangeOk`).
* The sandbox permits loads only from `[src, src + n)` and stores only to
  `[dst, dst + n)` (`Sandbox`).
* Neither range reaches the top of the address space, that is
  `src + n < 2^64` and `dst + n < 2^64`, and `R` is even (`ArgsOk`). The
  bounds are strict because the code needs them; see "Findings about
  memory.S".
* At entry, `pc = B`, `a0 = dst`, `a1 = src`, `a2 = n`, and `ra = R`
  (`CallEntry`). `Theorems.entry_symbols` checks that each entry symbol is at
  section offset 0.

The conclusion is `Exits cfg (inSection B W) (wrapperPost s₀ L F R dst src n) s₀`.
The outcome is a function of the arguments.

* (b) If `Mismatched dst src` (that is, `(dst - src) mod 8 ≠ 0`) and
  `n ≥ 544`, the wrapper reaches `AtLeaf`: `pc = L`, memory unchanged, every
  register except `t0` unchanged.
* (c) If `Mismatched dst src` and `12 ≤ n < 544`, it reaches `AtFallback`:
  `pc = F`, memory unchanged, every register except `t0` and `t1` unchanged.
  The fallback therefore receives the original `a0`, `a1`, `a2`, `ra`, `sp`,
  and callee-saved registers.
* (a) Otherwise it reaches `Returned`: `pc = R` and `a0 = dst`. Memory is
  `memmoveResult` of the entry memory: every byte of `[dst, dst + n)` gets the
  byte at the same position of `[src, src + n)` as it was at entry, and every
  other byte is unchanged. Every register outside `t0`–`t6` and `a1`–`a7` is
  unchanged. In particular `ra`, `sp`, `gp`, `tp`, and `s0`–`s11` are
  preserved.

Since the wrapper never faults in the sandbox, it reads only `[src, src + n)`,
writes only `[dst, dst + n)`, and makes only naturally aligned accesses. This
follows from `Exits` and needs no separate statement. memcpy gets the memmove
guarantee because both wrappers run identical code
(`Theorems.wrappers_same_code`).

`Theorems.jolt_copy_misaligned_correct` says that the leaf, entered with
`Mismatched dst src`, `n ≥ 544`, and the same kind of hypotheses, ends in
`Returned`. `Theorems.wrap_memcpy_linked_correct` and
`wrap_memmove_linked_correct` load the wrapper and the leaf together. They
show that a call ends in `AtFallback` exactly when the offsets are mismatched
and `12 ≤ n < 544`, and in `Returned` otherwise.

## How the proof is organized

Each wrapper theorem reduces to path lemmas under `CopyRuntime/Proof/`. All
wrapper lemmas are proved once for any section with the wrapper's bytes
(`Wrapper.IsWrapper`).

| Path | Lemma | File |
| --- | --- | --- |
| dispatch from the entry point to the paths below | `Wrapper.dispatch`, `Wrapper.general_entry` | `Wrapper/Main.lean` |
| `n = 0` | `Wrapper.zero_path` | `Wrapper/Exits.lean` |
| mismatched offsets, `12 ≤ n < 544`: tail call to `__real_mem*` | `Wrapper.fallback_path` | `Wrapper/Exits.lean` |
| mismatched offsets, `n ≥ 544`: jump to the leaf | `Wrapper.leaf_path` | `Wrapper/Exits.lean` |
| `a0`, `a1`, `n` multiples of 8, `8 ≤ n ≤ 88`: load table, then store table | `Wrapper.aligned_small_path` | `Wrapper/Tables.lean` |
| `.Lsmall_bytes`: `1 ≤ n ≤ 7`, or mismatched offsets and `8 ≤ n ≤ 11` | `Wrapper.small_bytes_path` | `Wrapper/Tables.lean` |
| `.Lgeneral`: `dst = src`, and the choice of direction | `Wrapper.general_path` | `Wrapper/General.lean` |
| `.Lgeneral`, `src > dst`: alignment prologue, 256-byte loop, 80-byte block loop, word loop, byte tail | `Wrapper.forward_path` (`fwd_prologue`, `fwd_large`, `fwd_pairs`, `fwd_blocks`, `fwd_words`, `fwd_tail`) | `Wrapper/Forward.lean` |
| `.Lgeneral`, `src < dst`: the same from the top down | `Wrapper.backward_path` (`bwd_prologue`, `bwd_large`, `bwd_pairs`, `bwd_blocks`, `bwd_words`, `bwd_tail`) | `Wrapper/Backward.lean` |
| `__jolt_copy_misaligned`, `dst < src`: peel, bound check, 128-byte and word combining loops, byte tail | `Leaf.fwd_path` | `Leaf/Forward.lean` |
| `__jolt_copy_misaligned`, `src < dst` | `Leaf.bwd_path` | `Leaf/Backward.lean` |
| `__jolt_copy_misaligned` from its entry point | `Leaf.leaf_path` | `Leaf.lean` |

The rest of `Proof/` is shared by all paths: `Arith.lean` (address
arithmetic), `Memory.lean` (little-endian memory and the `Copied` invariant),
`Exits.lean` (composition and loop rules for `Exits`, and `CodeAt`),
`Step.lean`, `SimpAttr.lean`, and `ExecLemmas.lean` (one-step rules and the
`rv_exec` simp set), `Tactic.lean` and `Region.lean` (the stepping tactics),
and `Returned.lean` (entry registers and the final state).

## Trusted base

The following must be right for the theorems to mean what they say.

1. **The Lean kernel**, plus the axioms `propext`, `Classical.choice`, and
   `Quot.sound`.
2. **The machine model**, which is four files read together with the
   statements.
   * `Machine/Instr.lean` is the instruction syntax.
   * `Machine/Image.lean` is the syntax of sections, relocations, and symbols.
   * `Machine/Decode.lean` is the decoder for the RV64I and RVC encodings that
     occur. All other encodings decode to `none` and fault.
   * `Machine/Semantics.lean` holds the state, little-endian memory,
     instruction semantics, relocation semantics, sandbox, and `Exits`.

   The decoder is **cross-checked against LLVM** on exactly the bytes we
   reason about. `DecoderCheck.lean` proves by kernel evaluation that a linear
   sweep of each section with the Lean decoder equals the listing that
   `llvm-objdump -M no-aliases -M numeric` printed for it. Python translates
   LLVM's text into Lean syntax, applying the RVC expansions of the ISA
   manual. The stepping tactic only executes instructions found at offsets in
   that listing (`CodeAt.fetch` requires listing membership). A decoder bug on
   an encoding that does not occur here would therefore not affect any proof.
   The semantics of each instruction, such as `srl` using the low six bits or
   `jalr` clearing bit 0, is trusted as written.
3. **The extraction.** `extract.py` reads the section bytes, RELA entries, and
   symbols from the object, and writes `Generated/Object.lean`. It also checks
   that the listing bytes and relocations agree with the ELF file.
4. **The assembler and linker**, in the following ways.
   * The object that `regenerate.sh` builds with rustc's integrated assembler
     (the pinned toolchain, `riscv64imac-unknown-none-elf`, `global_asm!`)
     must be what ends up in the guest. That is the same mechanism
     `jolt-platform/src/memory.rs` uses.
   * The linker must apply `R_RISCV_JAL` and `R_RISCV_CALL_PLT` as the model
     does: it writes `S + A - P` into the immediates and rejects out-of-range
     values. The source uses `.option norelax`, and `extract.py` rejects any
     other relocation type, including `R_RISCV_RELAX`, so the linker cannot
     rewrite code.
   * The wrappers must be linked at the addresses assumed, and no other
     section may overlap them.
5. **The contracts of the external code.** For `__real_memcpy` and
   `__real_memmove` this is their libc or `compiler_builtins` contract, given
   the unchanged arguments of outcome (c). Theorems about one section alone
   assume the leaf meets `jolt_copy_misaligned_correct`. The linked theorems
   prove that instead, from `Leaf.leaf_path`.

Fetching from the immutable extracted image rather than from `State.mem` is
faithful because stores never reach code. The hypothesis `CodeRangeOk` states
this. The proofs do not use it, but the model needs it to be faithful to the
hardware.

## Not covered

* Jolt's tracer and its expansion of instructions into virtual sequences.
  Jolt implements subword loads and stores (`lbu`, `lhu`, `lwu`, `sb`, `sh`,
  and `sw`) with word accesses. The theorems here are about RV64 ISA
  semantics, not about Jolt's constraint system or emulator.
* The final linked guest. Nothing here checks that a particular guest ELF
  contains these bytes at these addresses, or that `--wrap=memcpy` and
  `--wrap=memmove` route calls to the wrappers.
* Concurrency, interrupts, and self-modifying code. The model is a single hart
  with no traps other than the faults above.
* Misaligned accesses. The model faults on them, so a theorem proves that they
  do not happen, not how the hardware would handle them.
* The leaf for `n < 544`. Both wrappers enter it only for `n ≥ 544`.

## Findings about memory.S

Two edge cases need a range whose one-past-the-end address is 2^64. C forbids
such an object, since the end pointer would not be representable, and no Jolt
guest can produce one. They are why `ArgsOk` requires `src + n < 2^64` and
`dst + n < 2^64` instead of `≤`, and `memory.S` states that bound.
`tools/Simulate.lean` reproduces both cases on the extracted bytes.

* **Forward word loop and `src + n = 2^64`.** `.Lforward_words` computes the
  end pointer `t0 = a1 + (a2 & ~7)` and loops while `a1 < t0` (unsigned). If
  the source ends exactly at 2^64, `t0` wraps to 0 and the loop stops after
  one word. Take `src = 2^64 - 96`, `dst = src - 8`, and `n = 96`.
  `__wrap_memmove` returns normally, but `dst` bytes 88..95 are not written.
  The forward-path proof exposed this: `omega` could not show that `t0` does
  not wrap.
* **Leaf backward bound check and `dst + n = 2^64`.** In
  `.Lmis_backward_bound`, `addi t2, a1, 7; andi t2, t2, -8` wraps when
  `a1 > 2^64 - 8`. The check `bgeu a6, t2` then passes wrongly, and
  `.Lmis_backward_setup`'s `ld t3, 0(t2)` reads up to 7 bytes past
  `src + n`. The copied bytes stay correct, because the extra bytes are
  shifted out, but the sandbox faults on the read. An example is `n = 600`,
  `dst = 2^64 - 600`, `src = dst - 1`.

Two properties of the leaf matter for its proof but are not bugs. `memory.S`
notes both.

* **The leaf needs `n ≥ 143` on its own.** After the alignment prologue it
  enters its 128-byte loop without a length check. With `J` bytes already
  copied and source word offset `a ∈ 1..7`, the last load of the first
  iteration ends at source offset `J + 136 - a`, and the prologue leaves
  `J - a ≤ 7`.
  Both wrappers jump to the leaf only for `n ≥ 544`, and
  `jolt_copy_misaligned_correct` assumes that bound. A run of the model on the
  leaf alone with `|src - dst|` about 64 and every mismatched pair of offsets
  in both directions (112 cases) copies correctly at `n = 143`. At `n = 142`
  it faults in 14 of them, on the last load of the first iteration (offset
  `0x188` forward, `0x376` backward), for example `dst = 0x100000`,
  `src = 0x100041`.
* **The first aligned source word can hold copied bytes.** The setup loads
  `t3` from the aligned word just below the source cursor (forward) or at it
  (backward). If the distance `|src - dst|` is smaller than the offset `a`,
  some bytes of that word already hold copied data rather than the original
  source. The combining `srl` (forward) or `sll` (backward) discards them, so
  the result is correct. The proof therefore tracks `t3` only up to the
  shift: `t3 >>> 8a` (forward) or `t3 <<< (64 - 8a)` (backward) equals the
  shifted snapshot word.

`tools/Simulate.lean` also runs `__wrap_memmove` linked with the leaf on 3,480
ordinary inputs. For each it checks that the run does not fault and exits to
the right address, that the memory around `dst` matches `memmoveResult` (or is
unchanged when the call goes to the fallback), and that `sp`, `gp`, `tp`, `a0`,
and `s0`–`s11` are unchanged. The inputs are
lengths 0..1030 around every threshold, source alignments 0, 1, 3, 4, and 7,
destination alignments 0, 2, 5, and 6, and distances 8, 64, and 4224, in both
directions. They exercise every path. It is a cross-check of the model and
the extracted bytes, not part of the proof.

## Building

The project's `.lake` directory is large, so on a small disk point it
elsewhere first, for example `ln -s /path/to/scratch/lean-lake .lake`.
`.lake` is git-ignored.

```sh
cd proofs/lean/copy-runtime
lake build                          # proofs, decoder cross-check, and axiom audit
./regenerate.sh                     # reassemble memory.S and rewrite CopyRuntime/Generated
./check.sh                          # CI: regenerate, require no diff, build, reject sorry
lake env lean tools/Simulate.lean   # after lake build: the model cross-check
```

`regenerate.sh` needs the Rust toolchain pinned by `rust-toolchain.toml` with
the `riscv64imac-unknown-none-elf` target and `llvm-objdump`. It uses
`llvm-objdump` from the `llvm-tools` component when that is installed. Set
`RUSTC` and `LLVM_OBJDUMP` to override. `check.sh --no-regen` skips that step.

A clean `./check.sh` takes about 60 seconds on an Apple M-series machine, and
`tools/Simulate.lean` about 30 seconds. The longest file is
`Wrapper/Tables.lean`, which runs 29 symbolic paths. CI runs `./check.sh`
whenever memory.S or this project changes
(`.github/workflows/copy-runtime-formal-verification.yml`).

## Extending the proofs

* `rv_step hc [facts]` executes one listed instruction symbolically. It reads
  `pc`, looks the offset up in the LLVM listing, proves listing membership by
  `decide +kernel`, and evaluates the instruction with the `rv_exec` simp set.
  It then splits on a branch condition with `rv_split`, discarding any branch
  that `omega` refutes from the branch condition and the `Nat` facts in
  context. Pass register facts (for example `regs 11 = src + …`) and the
  sandbox equations as `facts`. `rv_steps_to hc off [facts]` steps until the
  pc reaches `base + off` on every branch. `rv_preserved` closes the
  register-preservation goal for a chain of writes to clobbered registers.
* Memory facts use `Copied m m₀ dst src lo hi`: memory `m` is the snapshot copy
  of `m₀` on destination offsets `[lo, hi)`, and equals `m₀` elsewhere.
  `Copied.extend_down` and `extend_up` add one store of a snapshot load.
  `Copied.store_fwd` and `store_bwd_self` add one store of a value loaded
  during a forward or backward copy, whose source bytes no store has reached
  yet. `Copied.memmove` turns `Copied … 0 n` into the specification. The
  tactics `copied_down`, `copied_fwd`, and `copied_bwd` apply these lemmas.
* Loops use `Exits.loop`, a loop-invariant rule in continuation-passing form.
  To prove one iteration, symbolically execute from an abstract state that
  satisfies the invariant, and at the back edge call the continuation.
  `Wrapper.fwd_words`, `fwd_blocks`, and `fwd_large` are worked examples.
* Repetitive straight-line code gets a generic lemma proved once by induction,
  so that the symbolic memory term never grows. `Wrapper.fwd_pairs` covers a
  run of `ld t1, 8i(a1); sd t1, 8i(a7)` pairs, as in the 256-byte forward
  loop. Its code hypothesis is discharged against the LLVM listing by
  `decide +kernel` when it is used. Without it, each store's value refers to
  the previous memory, and the term doubles as a tree with every pair.
* `Leaf.combine_byte` is the word-combining identity the misaligned leaf
  needs: `(w₀ >>> 8a) ||| (w₁ <<< (64 - 8a))` has bytes `a..7` of `w₀`
  followed by bytes `0..a-1` of `w₁`, for `a ∈ 1..7`. It fails for `a = 0`,
  because `sll` by 64 is a shift by 0. The setup lemmas (`fwd_setup`,
  `bwd_setup`) derive `a ≠ 0` from the mismatched offsets once `a7` is
  aligned. `Leaf.storeLE_combine` turns the store of a combined word into the
  store of the eight-byte load in between, so the ordinary `Copied` lemmas
  apply.
* The leaf's combine-and-store step (`ld; srl; sll; or; sd`) is one lemma per
  direction (`Leaf.fwd_comb`, `Leaf.bwd_comb`, in `Leaf/Comb.lean`). Its
  immediates and the code offset are parameters, and the two addresses are
  hypotheses of the form `regs 7 + ilo = src + ofNat x`, so `rv_simp` never
  sees an immediate. It hands a fresh abstract state to a continuation, with
  a frame hypothesis. `Leaf.fwd_run` and `bwd_run` chain sixteen of these
  steps by induction on pairs, for the 128-byte loops.
* Each `set_option maxHeartbeats` override is about 2.5 times what its proof
  needs (measured with `IO.getNumHeartbeats` and `Elab.async` off). Every
  other proof builds with half of the default limit of 200000.

### Performance pitfalls

Keep these in mind when writing new automation.

* `simp (disch := tac)` marks the discharger as not well behaved, which
  disables simp's cache, and symbolic runs became exponential. `rv_simp`
  calls `Simp.mainCore` with `wellBehavedDischarge := true`, which is valid
  because `rv_disch` only reads the initial local context.
* A discharger or pruning step that rewrites every hypothesis (`bv_omega`,
  `simp … at *`) made `omega` blow up on large contexts. Normalize only the
  goal or the one new hypothesis.
* `split`, and unification against an `if` whose condition contains a large
  literal such as `addi a2, a2, -80`, unfold `Nat.ble` in unary and hit the
  recursion limit. `rv_split` builds the `ite_goal` term directly.
* A negative immediate becomes a literal `c ≥ 2^63` (`addi a2, a2, -80` gives
  `x + 0xffffffffffffffb0#64`). `simp` with the rfl-lemma `BitVec.toNat_add`
  (directly, or through `rv_arith_simp` and `rv_disch`) on such an `x + c`
  with a symbolic `x` records no proof for that step. The kernel then
  rechecks it by unfolding `Nat.add x.toNat c` in unary, and fails with
  "deep recursion" after about 80 s. This only bites once a proof term
  reaches the kernel, so `omega` pruning that fails is harmless. Pass the
  `Arith` rules to `rv_step` instead: `add_ofNat_eq_sub_of_ge` keeps a
  register as `x - c'` with a small `c'` (and `toNat_sub_ofNat_of_le` gives
  its `Nat` value), and `ofNat_add_neg_lit` folds the immediate into an
  address offset `x + ofNat a`.
* `omega` decides whether two atoms are equal with `isDefEq`, and the
  elaborator settles `j + (256 * k + 256) =?= j + 256 * k + 256` by peeling
  the `+ 256` one successor at a time. Literals of about 256 and up then hit
  the maximum recursion depth. `c * (k + 1)` with such a `c` anywhere in
  `omega`'s context fails in the same way. In loop bodies, name the offset
  (`generalize hi : j + 256 * k = i at …`), and rewrite `I (k + 1)` goals with
  `Nat.mul_add` or `Nat.mul_add_one` by `rw`, not `omega` (see
  `Wrapper.fwd_large` and `Wrapper.bwd_large`).
* `assumption` or `exact` at default transparency against a `Copied` goal over
  a store chain unfolds `Copied` and every `storeLE`. Use `with_reducible`.
* Never compare two concrete states with `rfl` (for example
  `⟨B + 164, …⟩.mem = ⟨B + 704, …⟩.mem`), because Lean tries
  `B + 164 =?= B + 704`. Use pc-generic lemmas such as `Returned.change_pc`.
* Sorting a `writeReg` chain by register number inside `simp` is far too slow.
  `rv_exec` only collapses adjacent writes to the same register.
