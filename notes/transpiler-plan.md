# Source-to-source transpiler — plan of attack

Companion to [`source-to-source-transpiler.md`](./source-to-source-transpiler.md). That note states
the problem; this one is the plan.

## Goal

Replace the hand-written `expand_*` functions in `jolt-program` with a serialized, structured form
of each expansion, so we can later target other backends without parsing Rust. RISC-V only for now.
Inlines are out of scope.

## We don't need a new AST

`jolt-program/src/expand/grammar.rs` already emits the IR we want:

- `ExpandedInstructionSequence { source, ops: Vec<ExpansionOp> }`
- `ExpansionOp = Emit | Expand | Allocate | Release` (4 ops; ignore the inline variants)
- Operands: `RegisterOperand = Register(u8) | Temp(TempId) | InlineTemp(_)`, imm is `i128`

Two things about this IR that we want to keep:

- Temps are symbolic `TempId`s. Physical registers get assigned later in `materialize.rs`. Capture
  before that step.
- A non-native step (e.g. `SRL` inside `LW`) is an `Expand` node, not inlined. Leave it nested.

So this is an extraction job, not a language-design job.

## How to get the IR: run the expanders, don't parse them

The `expand_*` functions already build `ExpandedInstructionSequence`. Call each one and serialize
what it returns. No `syn`, no source parsing.

NOTE: (ari) This implies that to get the AST we need to execute jolt code, not execute a separate crate that reads source code

> Right. The extractor is a bin/test that links `jolt-program` and calls `expand_*` directly. It's
> not a source reader. Consequences:
> - It lives inside `jolt-program`, because the builders and `expand_*` fns are private
>   (`pub(in crate::expand)` / `pub(super)`). Or those types move to a shared crate (see below).
> - It only sees the branch each call actually takes. Value-dependent branches (Class B, CSR's
>   rd==0) need one call per branch, not one symbolic call.

NOTE: (ari) There is this incredibly annoying rd=x0 use case flying around. How do we get around that?

> Mostly already handled, so it doesn't leak into the recipes. `materialize.rs::dispatch_source`
> does this before expanding any instruction: if `rd == x0` and the kind isn't in
> `handles_rd_zero_internally` (`operands.rs:51` — ECALL/MRET/EBREAK/CSRRW/CSRRS + field inlines):
> - side-effecting kind → allocate a fresh temp, rewrite `rd → temp`, expand, release;
> - otherwise → emit a noop.
>
> That's why `subw` can reuse `rd` as scratch (`SUB rd,..` then `sext rd,rd`): when `rd == x0` the
> recipe never sees x0, it sees a real register. So x0 doesn't multiply the corpus. We just need to:
> - keep the corpus x0-agnostic;
> - carry two bits per kind so the backend can redo this pre-pass: `has_side_effects` and whether
>   it's in `handles_rd_zero_internally`.
>
> The only recipes that test `rd == 0` themselves are csrrw/csrrs, which are already Class B.

## Static vs. operand-dependent recipes

Branching found in the `expand_*` functions:

| Class | Where | Branches on | Result |
|---|---|---|---|
| A — variant flags | `division/shared.rs` (`word`, `remainder_output`), `memory/shared.rs` (`signed`, `min`, `byte_len`) | which source kind (DIV/DIVW/REM, LB/LBU, MIN/MAX), not operand values | one recipe per concrete kind |
| B — operand values | `csrrs.rs`, `csrrw.rs` | `rs1 == 0`, `rd == 0`, `rd == rs1` | recipe depends on the registers |

Class A is most of them: fix the kind, get one recipe.

Class B is a short, closed list of predicates. Emit one recipe per branch, tagged with its
predicate. Don't add runtime conditionals to the IR.

Still to confirm (Phase 0): no expansion has a loop whose length depends on operand values. If one
does, "one recipe per kind" doesn't hold for it and it needs special handling. Only spot-checked so
far.

## Running the extractor — no guest program

The input is one synthesized `SourceInstructionRow`, not a trace. `expand_*` takes a single decoded
instruction, so there's no ELF and no tracer. The loop:

1. iterate `SourceInstructionKind`, keep the ones where `is_source_only` is true
2. build a row with placeholder operands
3. call `expand_source_only_instruction`
4. save the returned `ExpandedInstructionSequence`

About 60 calls. We don't symbolically execute the Rust (the note rules that out). Three narrow
tricks make the output generic over operand values:

**Registers.** Capture the recipe (what `expand_*` returns), before `materialize.rs`. Temps are
already symbolic there. The only concrete registers are the ones read from `rs1(instruction)` etc.
Feed distinct placeholder indices for rd/rs1/rs2 (distinct, non-zero), then rewrite each placeholder
back to `Src(rd|rs1|rs2)`. A hardwired register (like a CSR virtual register) won't match a
placeholder, so it stays `Reg(u8)`.

**Register-equality branches (Class B).** One call per branch (`rd==0`, `rd==rs1`, general), one
tagged recipe each. Phase 0 has to list every such predicate.

**Immediate transforms.** Some recipes use the source imm directly, some use constants (`-8`, `3`),
and some transform it (`imm << 3`, sign-extend). A placeholder imm won't equal its transform, so
make just the imm a small expression type that records the operation. Everything else stays a
concrete run.

To check we caught every branch: run each kind with a few different operand vectors, lift each
result, and confirm they match (per predicate class). A mismatch means a branch we didn't enumerate.

## Do we need a new crate?

NOTE: (ari) do we need a separate fresh crate also ?

> Split it:
> - Extractor stays in `jolt-program` (needs the private builders and `expand_*`). Don't widen
>   visibility just to move it out.
> - The IR types + serde + the interpreter that runs the corpus + the eventual backend go in a new
>   crate (`jolt-expand-ir`). The IR types (`ExpansionOp`, `RowTemplate`, `TemplateOperands`,
>   `RegisterOperand`) are currently `pub(super)` in `grammar.rs`. Best to define them once in the
>   new crate and have `jolt-program` depend on it, so producer and consumer share one definition.
>   Keeps serde/backend deps out of `jolt-program`.

## Phases

### Phase 0 — audit (do first)
- [ ] Sweep every `expand_*` for control flow. Tag each as A (variant), B (operand predicate), or C
      (loop / unbounded). List the full Class B predicate set.
- [ ] Confirm there's no Class C (or list what there is).

### Phase 1 — serialize the IR
- [ ] Add serde for the corpus IR. Format: RON. It maps the Rust enums 1:1 (`Emit(..)`,
      `Allocate(0)`, `Src(Rs1)`) and takes comments, which helps hand-editing. One file per
      source-only kind (~60), plus one per Class B branch.
- [ ] Keep it symbolic and pre-recursion (nested non-natives stay as `Expand`, not inlined).
- [ ] Operand type is a lifted `RegisterOperand` with three cases: `Src(Rd|Rs1|Rs2)`, `Temp(n)`,
      `Reg(u8)`.
- [ ] Immediate is either a constant, a pass-through of the source imm, or a transform of it. The
      first two survive a plain run; transforms need the imm expression type from above.

Example (`srl.ron`):
```ron
(
    source: SRL,
    ops: [
        Allocate(0),
        Emit(( kind: VirtualShiftRightBitmask, rd: Temp(0), rs1: Src(Rs2), imm: 0 )),
        Emit(( kind: VirtualSRL,               rd: Src(Rd), rs1: Src(Rs1), rs2: Temp(0) )),
        Release(0),
    ],
)
```

### Phase 2 — extract
- [ ] For each source-only kind, run the expander with placeholder operands and capture the ops.
- [ ] Emit the tagged branch recipes for Class B kinds.
- [ ] Write the corpus out.

### Phase 3 — check it
- [ ] Structural: every `Emit` targets a native kind (`for_each_jolt_instruction_kind!`), every
      `Expand` targets a source-only kind and terminates (no cycle).
- [ ] Round-trip: materialize from the corpus and diff against live `materialize.rs` output on a
      real program (`muldiv`, `sha3`).

### Phase 4 — backend (later)
- [ ] Pick the target language / codegen once the corpus is stable. Not now.

## What "correct" means

The live pipeline `expand_* -> materialize -> Vec<JoltInstructionRow>` is the oracle. The transpiler
has to produce the same final rows for the same program. Golden-diff against a real trace is the
acceptance test.

## Out of scope

- Inlines (`inline.rs`, `field_inline/`).
- Backend / target language.
- Changing what counts as native in `jolt-riscv`.

## Open questions

1. Corpus location — checked-in `notes/`, a data crate, or generated at build time?
2. Class B — tagged branch recipes (plan's assumption) or a minimal `if`-node in the IR?
3. One-time migration, or a permanent build step where Rust stays the source and the corpus is
   regenerated + diffed?
