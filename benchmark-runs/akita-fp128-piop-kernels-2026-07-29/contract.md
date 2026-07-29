# Akita Fp128 PIOP arithmetic kernels

Date: 2026-07-29 EDT
Parent: `2f9f134f641f1025f6f7edeb8169164307611ef6`
Machine: Apple Silicon (`aarch64-apple-darwin`)
Compiler: `rustc 1.95.0 (59807616e 2026-04-14)`, LLVM 22.1.2

## Question

Can an Fp128 accumulator that applies one pseudo-Mersenne fold per product
outperform both the current raw-product accumulator and eager canonical
reduction in the inner-loop patterns used by the Akita PIOP?

The candidate uses

```text
p = 2^128 - C
a * b = lo + 2^128 * hi = lo + C * hi (mod p)
```

to reduce the live accumulator state before its final canonical reduction.

## Fixed evaluator

The benchmark compares these implementations in one binary:

1. Current Akita `Folded128Product` accumulation.
2. Eager canonical Akita accumulation.
3. One-fold Akita accumulation.
4. Current BN254 accumulation as a cross-field reference.

The first two production-loop replicas are:

- a multi-column, two-output Booleanity accumulation;
- the D=4 sum-of-products and split-equality accumulation used by
  `InstructionRaSumcheckProver`.

Primitive throughput is diagnostic only. A candidate cannot advance based on
an isolated multiplication benchmark.

The evaluator reports:

- time per input row or product;
- accumulator size;
- single-thread and production-parallel throughput;
- equality of every candidate output with canonical field arithmetic.

Small and large working sets distinguish register/cache effects from streaming
memory effects. Inputs and benchmark parameters remain fixed after the first
baseline is recorded.

## Expected and falsifying outcomes

Hypothesis: the one-fold accumulator cuts live Fp128 product state enough to
improve both production-loop replicas by at least 10%, without a regression
larger than 3% on either working-set regime.

The hypothesis is falsified if:

- the gain exists only in an isolated primitive;
- either production-loop replica regresses by more than 3%;
- the one-fold representation loses to both existing policies;
- generated code retains the same spill/load pattern; or
- the required headroom cannot be established for all production callers.

A candidate reaches a `2^22` PIOP trace only after clearing the kernel gate. A
`2^26` proof requires a projection of at least one second from named affected
spans. Every proof must verify.

## Correctness requirements

- Random differential tests against canonical Akita arithmetic.
- Maximum field limbs and maximum accumulator term count.
- Parallel merge trees with unequal partition sizes.
- Exact reduction after zero, one, and maximum terms.
- Explicit headroom bound for every production use before changing
  `JoltField::UnreducedProductAccum`.
- Existing Akita and BN254 behavior remains unchanged during the benchmark-only
  phase.

## Scope and budget

Editable during the kernel phase:

- `crates/jolt-prover-legacy/src/field/`
- `crates/jolt-prover-legacy/benches/fp128_piop_arith.rs`
- `crates/jolt-prover-legacy/Cargo.toml`
- this experiment directory

The benchmark parser, existing traces, and prover code outside the field module
are read-only until a kernel clears the promotion gate.

Maximum kernel candidates: six. Keep one candidate at a time. Each attempt gets
one append-only event with `keep`, `discard`, or `crash`.
