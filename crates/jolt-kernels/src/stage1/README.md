Stage 1 kernel modules
======================

`stage1.rs` owns the generated-code ABI: static plans, sumcheck execution,
proof verification, and the generic R1CS-backed fallback evaluator.

`rv64_typed.rs` owns the RV64 coarse CPU specialization. It must remain a
semantic refinement of the generic R1CS evaluator: typed oracle evaluations and
sumcheck products are tested against the R1CS column path before equivalence
tests compare Bolt against jolt-core.

Future Stage 1 kernels should follow the same shape:

- Keep protocol scheduling in `stage1.rs`.
- Put workload-specific arithmetic in a focused typed module.
- Preserve a generic fallback or parity oracle for new specializations.
- Add a local typed-vs-generic test before wiring core equivalence.
