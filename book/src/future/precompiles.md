# Precompiles
Precompiles are highly optimized SNARK gadgets which can be invoked from the high-level programming language of the VM user. These gadgets can be much more efficient for the prover than compiling down to the underlying ISA by exploiting the structure of the workload. In practice zkVMs use these for heavy cryptographic operations such as hash functions, signatures and other elliptic curve arithmetic.

By popular demand, Jolt will support these gadgets as well. The short term plan is to optimize for minimizing Jolt-core development resources rather than optimal prover speed.

Precompile support plan:
1. RV32 library wrapping syscalls of supported libraries
2. Tracer picks up syscalls, sets relevant flag bits and loads memory accordingly
3. Individual (uniform) Spartan instance for each precompile, repeated over `trace_length` steps
4. Jolt config includes which precompiles are supported (there is some non-zero prover / verifier cost to including an unused precompile)
5. Survey existing hash / elliptic curve arithmetic R1CS arithmetizations. Prioritize efficiency and audits.
6. Use [circom-scotia](https://github.com/lurk-lab/circom-scotia) to convert $A, B, C$ matrices into static files in the Jolt codebase
7. Write a converter to uniformly repeat the constraints `trace_length` steps

See the documentation on Jolt's use/handling of [sparse constraint systems](./how/sparse-constraint-systems.md) for a detailed overview of how the Jolt proof machinery will incorporate these pre-compiled constraint systems.

*TODO(sragss): How do we deal with memory and loading more than 64-bits of inputs to precompiles.*
