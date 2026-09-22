# Spartan verifier

This crate owns the checked sparse-R1CS key, clear proof format, transcript
sequence, and PCS-generic verification. See the
[protocol and source map](../jolt-spartan-prover/README.md) for the standalone
contract, matrix costs, tests, and unresolved cryptographic proof obligations.

The verifier has no dependency on `jolt-spartan-prover` or a concrete PCS.
Construct `SpartanKey` only from authenticated relation/policy data; pass the
application's public inputs and trusted PCS verifier setup to `verify`.
