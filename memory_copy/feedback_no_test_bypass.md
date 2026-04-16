---
name: No test-harness bypass for polynomial data
description: All polynomial construction must live in jolt-zkvm crates, not be injected from test harness
type: feedback
---

Polynomial data needed by the jolt-zkvm prover must be produced by the runtime/Module infrastructure itself. Never bypass by constructing polynomials in the test harness and injecting them.

**Why:** The Module and runtime need to be self-sufficient — if the prover needs a polynomial, jolt-zkvm must know how to build it from the data it receives. Test-harness injection hides missing functionality and creates false confidence.

**How to apply:** When a new polynomial is needed (e.g., RamCombinedRa, RamVal), implement the construction logic inside a jolt-zkvm crate (DerivedSource, PreprocessedSource, or a new module). The test should only provide raw inputs (trace, memory state) through proper interfaces that the runtime also uses.
