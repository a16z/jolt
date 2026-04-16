---
name: minimal abstractions / occam's razor
description: User strongly prefers minimal abstractions — don't cargo-cult existing patterns, think from first principles
type: feedback
---

Don't copy existing abstractions into new crates. Start from what the system actually DOES, not how the old code organized it.

**Why:** The user is building a clean-room rewrite and wants the simplest possible generalized system. Existing code has accumulated unnecessary complexity (KernelEvaluator, ProtocolGraph, StageBlock extraction, etc.) that should not be replicated.

**How to apply:** When designing new crates or rewriting modules:
1. Ask "what does this actually compute?" not "what does the old code look like?"
2. Inline logic that's only used once — no premature abstraction
3. Spartan, Toom-Cook, etc. are just sumcheck stages with different formulas — treat them uniformly
4. A "prover" is a loop: pairwise_reduce → round poly → transcript → squeeze → bind. Period.
5. The user wants to dissolve jolt-spartan — Spartan becomes just another compiler-emitted stage
