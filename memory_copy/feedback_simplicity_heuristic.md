---
name: Simplicity heuristic for design decisions
description: Whenever something feels complex, stop and find the simpler abstraction. Production-grade code quality target.
type: feedback
originSessionId: f0719e20-9c07-4478-bf0f-7ad6f4e1a5cf
---
Whenever something feels "complex" or is getting complex, stop and think if there's a nicer/simpler way to implement the abstraction or pattern.

**Why:** The user values Occam's razor applied to software design. The codebase should be the "ML compiler of cryptography" — simple, effective abstractions that are immediately understandable. Production-grade means: idiomatic comments, all tests passing, no placeholders, no janky code, no hacky workarounds. If something looks like an anti-pattern, it needs to be addressed.

**How to apply:** Before implementing, ask: "Is there a simpler way?" After implementing, review: "Could this be simpler?" Each abstraction should have a clear, single purpose. Prefer fewer layers. Prefer explicit over clever. Three lines of direct code > one line of abstraction that requires context to understand.
