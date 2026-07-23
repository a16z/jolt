# Test-quality rubric

Scoring framework for the automated test-quality review defined in
`specs/test-quality-ci.md`. The reviewer scores **every test function added or
modified by a PR** on five dimensions, 0–2 each, summed to a 1–10 score
(a sum of 0 is reported as 1). Tests scoring below the threshold in
`ci/test-quality.toml` get a **rework-or-delete** suggestion: a low-quality
test is worse than no test, because it converts coverage into false assurance.

The context is a zkVM: systems Rust plus cryptographic protocols. Soundness
bugs here do not crash — they accept things that should be rejected. Score
accordingly: the central question for every test is *"what class of bug would
this test catch, and would it catch it for the right reason?"*

## Dimensions

### 1. Oracle strength (0–2)

Where does the expected value come from?

- **2** — An oracle independent of the code under test: a differential
  reference (naive implementation, wide-integer arithmetic, a different
  algorithm), an algebraic law (`x.square() == x * x`, commutativity,
  Karatsuba vs schoolbook), a pinned known-answer vector, or an emulator /
  golden fixture produced outside the unit under test.
- **1** — Structural or shape assertions (dimensions, ordering, round-trip
  through the same library) that would catch gross breakage but not a wrong
  value.
- **0** — Tautological: the expected value is computed by the same code path
  being tested, or the assertion restates the implementation.

### 2. Adversarial reach (0–2)

Which inputs does the test explore?

- **2** — Boundaries and hostile cases: zero/one/max, modulus boundaries,
  carry chains, empty and maximal collections, malformed or tampered inputs,
  truncated encodings; randomized inputs that can actually reach edge cases.
- **1** — Some randomization or at least one deliberate edge case beyond the
  happy path.
- **0** — A single benign input.

### 3. Failure specificity (0–2)

When the test expects a failure, does it check the *right* failure?

- **2** — Asserts the exact error variant, rejection stage, or panic message;
  a tampered proof must be rejected by the check that guards the tampered
  field, not by an accidental earlier failure.
- **1** — Asserts `is_err()` / `#[should_panic]` without discriminating the
  cause.
- **0** — Could pass for the wrong reason entirely (e.g. asserts an error
  while the setup itself is what fails), or asserts nothing meaningful.

For tests with no failure path (pure happy-path property checks), score this
dimension on whether the assertion would localize a failure: comparing whole
structures with a meaningful `assert_eq!` scores 2; asserting `result.is_ok()`
alone scores 1.

### 4. Independence & determinism (0–2)

Would the test survive a refactor, and does it fail only when something is
wrong?

- **2** — Deterministic (seeded RNG, no wall-clock or ordering dependence),
  exercises public behavior rather than mirroring private implementation
  structure, isolated from other tests.
- **1** — Deterministic but tightly coupled to implementation details that a
  behavior-preserving refactor would break.
- **0** — Flaky potential (unseeded randomness, timing, filesystem/env
  dependence, inter-test ordering), or copies the implementation line-for-line
  so both drift together.

### 5. Property clarity (0–2)

Can a reviewer tell what correctness property dies if this test is deleted?

- **2** — The name and body state a specific property ("rejects proof with
  tampered stage-3 opening claim", "RVC C.ADDI4SPN expansion matches the
  RV64 reference"); a failure message would point at the broken invariant.
- **1** — The intent is recoverable with effort.
- **0** — Unclear what is being protected; grab-bag tests asserting many
  unrelated things; names like `test_works`.

## Crypto-specific tripwires

Apply these regardless of dimension scores; each listed violation caps the
total at 5 unless the test explicitly addresses it:

- **Self-consistency trap** — a prover/verifier (or serializer/deserializer,
  encoder/decoder) round-trip is *not* an oracle when both sides share the
  code under test. A transcript or wire-format bug that is symmetric passes
  every round-trip. Such tests need a pinned vector or cross-implementation
  check to score above 5.
- **Wrong-object tampering** — a tamper/negative test must mutate the object
  that the targeted check actually guards, and assert rejection by that check.
  Tampering a field that fails deserialization before verification even runs
  proves nothing about the verifier.
- **Unseeded randomness** — randomized crypto tests must use a fixed seed (or
  log the seed on failure). A test that cannot be reproduced cannot be
  debugged.
- **Trivial-domain sampling** — random field elements drawn only from a tiny
  range (e.g. `0..100`) do not exercise reduction, carries, or high-limb
  paths; say so and score adversarial reach 0.

## Output format

The reviewer emits JSON conforming to:

```json
{
  "tests": [
    {
      "name": "module::tests::rejects_tampered_stage3_claim",
      "file": "crates/jolt-verifier/tests/soundness/tampering/sumcheck.rs",
      "change": "added",
      "dimensions": {
        "oracle_strength": 2,
        "adversarial_reach": 2,
        "failure_specificity": 2,
        "independence_determinism": 2,
        "property_clarity": 2
      },
      "tripwires": [],
      "score": 10,
      "verdict": "keep",
      "rationale": "one sentence citing the decisive dimension(s)"
    }
  ]
}
```

- `change` is `added` or `modified`. Unchanged tests are not scored.
- `score` = max(1, sum of dimensions), then capped at 5 if any tripwire fired.
- `verdict` is `keep` (score ≥ threshold) or `rework-or-delete`.
- `tripwires` lists fired tripwire names, empty otherwise.

## Calibration examples

- A lookup-table test running the shared `mle_full_hypercube_test` harness
  (exhaustive domain, independent MLE reference): 2/2/2/2/2 → **10**.
- `assert!(verify(tampered_proof).is_err())` with a well-named test: oracle 2
  (rejection is the property), reach 2, specificity **1**, independence 2,
  clarity 2 → **9**; the fix to specificity (assert the variant) is exactly
  what the sweep hardened.
- A round-trip `serialize(deserialize(x)) == x` with no pinned bytes:
  self-consistency tripwire → capped at **5** → rework (add a golden vector).
- `let r = f(3); assert_eq!(r, f(3));`: 0/0/0/0/0 → **1** → delete.
