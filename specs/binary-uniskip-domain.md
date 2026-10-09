# Spec: An Evaluation Domain for Univariate Skip over Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt's univariate-skip round sums a univariate polynomial over a small domain of consecutive integers, and its provers and verifier compute Lagrange bases over that domain. In characteristic 2 the integers collapse to $\{0, 1\}$, so `specs/binary-sumcheck.md` makes `CenteredIntegerDomain` return an error there. This spec adds the replacement: a domain whose points are elements of `F8`, embedded in the proof field by `From<F8>` (`specs/binary-f8.md`), with the round-sum rule and the Lagrange helpers that a univariate-skip round needs. The change is additive: no existing function changes. Wiring the domain into the Spartan stages is a later change.

## Intent

### Goal

Give `jolt-poly` Lagrange evaluation and interpolation over an arbitrary list of distinct nodes, give it the list of nodes of the `F8` domain, and give `jolt-sumcheck` a `SumcheckDomain` over those nodes.

The **`F8` domain of size $n$**, for $1 \le n \le 256$, is the ordered list of points

$$p_i = \iota(\texttt{F8::from\_raw}(i)), \qquad i = 0, \ldots, n-1,$$

where $\iota$ is `From<F8>` for the proof field. Its points are the first $n$ elements of `F8` in raw order.

The public surface is exactly this:

```rust
// jolt_poly::lagrange, no feature
pub enum LagrangeNodesError {
    EmptyNodes,
    RepeatedNode { first: usize, second: usize },
    LengthMismatch { nodes: usize, values: usize },
}
pub fn lagrange_evals_at_nodes<F: Field>(nodes: &[F], r: F) -> Result<Vec<F>, LagrangeNodesError>;
pub fn interpolate_nodes_to_coeffs<F: Field>(nodes: &[F], values: &[F]) -> Result<Vec<F>, LagrangeNodesError>;

// jolt_poly::lagrange, feature `binary`
pub const F8_DOMAIN_MAX_SIZE: usize = 256;
pub struct F8DomainSizeError { pub size: usize }
pub fn f8_domain_nodes<F: Field + From<F8>>(size: usize) -> Result<Vec<F>, F8DomainSizeError>;

// jolt_sumcheck::domain, feature `binary`, re-exported at the crate root as `CenteredIntegerDomain` is
pub struct F8Domain { /* size */ }
impl F8Domain { pub const fn new(size: usize) -> Self; pub const fn size(self) -> usize; }
impl<F: Field + From<F8>> SumcheckDomain<F> for F8Domain { /* invariant 4 */ }

// jolt_sumcheck::SumcheckError, last variant, present with and without the feature
InvalidF8Domain { size: usize }
```

No node-generic version of `centered_lagrange_kernel` is added; it gets one with its first caller.

### Invariants

1. **Distinct points.** The $n$ points are distinct in every field whose `From<F8>` is a field embedding, because an embedding is injective. The bound `From<F8>` cannot express that, so the rustdoc of `f8_domain_nodes` and of `F8Domain` states it as a requirement on the caller's field; the implementations for `F64`, `F128` and `F192` satisfy it by `specs/binary-f8.md`. A size of 0 or above 256 is a typed error, never a panic and never a truncated domain. The rule $1 \le n \le 256$ is checked in one place, `f8_domain_nodes`, before any index is converted to `u8`.
2. **Prefix property.** The domain of size $n$ is a prefix of the domain of size $n' > n$, in the same order. A round that evaluates on an extended domain therefore reuses the base domain's points as its first $n$.
3. **Subspace at powers of two.** For $n = 2^k$ with $0 \le k \le 8$ the points form the $\mathbb F_2$-subspace spanned by $\iota(x^0), \ldots, \iota(x^{k-1})$. For $0 \le k \le 7$ the next $2^k$ points are its coset by $\iota(x^k)$, in the same order: $p_{2^k + i} = p_i + p_{2^k}$. At $k = 8$ the subspace is all of `F8` and there is no next block. This is the domain shape of the binary-field systems cited in `specs/binary-sumcheck.md`; nothing in this spec uses the subspace structure beyond a test. A later kernel that needs $2n - 1$ distinct points for a base domain of size $n$ has them only for $n \le 128$.
4. **Round-sum rule.** For the `F8` domain $D$ of size $n$ and a round polynomial $s(X) = \sum_k c_k X^k$, the round check is $\sum_{p \in D} s(p) = \sum_k c_k S_k$ with $S_k = \sum_{p \in D} p^k$ and $S_0 = n \cdot 1$. `round_sum_coefficients(degree)` returns $S_0, \ldots, S_{\text{degree}}$, computed in the proof field. It validates in this order: the size, returning `InvalidF8Domain`; then `degree + 1` by checked addition, returning the existing `DegreeOverflow`; then it allocates. `F8Domain::new` is infallible, as `CenteredIntegerDomain::new` is.
5. **Node-generic Lagrange.** For distinct nodes $x_0, \ldots, x_{N-1}$ in any field, `lagrange_evals_at_nodes` returns $L_i(r)$ for all $i$, which is the indicator vector when $r$ is a node, and `interpolate_nodes_to_coeffs` returns the $N$ monomial coefficients, low degree first and with trailing zeros kept, of the unique polynomial of degree below $N$ through the given values. Both check the node list before anything else: an empty list is `EmptyNodes`; two equal nodes are `RepeatedNode { first, second }` with `first < second` the lexicographically smallest pair of positions that hold equal nodes, also when $r$ equals the repeated node; interpolation with a different number of values is `LengthMismatch`, checked after the node list. Neither function panics on any input.
6. **Integer-domain functions are not modified.** `lagrange_evals`, `centered_lagrange_evals`, `centered_lagrange_kernel`, `interpolate_to_coeffs`, `centered_power_sums` and `centered_domain_start` keep their bodies, their panics and their behaviour on every input, including integer nodes that collide in the field, where `lagrange_evals` returns the indicator of the first matching node. They are on the path of the shipped Spartan provers, the shared claims and the ZK lowering, none of which this spec touches.
7. **No protocol change.** No stage, kernel, proof type or transcript changes. `SumcheckDomainSpec`, which is implemented for every field, gains no variant. `SumcheckError` gains one variant after the existing ones, declared without a feature gate so that the enum is the same type in every build.

### Non-Goals

- Using the `F8` domain in the Spartan outer and product stages. Their kernels, `jolt-claims` and `jolt-verifier`'s `uniskip` name the centered integer domain directly, and `prove_uniskip_clear` and `prove_uniskip_committed` take a domain size. Making them generic in the domain belongs with the first binary-field instantiation of those stages.
- The committed recorder and the R1CS lowering over binary fields. They stay unsupported, as `jolt-sumcheck`'s module documentation says.
- Unifying the integer-domain functions with the node-generic ones. It would change the bodies that invariant 6 leaves alone and needs a before-and-after measurement on the provers that call them.
- An additive NTT, subspace-polynomial evaluation, or any method that is faster than $O(N^2)$ on a subspace. Jolt's skip domains have at most a few dozen points.
- Domains with more than 256 points.
- A soundness-motivated choice of which coordinates of the eq polynomial are fixed. That concerns the zerocheck formulation of the systems surveyed, and Jolt's univariate skip sums over the domain.

## Evaluation

### Acceptance Criteria

- [ ] The public surface is the one listed under Goal: same names, signatures, bounds and feature gates, and nothing else is exported. `jolt-poly` gains a `binary` feature that forwards to `jolt-field/binary`, and `jolt-sumcheck` gains a `binary` feature that forwards to `jolt-poly/binary` and `jolt-field/binary`.
- [ ] Node-generic basis, tested over `Fr` and over `F64` on a node list that is neither consecutive nor sorted, and on a single node: $L_i(x_j) = \delta_{ij}$; and polynomial reproduction, $\sum_i L_i(r) f(x_i) = f(r)$ at a non-node $r$, for $f$ given by fixed coefficients of degree $N - 1$ and evaluated by Horner's rule.
- [ ] Node-generic interpolation, over the same fields and node lists: for fixed coefficients of degree $N - 1$ and of degree $N - 3$, interpolating the Horner values returns those coefficients, with $N$ entries in both cases.
- [ ] Node-list errors, for both functions: an empty list; a repeated node with $r$ a non-node; a repeated node with $r$ equal to it; and, for interpolation, a value count that differs from the node count.
- [ ] Integer nodes over `Fr`: for the nodes `F::from_i64(s + k)` with one centered and one non-centered start, the node-generic interpolation and `interpolate_to_coeffs` each return the fixed coefficients from which the values were computed.
- [ ] `F8` nodes, for `F64`, `F128` and `F192`: `f8_domain_nodes(n)[i]` equals `F::from(F8::from_raw(i))`, written out in the test, for every $i < n$ at $n = 8$ and $n = 256$; sizes 1, 2, 10, 64, 256 give that many distinct points; size 64 is a prefix of size 128; sizes 0, 257 and `usize::MAX` are errors; point 2 is the constant $\beta$ of `specs/binary-f8.md` for `F64` and `F128`.
- [ ] Coset order, for the three fields and $k = 2$ and $k = 7$: $p_{2^k + i} = p_i + p_{2^k}$ for all $i < 2^k$.
- [ ] Round-sum rule, for the three fields and sizes 1, 2, 3, 10, 64: for each degree in $\{0, 1, n - 1, n, 2n\}$ (distinct values only, and none below 0), a polynomial of exactly that degree with fixed nonzero coefficients built from raw words satisfies $\sum_k c_k S_k = $ the sum of its Horner evaluations at the points.
- [ ] Subspace power sums, for the three fields and $k = 1, \ldots, 6$: $S_j = 0$ for $0 \le j < 2^k - 1$, and $S_{2^k-1}$ equals the product of the nonzero points, which is nonzero.
- [ ] `F8Domain` errors: sizes 0, 257 and `usize::MAX` give `InvalidF8Domain`; size 4 with degree `usize::MAX` gives `DegreeOverflow`; size 0 with degree `usize::MAX` gives `InvalidF8Domain`.
- [ ] One-round reduction, for the three fields, with domain size 4 and a round polynomial of degree 5 with fixed nonzero coefficients, built directly as a full clear round (the uni-skip prover entry points remain integer-only): the `SumcheckProof` verifies against `F8Domain` with the claim computed by direct summation, and the returned value equals the Horner evaluation at the returned challenge. Changing $c_3$ by a fixed nonzero element is rejected with `RoundCheckFailed`. Changing $c_0$ by one, for which $S_0 = 0$, verifies, and the returned value differs from the Horner evaluation of the original polynomial at the challenge of that run: the round check alone does not catch it, and the difference is what a later evaluation check catches.
- [ ] `CenteredIntegerDomain` still returns `IntegerDomainNotDistinct` over the binary fields; the existing test is unchanged and not duplicated.
- [ ] No existing test in `jolt-poly` or `jolt-sumcheck` changes, and `git diff` shows no change inside the body of any function named in invariant 6.
- [ ] `cargo check -p jolt-poly --lib` and `cargo check -p jolt-sumcheck --lib` pass with default features, so that no item names `F8` outside the feature. `cargo clippy --all-targets -- -D warnings` passes for `jolt-poly` and for `jolt-sumcheck` with default features, with `--features binary`, and for `jolt-sumcheck` with `--features binary,committed,r1cs`. `cargo nextest run -p jolt-poly --cargo-quiet` and `-p jolt-sumcheck`, each with and without `--features binary`, pass. `cargo fmt --check` passes.

### Testing Strategy

Ground truth is algebraic and independent of the code under test: the Kronecker-delta property and polynomial reproduction for a Lagrange basis, coefficient round trips against Horner evaluation, the definition of the nodes written out next to the function that produces them, direct summation for the round-sum rule, and the classical vanishing of low power sums over an $\mathbb F_2$-subspace. Partition of unity is not tested on its own: reproduction of a polynomial with nonzero constant term implies it, and a basis that returns the first indicator everywhere satisfies it.

No test compares an old body with a new one, because no body is replaced. The integer-nodes criterion checks two interpolation algorithms that both ship against the same known coefficients, not against each other.

### Performance

None claimed and none at risk. No existing function body changes, so no path of a shipped prover or verifier changes. No shipped configuration selects `F8Domain`.

## Design

### Architecture

`jolt_poly::lagrange` already separates the rule that fixes the domain (`centered_domain_start`) from the algebra over it, and `jolt-sumcheck`'s `CenteredIntegerDomain` wraps the first. The `F8` domain follows the same layering: the node rule in `jolt-poly`, the `SumcheckDomain` in `jolt-sumcheck`. The node-generic algebra needs no feature because it never names `F8`. Errors stay in the layer that raises them: `jolt-poly` has its own two error types, which implement `Display` and `std::error::Error` as `CenteredIntegerDomainError` does, and does not depend on `jolt-sumcheck`; `F8Domain` maps `F8DomainSizeError` to `SumcheckError::InvalidF8Domain`.

Both node-generic functions do $O(N^2)$ multiplications and $N$ inversions, plus the pairwise check for repeated nodes, which compares elements because a field has no order to sort by. The basis is evaluated from the product formula, which gives the indicator at a node without a special case. Interpolation forms the vanishing polynomial of the nodes once and adds, for each node, its quotient by that node's linear factor, scaled.

The points are taken in raw order for three reasons. The order is a rule that a reader can state in a line. Extended domains are prefixes, which the centered integer domain does not give. And at powers of two the domain is the subspace-and-coset pair that an additive NTT would need, should a larger skip ever call for one.

### What the round check gives

For a subspace domain most of the $S_k$ vanish, so the check $\sum_k c_k S_k = \text{claim}$ constrains few coefficients of the round polynomial. That does not weaken the round as a reduction. The check is one linear condition on the prover's polynomial in every sum-check. A false claim fails because the honest polynomial does not satisfy the condition with the false claim, so the prover must send a different polynomial, and the two then differ at the challenge unless the challenge is a root of their difference. For a fixed nonzero difference of degree at most $d$, that happens with probability at most $d \cdot \max_a \Pr[r = a]$. Neither step depends on how many coefficients the condition touches.

Two qualifications belong with that bound. First, it is $d / |F|$ only for a challenge uniform on the field. Jolt's stock transcripts draw 16 bytes, which `specs/binary-field.md` maps to a set of $2^{64}$ elements of `F64` and of $2^{128}$ elements of `F128` and of `F192`; this spec changes no transcript. Second, the round reduces the claim to an evaluation of the round polynomial at the challenge and does not check that evaluation. The verifier's skip stage checks it against the opening claims it is given, and those are discharged by the rest of the protocol. The statement above is the soundness of one reduction step, not a Fiat–Shamir bound for a proof.

The sparse moments do matter for wire formats: a compressed round that omits a coefficient and recovers it from the round sum cannot omit one whose $S_k$ is zero. This spec adds no compressed format for the skip round. Today the clear skip round transmits every coefficient, and the committed skip round commits to the full coefficient vector and proves its sum and its evaluation in BlindFold; neither reconstructs an omitted coefficient from a moment. The compressed clear verifier is restricted to the Boolean hypercube, where $S_1 = 1$.

### Alternatives Considered

1. **A subspace-only domain type.** It would force skip sizes to powers of two. Jolt's sizes come from constraint counts.
2. **Points as successive powers of a generator.** Also distinct, but the subspace structure and the prefix-as-coset property are lost.
3. **A variant of `SumcheckDomainSpec`.** The enum implements `SumcheckDomain<F>` for every field, and the new domain exists only where `From<F8>` does. A variant would need either a bound on the enum's impl, which breaks its prime-field users, or a runtime error for fields without the embedding.
4. **Re-expressing `lagrange_evals` through the node-generic evaluator**, to keep one body for the barycentric formula. Rejected for this change. The integer function returns the indicator of the first matching node before it looks at any denominator, so on integer nodes that collide in the field it succeeds where an evaluator that validates its nodes must fail; `lagrange_evals(0, 3, 0)` over `F64` is such a call. And the function is called by the shipped Spartan provers, so replacing its body is a performance-relevant change that this spec has no reason to make.
5. **A fallible `F8Domain::new`.** `SumcheckError` is generic in the field and the constructor is not, and `CenteredIntegerDomain` already validates in `round_sum_coefficients`. The same shape keeps the two domains interchangeable for callers.
6. **A feature-gated `SumcheckError` variant.** It would make the enum a different type with and without `binary`, and every exhaustive match downstream would need the same gate.

## Documentation

Module docs of `jolt_poly::lagrange` (no longer "over integer domains" only) and of `jolt-sumcheck`'s "Binary fields" section, which names `F8Domain` as the domain for sizes above 2 and keeps the sentence on the committed recorder and the R1CS lowering. No book change.

## Execution

Node-generic Lagrange with its error type; the `F8` nodes behind the feature; `F8Domain` and the error variant; tests. One commit.

## References

- `specs/binary-sumcheck.md`, Prior Art, and `specs/binary-f8.md`.
- `crates/jolt-poly/src/lagrange.rs`; `crates/jolt-sumcheck/src/domain.rs`; `crates/jolt-verifier/src/stages/uniskip.rs`.
