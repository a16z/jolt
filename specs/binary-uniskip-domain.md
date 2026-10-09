# Spec: An Evaluation Domain for Univariate Skip over Binary Fields

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Quang Dao, Claude              |
| Created     | 2026-10-09                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

Jolt's univariate-skip round sums a univariate polynomial over a small domain of consecutive integers, and its provers and verifier compute Lagrange bases over that domain. In characteristic 2 the integers collapse to $\{0, 1\}$, so `specs/binary-sumcheck.md` makes `CenteredIntegerDomain` return an error there. This spec adds the replacement: a domain whose points are elements of `F8`, embedded in the proof field by `From<F8>` (`specs/binary-f8.md`), with the round-sum rule and the Lagrange helpers that a univariate-skip round needs. Wiring the domain into the Spartan stages is a later change.

## Intent

### Goal

Give `jolt-poly` Lagrange evaluation and interpolation over an arbitrary list of distinct nodes, give it the list of nodes of the `F8` domain, and give `jolt-sumcheck` a `SumcheckDomain` over those nodes.

The **`F8` domain of size $n$**, for $1 \le n \le 256$, is the ordered list of points

$$p_i = \iota(\texttt{F8::from\_raw}(i)), \qquad i = 0, \ldots, n-1,$$

where $\iota$ is `From<F8>` for the proof field. Its points are the first $n$ elements of `F8` in raw order.

### Invariants

1. **Distinct points.** The $n$ points are distinct in every field with `From<F8>`, because $\iota$ is injective. A size of 0 or above 256 is a typed error, never a panic and never a truncated domain.
2. **Prefix property.** The domain of size $n$ is a prefix of the domain of size $n' > n$, in the same order. A round that evaluates on an extended domain therefore reuses the base domain's points as its first $n$.
3. **Subspace at powers of two.** For $n = 2^k$ the points form the $\mathbb F_2$-subspace spanned by $\iota(x^0), \ldots, \iota(x^{k-1})$, and the next $2^k$ points are its coset by $\iota(x^k)$. This is the domain shape of the binary-field systems cited in `specs/binary-sumcheck.md`; nothing in this spec uses the subspace structure beyond a test.
4. **Round-sum rule.** For the `F8` domain $D$ of size $n$ and a round polynomial $s(X) = \sum_k c_k X^k$, the round check is $\sum_{p \in D} s(p) = \sum_k c_k S_k$ with $S_k = \sum_{p \in D} p^k$ and $S_0 = n \cdot 1$. `round_sum_coefficients(degree)` returns $S_0, \ldots, S_{\text{degree}}$, computed in the proof field.
5. **Node-generic Lagrange.** For distinct nodes $x_0, \ldots, x_{N-1}$ in any field: the basis evaluation returns $L_i(r)$ for all $i$, equal to the indicator vector when $r$ is a node; interpolation returns the monomial coefficients of the unique polynomial of degree below $N$ through the given values. Repeated nodes are a typed error.
6. **Integer domains are unchanged.** Every existing function of `jolt_poly::lagrange` returns the same field elements as before, in every field, and no existing caller changes. `lagrange_evals` is re-expressed through the node-generic basis evaluation, which performs the same field operations up to how a node difference is formed. `interpolate_to_coeffs` and `centered_power_sums` keep their own bodies: the first uses that consecutive nodes make every divided-difference denominator of one step equal, which costs $N-1$ inversions against $N(N-1)/2$, and the second is exact integer arithmetic.
7. **No protocol change.** No stage, kernel, proof type or transcript changes. `SumcheckDomainSpec`, which is implemented for every field, gains no variant.

### Non-Goals

- Using the `F8` domain in the Spartan outer and product stages. Their kernels, `jolt-claims` and `jolt-verifier`'s `uniskip` name the centered integer domain directly, and `prove_uniskip_clear` and `prove_uniskip_committed` take a domain size. Making them generic in the domain belongs with the first binary-field instantiation of those stages.
- An additive NTT, subspace-polynomial evaluation, or any method that is faster than $O(N^2)$ on a subspace. Jolt's skip domains have at most a few dozen points.
- Domains with more than 256 points.
- A soundness-motivated choice of which coordinates of the eq polynomial are fixed. That concerns the zerocheck formulation of the systems surveyed, and Jolt's univariate skip sums over the domain.

## Evaluation

### Acceptance Criteria

- [ ] `jolt_poly::lagrange` has node-generic basis evaluation and interpolation taking `&[F]` nodes, for `F: Field`, with no feature gate, returning a typed error on an empty node list, on repeated nodes, and on a length mismatch between nodes and values.
- [ ] `jolt_poly::lagrange`, under a new `binary` feature that forwards to `jolt-field/binary`, has the one function that produces the nodes of the `F8` domain of a given size for `F: Field + From<F8>`, with a typed error for sizes 0 and above 256.
- [ ] `jolt-sumcheck`, under a new `binary` feature that forwards to `jolt-poly/binary`, has `F8Domain`, constructed from a size, implementing `SumcheckDomain<F>` for `F: Field + From<F8>` by invariant 4. Its errors are `SumcheckError` variants appended after the existing ones.
- [ ] Node-generic Lagrange, tested over `Fr` and over `F64`: $L_i(x_j) = \delta_{ij}$; $\sum_i L_i(r) = 1$ at a non-node $r$; interpolation of the values of a fixed polynomial given by its coefficients returns those coefficients; the three error cases.
- [ ] Integer agreement over `Fr`: the node-generic functions applied to the nodes `F::from_i64(s + k)` return what `lagrange_evals` and `interpolate_to_coeffs` return, for one centered and one non-centered start. This pins invariant 6 for the function that keeps its own body.
- [ ] `F8` nodes, for `F64`, `F128` and `F192`: sizes 1, 2, 10, 64, 256 give that many distinct points; size 64 is a prefix of size 128; sizes 0 and 257 are errors; point 2 is the constant $\beta$ of `specs/binary-f8.md` for `F64` and `F128`.
- [ ] Round-sum rule, for the three fields and sizes 1, 2, 3, 10, 64: for a polynomial with fixed nonzero coefficients of degree 0, 1, size − 1, size, and 2·size, $\sum_k c_k S_k$ equals the sum of Horner evaluations at the points.
- [ ] Subspace power sums, for the three fields and $k = 1, \ldots, 6$: $S_j = 0$ for $0 \le j < 2^k - 1$, and $S_{2^k-1}$ equals the product of the nonzero points, which is nonzero.
- [ ] End to end, for the three fields: a one-round `SumcheckProof` whose round polynomial has degree above the domain size verifies against `F8Domain` with the claim computed by direct summation, the reduction value equals the polynomial at the challenge, and a proof with one coefficient changed is rejected with `RoundCheckFailed`.
- [ ] `CenteredIntegerDomain` still returns `IntegerDomainNotDistinct` over the binary fields; the existing test is unchanged.
- [ ] Existing `jolt-poly` and `jolt-sumcheck` tests pass unchanged. `cargo clippy --all-targets -- -D warnings` passes for `jolt-poly` and `jolt-sumcheck` with default features and with `--features binary`; `cargo nextest run -p jolt-poly --cargo-quiet` and `-p jolt-sumcheck`, each with and without `--features binary`, pass; `cargo fmt --check` passes.

### Testing Strategy

Ground truth is algebraic: the Kronecker-delta and partition-of-unity properties of a Lagrange basis, coefficient round trips against Horner evaluation, direct summation for the round-sum rule, and the classical vanishing of low power sums over an $\mathbb F_2$-subspace. The integer-agreement test compares two algorithms that both stay in the tree (general nodes and consecutive nodes), so it is not an old-against-new test.

### Performance

None claimed. `lagrange_evals` keeps its operation count. No shipped prover reaches the new code.

## Design

### Architecture

`jolt_poly::lagrange` already separates the rule that fixes the domain (`centered_domain_start`) from the algebra over it, and `jolt-sumcheck`'s `CenteredIntegerDomain` wraps the first. The `F8` domain follows the same layering: the node rule in `jolt-poly`, the `SumcheckDomain` in `jolt-sumcheck`. The algebra becomes node-generic because the binary-field points are not consecutive integers, and it needs no feature because it never names `F8`.

The points are taken in raw order for three reasons. The order is a rule that a reader can state in a line. Extended domains are prefixes, which the centered integer domain does not give. And at powers of two the domain is the subspace-and-coset pair that an additive NTT would need, should a larger skip ever call for one.

### Why the round check stays sound

For a subspace domain most of the $S_k$ vanish, so the check $\sum_k c_k S_k = \text{claim}$ constrains few coefficients of the round polynomial. That does not weaken the round. The check is one linear condition on the prover's polynomial in every sum-check; what makes a false claim fail is that the honest polynomial does not satisfy the condition with the false claim, so the prover must send a different polynomial, and two different polynomials of degree at most $d$ agree at the random challenge with probability at most $d / |F|$. Neither step depends on how many coefficients the condition touches.

It does matter for wire formats: a compressed round that omits a coefficient and recovers it from the round sum cannot omit one whose $S_k$ is zero. This spec adds no compressed format for the skip round, and Jolt's skip round is sent in full.

### Alternatives Considered

1. **A subspace-only domain type.** It would force skip sizes to powers of two. Jolt's sizes come from constraint counts.
2. **Points as successive powers of a generator.** Also distinct, but the subspace structure and the prefix-as-coset property are lost.
3. **A variant of `SumcheckDomainSpec`.** The enum implements `SumcheckDomain<F>` for every field, and the new domain exists only where `From<F8>` does. A variant would need either a bound on the enum's impl, which breaks its prime-field users, or a runtime error for fields without the embedding.
4. **Leaving the integer functions alone and adding parallel node-generic ones.** Two bodies for the barycentric formula. Rejected for `lagrange_evals`; accepted, with the reason in invariant 6, for interpolation.

## Documentation

Module docs of `jolt_poly::lagrange` (no longer "over integer domains" only) and of `jolt-sumcheck`'s "Binary fields" section. No book change.

## Execution

Node-generic Lagrange and the re-expression of `lagrange_evals`; the `F8` nodes behind the feature; `F8Domain`; tests. One commit.

## References

- `specs/binary-sumcheck.md`, Prior Art, and `specs/binary-f8.md`.
- `crates/jolt-poly/src/lagrange.rs`; `crates/jolt-sumcheck/src/domain.rs`; `crates/jolt-verifier/src/stages/uniskip.rs`.
