# Zero Knowledge: BlindFold

Jolt achieves zero-knowledge natively via the **BlindFold** protocol — a folding-based scheme that makes all sumcheck proofs ZK without SNARK composition. Unlike most zkVMs that wrap their IOP in Groth16 or Plonk to get ZK, BlindFold operates at the same algebraic level as Jolt's existing sumcheck machinery, avoiding composition overhead entirely.

The core idea: instead of the prover sending sumcheck round polynomial coefficients in the clear, it sends **Pedersen commitments** to them. The sumcheck verifier's algebraic consistency checks are encoded into a small **verifier R1CS** circuit, and a single Nova fold + Spartan proof over this R1CS proves all rounds were executed correctly without revealing the witness.

## Background

In standard (non-ZK) Jolt, each sumcheck round reveals the round polynomial's coefficients — typically 4 field elements per round for degree-3 sumchecks. The verifier checks:

1. **Round consistency**: $g_j(0) + g_j(1) = \text{claim}_j$
2. **Chaining**: $\text{claim}_{j+1} = g_j(r_j)$
3. **Final binding**: the last round's evaluation matches a claimed polynomial opening

These checks are purely algebraic — linear and quadratic relations over field elements. This makes them amenable to R1CS encoding.

The ZK challenge is: how do we let the verifier perform these checks without seeing the coefficients?

## Protocol overview

BlindFold proceeds in six phases, executed after all of Jolt's sumcheck stages (Spartan, instruction lookups, RAM/register checking, bytecode, claim reductions) complete.

### Phase 1: ZK sumcheck rounds

During each of the 7 sumcheck stages, `prove_zk` replaces the standard sumcheck prover:

**Per round**, the prover:
1. Computes the batched univariate polynomial as usual
2. Commits to all coefficients via Pedersen: $C_j = \sum_i c_{j,i} \cdot G_i + \rho_j \cdot H$
3. Appends $C_j$ (not the coefficients) to the Fiat-Shamir transcript
4. Derives the challenge $r_j$ from the transcript
5. Caches coefficients $c_{j,i}$, blinding $\rho_j$, and commitment $C_j$

The verifier receives only commitments. It derives the same challenges (since it hashes the same commitments) but cannot check round consistency without knowing the coefficients. That verification is deferred to BlindFold.

Uni-skip first rounds (used in Spartan outer and product sumchecks) follow the same pattern: commit, derive challenge, cache.

### Phase 2: Verifier R1CS construction

Both prover and verifier deterministically construct a sparse R1CS that encodes the sumcheck verifier's checks. The R1CS operates over a witness vector $Z$ with a Hyrax grid layout:

$$Z = [u, W_0, W_1, \ldots]$$

where $u$ is the relaxation scalar (1 for non-relaxed instances) and $W$ is the witness arranged as an $R' \times C$ grid:

- **Rows $0 \ldots R_{\text{coeff}}$**: coefficient rows (one per sumcheck round, zero-padded to $C$ columns)
- **Rows $R_{\text{coeff}} \ldots R'$**: non-coefficient values (next claims, Horner intermediates, opening evaluations)

**Constraints encoded per sumcheck round:**

For standard sumcheck ($g$ of degree $d$):
- **Sum constraint**: $2c_0 + c_1 + c_2 + \cdots + c_d = \text{claim}$
- **Chain constraint**: $g(r_j) = \text{next\_claim}$, computed via Horner's method with auxiliary variables

For uni-skip rounds:
- **Power-sum constraint**: $\sum_k \text{PowerSum}[k] \cdot c_k = \text{claim}$, where $\text{PowerSum}[k] = \sum_{t \in D} t^k$ over the symmetric evaluation domain $D$

**Additional constraints:**
- **Final output binding**: At chain end, $\text{final\_claim} = \sum_j \alpha_j \cdot y_j$ relates the last sumcheck claim to batched polynomial evaluations
- **Input claim binding**: At chain start, verifies the initial claim is correctly derived from prior stage openings via sum-of-products constraints
- **PCS evaluation binding**: Relates Dory evaluation commitments to witness variables

**Baked public inputs**: Fiat-Shamir-derived values (challenges $r_j$, initial claims, batching coefficients) are embedded directly into R1CS matrix coefficients rather than appearing as witness variables. Both prover and verifier derive these identically from the transcript, so the resulting matrices $A$, $B$, $C$ are identical. This keeps the witness vector minimal.

### Phase 3: Nova folding

The prover holds a satisfying witness $Z_1$ for the verifier R1CS. To hide it:

1. **Sample random instance**: Generate a random satisfying pair $(Z_2, E_2)$ for the same R1CS (with random coefficients and consistent constraints)
2. **Compute cross-term**: $T = (AZ_1) \circ (BZ_2) + (AZ_2) \circ (BZ_1) - u_1 \cdot (CZ_2) - u_2 \cdot (CZ_1)$
3. **Commit cross-term rows**: Commit each row of $T$ (in $R_E \times C_E$ grid) with Pedersen
4. **Derive folding challenge**: $r \leftarrow \text{Transcript}$
5. **Fold**: $Z' = Z_1 + r \cdot Z_2$, $E' = E_1 + r \cdot T + r^2 \cdot E_2$, $u' = u_1 + r \cdot u_2$

The folded instance satisfies relaxed R1CS:

$$(A \cdot Z') \circ (B \cdot Z') = u' \cdot (C \cdot Z') + E'$$

The random instance acts as a one-time pad: knowing $Z'$ alone reveals nothing about $Z_1$.

Row commitments fold homomorphically. The round commitments from Phase 1 serve as the real instance's coefficient-row commitments, so no additional group operations are needed for those rows.

### Phase 4: Spartan outer sumcheck

Proves the folded relaxed R1CS is satisfied via the standard Spartan sumcheck:

$$\sum_{x \in \{0,1\}^{\log m}} \widetilde{\text{eq}}(\tau, x) \cdot \left[\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - u' \cdot \widetilde{Cz}(x) - \widetilde{E}(x)\right] = 0$$

where $\tau$ is a verifier challenge and the sum is over all $m$ constraints (padded to a power of two). The round polynomial has degree 3 ($\widetilde{\text{eq}} \cdot \widetilde{Az} \cdot \widetilde{Bz}$ is the cubic term).

After $\log m$ rounds, the verifier holds evaluation claims $\widetilde{Az}(r_x)$, $\widetilde{Bz}(r_x)$, $\widetilde{Cz}(r_x)$ at the challenge point $r_x$, plus $\widetilde{E}(r_x)$.

### Phase 5: Spartan inner sumcheck

Each of $\widetilde{Az}(r_x)$, $\widetilde{Bz}(r_x)$, $\widetilde{Cz}(r_x)$ decomposes as a public contribution (from the $u$ column and baked constants) plus a witness contribution. The inner sumcheck reduces the witness contributions to a single evaluation $W(r_y)$:

$$r_a \cdot w_{Az} + r_b \cdot w_{Bz} + r_c \cdot w_{Cz} = \sum_{j \in \{0,1\}^{\log |W|}} L_w(j) \cdot W(j)$$

where $L_w$ is a structured polynomial derived from the sparse R1CS matrices and outer challenges $r_x$, and $r_a, r_b, r_c$ are verifier challenges. The round polynomial has degree 2.

After $\log |W|$ rounds, the verifier needs $W(r_y)$.

### Phase 6: Hyrax-style openings

The witness $W$ and error $E$ are committed row-wise (as Pedersen commitments over the Hyrax grid). To prove evaluations at points $r_y$ and $r_x$ respectively:

**For $W(r_y)$**: Split $r_y = (r_{\text{row}}, r_{\text{col}})$ where $r_{\text{row}} \in \mathbb{F}^{\log R'}$, $r_{\text{col}} \in \mathbb{F}^{\log C}$.

1. Compute the **combined row**: $\bar{w} = \sum_{i=0}^{R'-1} \widetilde{\text{eq}}(r_{\text{row}}, i) \cdot \text{row}_i$ and combined blinding $\bar{\rho}$
2. Send $\bar{w}$ and $\bar{\rho}$ to verifier
3. Verifier checks: $\sum_i \widetilde{\text{eq}}(r_{\text{row}}, i) \cdot C_i \stackrel{?}{=} \text{Ped}(\bar{w}, \bar{\rho})$
4. Verifier computes: $W(r_y) = \sum_j \widetilde{\text{eq}}(r_{\text{col}}, j) \cdot \bar{w}_j$

The $E$ opening follows the same pattern over the $R_E \times C_E$ grid.

The coefficient-row commitments $C_0, \ldots, C_{R_{\text{coeff}}-1}$ are exactly the Pedersen commitments from Phase 1 — no additional commitments needed.

## Integration with Jolt

BlindFold sits at Stage 8 of the Jolt prover pipeline, after all sumcheck stages complete:

| Stage | Protocol | BlindFold role |
|-------|----------|----------------|
| 1 | Spartan (outer + product) | Collect round commitments, stage configs |
| 2 | Spartan (shift + instruction input) | Collect round commitments, stage configs |
| 3 | Instruction lookups (RA virtual + read-RAF) | Collect round commitments, stage configs |
| 4 | Bytecode (read-RAF) | Collect round commitments, stage configs |
| 5 | RAM (read-write + val-eval + val-final + output + booleanity + RAF) | Collect round commitments, stage configs |
| 6 | Register (read-write + val-eval) | Collect round commitments, stage configs |
| 7 | Claim reductions (instruction, register, RAM RA, increments, Hamming, advice) | Collect round commitments, stage configs |
| **8** | **Batched opening proof + BlindFold** | Build R1CS, fold, prove |

Each stage contributes:
- **Round commitments**: Pedersen commitments to sumcheck polynomial coefficients
- **Stage configs**: Round count, polynomial degree, chain structure, uni-skip power sums, output/input constraints

The `ProverOpeningAccumulator` collects `ZkStageData` — coefficients, blindings, challenges — across all stages.

### Dory ZK evaluation commitments

The Dory polynomial commitment scheme is extended with ZK evaluation commitments. Instead of revealing the polynomial evaluation $v = \tilde{f}(r)$ in the clear, the prover sends a Pedersen commitment $y_{\text{com}} = v \cdot G_0 + \rho \cdot H$ and proves consistency within the Dory opening proof.

BlindFold's extra constraints verify that $y_{\text{com}}$ matches the folded evaluation and blinding values, binding the PCS to the BlindFold witness.

## Verification

The verifier:

1. **Derives challenges** from round commitments (same Fiat-Shamir transcript)
2. **Constructs the verifier R1CS** deterministically from stage configs and baked public inputs
3. **Reconstructs the real instance** from round commitments (proof data) with $u=1$, $E=\mathbf{0}$
4. **Absorbs the random instance** from the proof
5. **Folds** the two instances using the cross-term commitments and derived challenge $r$
6. **Verifies Spartan outer sumcheck** ($\log m$ rounds, degree-3 polynomials)
7. **Verifies Spartan inner sumcheck** ($\log |W|$ rounds, degree-2 polynomials)
8. **Checks Hyrax openings**: verifies $E(r_x)$ and $W(r_y)$ against folded row commitments
9. **Checks evaluation commitments**: verifies $y_{\text{com}}$ consistency for each PCS opening

The verifier never sees polynomial coefficients, intermediate claims, or evaluation values from any sumcheck stage.

## Security

- **Hiding**: Pedersen commitments are computationally hiding under the discrete logarithm assumption. The prover's sumcheck messages reveal no information about the witness polynomials.
- **Binding**: Pedersen commitments are computationally binding. A malicious prover cannot open a commitment to different coefficients.
- **Folding soundness**: The folded instance satisfies relaxed R1CS if and only if both constituent instances do (with overwhelming probability over the folding challenge).
- **Extraction**: An extractor can recover the real witness from two accepting transcripts with different folding challenges (via rewinding).

## References

- [Hyrax](https://eprint.iacr.org/2017/1132.pdf) — Matrix-commitment-based polynomial evaluation proofs (Wahby et al., 2018)
- [Nova](https://eprint.iacr.org/2021/370) — Recursive SNARKs via folding (Kothapalli, Setty, Tzialla, 2022)
- [Spartan](https://eprint.iacr.org/2019/550) — Sumcheck-based R1CS proving (Setty, 2020)
- [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html), Section 13.2 — ZK sumcheck via Pedersen commitments (Thaler, 2022)
