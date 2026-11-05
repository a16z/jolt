# Zero Knowledge: BlindFold

BlindFold is available with Jolt's [Dory](./dory.md) backend by enabling the `zk` Cargo feature. The [Akita](./akita.md) backend currently supports only proofs without zero knowledge; combining `akita` and `zk` is rejected at compile time. This chapter describes the Dory ZK protocol.

Jolt achieves zero-knowledge natively via the **[BlindFold](https://eprint.iacr.org/2025/2094)** protocol — a folding-based scheme that makes all sumcheck proofs ZK without SNARK composition. Unlike approaches that require per-round sigma protocols or expensive homomorphic operations on commitments, BlindFold defers all verification to a single small R1CS instance proved via Nova folding + Spartan, keeping most of the work in the field arithmetic layer.

The core idea: the prover never sends a private value in the clear. Before the first challenge it commits a vector of random **pads**; every private message of the sumcheck stages (round coefficients, output claims, uni-skip rounds) goes out as $\hat m = m + p$, masked by its own pad. The verifier cannot check a round on masked values, so it states every check it would have run as a row of a **residual relation** over the pads, and a single Nova fold + Spartan proof over a small relaxed R1CS shows that the pads satisfy all of them.

## Background

In standard (non-ZK) Jolt, each sumcheck round reveals the round polynomial's coefficients. The verifier checks:

1. **Round consistency**: $g_j(0) + g_j(1) = \text{claim}_j$
2. **Chaining**: $\text{claim}_{j+1} = g_j(r_j)$
3. **Final binding**: the last round's evaluation matches the relation's output formula over the claimed openings

These checks are affine or low-degree polynomial relations over field elements. Substituting $m = \hat m - p$ turns each one into a relation over the pads with public coefficients, which R1CS can express.

## Protocol overview

### Phase 1: Pad layout and pad commitment

Both roles derive the **pad layout** from the proof's public shape alone: they replay the masked stage verifiers over a shape-correct proof on a transcript that skips every proof of work (`pad_layout`), counting one pad per private value plus one auxiliary slot per intermediate product of the claim formulas. The prover samples the pads, commits them row by row with Pedersen, and absorbs the row commitments right after the trace commitments, before the first stage challenge.

### Phase 2: Masked stages and the residual relation

During each of the 7 sumcheck stages the prover records through a masking recorder: each round is sent at the fixed width of its scheduled degree (so its length never depends on the witness), and each value goes onto the proof's masked tape and into the transcript as $\hat m$. Challenges are drawn exactly as in the clear protocol.

The verifier reads the tape in the same order and records each masked value with its pad. It then builds the residual relation: per stage, sumcheck rows chain the masked rounds from the batched input claim to a running claim, and one row equates that claim with the batched output formula, lowered from its factored form (`jolt-claims`' `output_factored`) with one auxiliary slot per intermediate product. BlindFold lowers the relation to relaxed R1CS over a witness laid out in rows:

- the pad rows, whose commitments are the pad commitment from Phase 1;
- two rows for the final evaluation $y$ and its blind, bound to Dory's hiding evaluation commitment;
- one row of intermediates per multi-pair product row;
- zero padding to a power of two.

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

Row commitments fold homomorphically. The pad commitment from Phase 1 serves as the real instance's pad-row commitments, so those rows are never recommitted.

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

The pad-row commitments are exactly the Pedersen commitments from Phase 1 — no additional commitments needed.

## Integration with Jolt

The pads are committed after the trace commitments; stages 1–7 run masked; stage 8 produces Dory's hiding batched opening. After stage 8 the prover replays its proof through the verifier's own stages to build the residual relation (the replay must land on the forward transcript's exact state), then proves it with BlindFold. A masked proof carries the masked tape, the pad commitment and BlindFold's proof; its stage sumcheck proofs are empty.

### Dory ZK evaluation commitments

The Dory polynomial commitment scheme is extended with ZK evaluation commitments. Instead of revealing the polynomial evaluation $v = \tilde{f}(r)$ in the clear, the prover sends a Pedersen commitment $y_{\text{com}} = v \cdot G_0 + \rho \cdot H$ and proves consistency within the Dory opening proof.

BlindFold's extra constraints verify that $y_{\text{com}}$ matches the folded evaluation and blinding values, binding the PCS to the BlindFold witness.

## Verification

The verifier:

1. **Derives the pad layout** from the proof's shape and absorbs the pad commitment
2. **Replays stages 1–7** over the masked tape, deriving the same challenges, and records each masked value with its pad
3. **Builds the residual relation** and lowers it to relaxed R1CS
4. **Reconstructs the real instance** from the pad commitment and the proof's auxiliary-row commitments, with $u=1$, $E=\mathbf{0}$
5. **Absorbs the random instance** from the proof and **folds** using the cross-term commitments and derived challenge $r$
6. **Verifies Spartan outer and inner sumchecks** and the **Hyrax openings** of $E(r_x)$ and $W(r_y)$
7. **Checks the evaluation commitment**: the final evaluation rows match Dory's hiding evaluation commitment
8. **Checks that every pad and every tape entry was used**

The verifier never sees a round coefficient, an intermediate claim or an evaluation value from any sumcheck stage.

## Security

- **Hiding**: each private message is masked by a fresh uniform pad, and the pads are committed with Pedersen, which is hiding; the folded BlindFold witness is uniform. The prover's sumcheck messages reveal no information about the witness polynomials.
- **Binding**: Pedersen commitments are computationally binding. A malicious prover cannot open a commitment to different coefficients.
- **Folding soundness**: The folded instance satisfies relaxed R1CS if and only if both constituent instances do (with overwhelming probability over the folding challenge).
- **Extraction**: An extractor can recover the real witness from two accepting transcripts with different folding challenges (via rewinding).

## References

- [Hyrax](https://eprint.iacr.org/2017/1132.pdf) — Matrix-commitment-based polynomial evaluation proofs (Wahby et al., 2018)
- [Nova](https://eprint.iacr.org/2021/370) — Recursive SNARKs via folding (Kothapalli, Setty, Tzialla, 2022)
- [Spartan](https://eprint.iacr.org/2019/550) — Sumcheck-based R1CS proving (Setty, 2020)
- [Vega](https://eprint.iacr.org/2025/2094) — Low-latency zero-knowledge proofs over existing credentials (Kaviani, Setty, 2025)
- [BlindFold interactive explainer](https://blindfoldzkp.github.io/)
