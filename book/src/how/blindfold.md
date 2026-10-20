# Zero Knowledge: BlindFold

BlindFold is available with Jolt's [Dory](./dory.md) backend by enabling the `zk` Cargo feature. The [Akita](./akita.md) backend currently supports only proofs without zero knowledge; combining `akita` and `zk` is rejected at compile time. This chapter describes the Dory ZK protocol.

Jolt achieves zero knowledge with **one-time pads committed in advance** and **[BlindFold](https://eprint.iacr.org/2025/2094)**, a folding-based argument that proves the verifier's checks without revealing what they check. The prover masks every private field message of the sumcheck stages with a pad, the verifier states every check it can no longer evaluate as a row over the pads, and a single Nova fold + Spartan proof shows the pads satisfy those rows.

The core idea: before the first challenge, the prover samples a uniform pad $p_i$ for every private field message $m_i$ the proof will send and commits to the pads with Pedersen row commitments. It then sends $\hat m_i = m_i + p_i$ in place of each $m_i$. Every message is uniform to the verifier, and every message is fixed before the challenge that follows it, because its pad already is.

## Background

In standard (non-ZK) Jolt, each sumcheck round reveals the round polynomial's coefficients, and each stage reveals its opening claims. The verifier checks:

1. **Round consistency**: $g_j(0) + g_j(1) = \text{claim}_j$
2. **Chaining**: $\text{claim}_{j+1} = g_j(r_j)$
3. **Final binding**: the last round's evaluation matches the stage's output formula over its opening claims

Once the challenges are fixed, each check is affine in the messages, except for products of opening claims in the output formulas. Substituting $m_i = \hat m_i - p_i$ turns every check into an affine or product relation over the pads.

## Protocol overview

### Phase 0: Pad commitment

At the end of stage 0, both prover and verifier derive the **pad layout**: the number of private messages stages 1–7 send, plus one slot per intermediate product of the residual relation (Phase 2). They derive it by running the masked stage verifiers over the proof's seeded prefix padded with zero bytes, so the layout has one owner: the verifier's own receives. The prover samples the pads and sends one Pedersen commitment per row of $L$ pads, each row with its own independent blind, where $L$ is the vector-commitment capacity.

### Phase 1: Masked sumcheck stages

During each of the 7 sumcheck stages the prover runs exactly the clear protocol, with one change: every coefficient and opening claim it would send goes out as $\hat m = m + p$ under the next pad. The verifier draws the same challenges from the masked messages and records each masked value with its pad slot, but checks nothing hidden. Uni-skip first rounds send all their coefficients and their output claim masked. Input claims, which the verifier derives from earlier claims, are not sent.

Stage 6's booleanity `bytecode_ra` openings alias their read-RAF source only when two challenge-derived points coincide; since the pad layout is fixed before any challenge, a masked proof always sends those cells and states the alias as an equality row.

### Phase 2: The residual relation

After stage 8, the verifier lowers its checks into the **residual relation**: affine rows $\sum_k a_k x_k + b = 0$ and product rows $x \cdot y = z$, over the pads and the residual witness (the stage-8 evaluation):

- **Compressed rounds** add no rows: the omitted linear coefficient is solved from the running claim, so the running claim after each round stays an affine form in the pads.
- **Uni-skip rounds** add their power-sum row $\sum_k \text{PowerSum}[k] \cdot c_k = \text{claim}$.
- **Each stage** adds one row equating its final running claim with the batched output formula.
- **Products** come from the stages' output formulas in factored form ($\tau \cdot A_z \cdot B_z$, $\prod_i \text{ra}_i \cdot (\ldots)$): each product takes one masked auxiliary slot, sent in one block after stage 8, and one product row.
- **Stage 8** adds one row: the evaluation inside Dory's hiding evaluation commitment equals the batched opening claims.

### Phase 3: Nova folding

BlindFold lowers the residual relation to relaxed R1CS over a witness $Z = [u, W]$ laid out in rows of $L$: the pad rows first, then two dedicated rows holding the evaluation $y$ and its blind $\rho$. The pad-row commitments of the real instance are exactly the Phase 0 commitments.

1. **Sample random instance**: Generate a random satisfying pair $(Z_2, E_2)$ with fresh uniform pad rows, dedicated rows and blinds
2. **Compute cross-term**: $T = (AZ_1) \circ (BZ_2) + (AZ_2) \circ (BZ_1) - u_1 \cdot (CZ_2) - u_2 \cdot (CZ_1)$
3. **Commit cross-term rows**: Commit each row of $T$ with Pedersen
4. **Derive folding challenge**: $r \leftarrow \text{Transcript}$
5. **Fold**: $Z' = Z_1 + r \cdot Z_2$, $E' = E_1 + r \cdot T + r^2 \cdot E_2$, $u' = u_1 + r \cdot u_2$

The folded instance satisfies relaxed R1CS:

$$(A \cdot Z') \circ (B \cdot Z') = u' \cdot (C \cdot Z') + E'$$

The random instance acts as a one-time pad: knowing $Z'$ alone reveals nothing about $Z_1$.

### Phase 4: Spartan outer sumcheck

Proves the folded relaxed R1CS is satisfied via the standard Spartan sumcheck:

$$\sum_{x \in \{0,1\}^{\log m}} \widetilde{\text{eq}}(\tau, x) \cdot \left[\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - u' \cdot \widetilde{Cz}(x) - \widetilde{E}(x)\right] = 0$$

### Phase 5: Spartan inner sumcheck

Each of $\widetilde{Az}(r_x)$, $\widetilde{Bz}(r_x)$, $\widetilde{Cz}(r_x)$ decomposes as a public contribution (from the $u$ column, which carries every constant, including the masked values) plus a witness contribution, which the inner sumcheck reduces to a single evaluation $W(r_y)$.

### Phase 6: Hyrax-style openings

The witness $W$ and error $E$ are committed row-wise. To prove $W(r_y)$, the prover sends the $\widetilde{\text{eq}}(r_{\text{row}}, \cdot)$-combined row and blind; the verifier checks it against the combined row commitments and evaluates it at $r_{\text{col}}$. The $E$ opening follows the same pattern. The folded $y$ and $\rho$ are opened at their dedicated coordinates and checked against the folded evaluation commitment.

## Integration with Jolt

| Stage | Protocol | ZK role |
|-------|----------|---------|
| 0 | Commitments | Pad layout, pad commitment |
| 1–7 | Sumcheck stages | Masked rounds and claims |
| **8** | **Batched opening proof** | Hiding evaluation commitment |
| tail | Residual relation | Masked auxiliary block, BlindFold |

The prover derives the residual relation by replaying its own argument string through the verifier, so the relation it proves is the one the verifier builds; it computes the auxiliary values with the same lowering over the values its pads decode.

### Dory ZK evaluation commitments

The Dory polynomial commitment scheme is extended with ZK evaluation commitments. Instead of revealing the polynomial evaluation $v = \tilde{f}(r)$ in the clear, the prover sends a Pedersen commitment $y_{\text{com}} = v \cdot G_0 + \rho \cdot H$ and proves consistency within the Dory opening proof. The residual relation's stage-8 row and BlindFold's dedicated-row check bind $y$ to the masked claims.

## Verification

The verifier:

1. **Derives the pad layout** and reads the pad commitment before stage 1
2. **Runs stages 1–8**, recording masked values and drawing challenges from them
3. **Builds the residual relation** from the recorded values, challenges and claim formulas, and reads the masked auxiliary block
4. **Lowers the relation to relaxed R1CS** and reconstructs the real instance from the pad commitment with $u=1$, $E=\mathbf{0}$
5. **Folds** with the random instance, the cross-term commitments and the derived challenge $r$
6. **Verifies the Spartan outer and inner sumchecks** and the Hyrax openings
7. **Checks the evaluation commitment** against the folded $y$ and $\rho$

The verifier never sees polynomial coefficients, intermediate claims, or evaluation values from any sumcheck stage.

## Security

- **Hiding**: Each pad is uniform and used once, so every masked message is uniform. The pad rows are hidden by independent Pedersen blinds, and the folded witness is uniform.
- **Binding**: Pedersen commitments are computationally binding under the discrete logarithm assumption, so the pads, and with them every decoded message, are fixed before the challenge that follows each message.
- **Residual relation**: A witness satisfying the residual relation decodes to a clear transcript the ordinary verifier accepts, with the same challenges.
- **Folding soundness**: The folded instance satisfies relaxed R1CS if and only if both constituent instances do (with overwhelming probability over the folding challenge).

## References

- [Hyrax](https://eprint.iacr.org/2017/1132.pdf) — Matrix-commitment-based polynomial evaluation proofs (Wahby et al., 2018)
- [Nova](https://eprint.iacr.org/2021/370) — Recursive SNARKs via folding (Kothapalli, Setty, Tzialla, 2022)
- [Spartan](https://eprint.iacr.org/2019/550) — Sumcheck-based R1CS proving (Setty, 2020)
- [Vega](https://eprint.iacr.org/2025/2094) — Low-latency zero-knowledge proofs over existing credentials (Kaviani, Setty, 2025)
- [BlindFold interactive explainer](https://blindfoldzkp.github.io/)
