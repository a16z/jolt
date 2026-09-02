# Lane L (G3) — Dory offloading study: can Dory be verified inside the Fr wrapper at ≪100k constraints and single-digit KB?

Date 2026-09-02 · tree f0d633e80 (origin/main 756bddce3 + lane commits) · read-only. Sources: `dory-pcs 0.4.2` (Cargo.lock:2275; `D` = `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/dory-pcs-0.4.2/src`), `crates/jolt-dory/src/{scheme,routines,transcript,types}.rs` (`V` = `crates/jolt-verifier/src`, `A` = a16z arkworks fork checkout), Dory paper eprint 2020/1274 (§3.1–3.4, §4.1–4.4, §5, §6.1–6.2), `.journals/discovery/{verifier-structure,prior-art,prover-stack-prior-art,proof-size}.md`, `.journals/plan-relation.md`, `.journals/lanes/{blake2b-gadget,oracle-dory-offload}.md`. Parameters: L=18 → n=L+4=22, σ=ν=11, N=41; L=20 → n=24, σ=12. "exact" = counted from code; "est." marked. Additive GT notation: `X+Y` = Fq12 mult, `s·X` = Fq12 pow (`D/backends/arkworks/ark_group.rs:197-199`, generic `pow`, no cyclotomic).

## 0. Verdict

**No design keeps Dory, stays ≤~100k in-circuit AND reaches single-digit KB.** Two independent floors, each sufficient:
- **Transcript floor (exact):** the production `LegacyBlake2bTranscript` derives β_j, α_j, γ, d by hashing the 384-B serialisation of every Dory GT element and the 41 commitments (`crates/jolt-dory/src/transcript.rs:56-66`, `D/evaluation_proof.rs:360-363, 429-443`). Whoever derives the challenges holds the bytes: natively ⇒ they travel in the proof (≥ 13.95 KB at 128 B/GT); in-circuit ⇒ 502/546 compressions × 52,416 = **26.3M / 28.6M constraints** (L=18/20; +205 × 52,416 = 10.7M if the commitments are hidden too). Any Fq-side sub-proof that hides them must prove the same Blake2b bytes.
- **Multi-exp floor:** the accept decision is one GT multi-exp over 6σ+2 proof elements + N commitments + 3σ+2 setup constants (§1.4) with distinct challenge-monomial scalars. 6σ−2+N of the proof/commitment bases have no efficiently computable single-pair preimage (§3), so they are either held natively (bytes) or their exponentiations are proven: ≥ 0.7–3.2M non-native Fq mults ⇒ **≥ 0.7–8.6 B** Fr constraints (§2, §4.ii–iii), or an Fq-native sub-proof whose Fr verifier is ≥ 3M constraints and whose prover is 3–50 s (§4.iv).
- The only Dory-kept design under 1 s is **deferred-native Dory (§4.i)**: ~0–500 constraints in-circuit, byte-identical transcript, **≈21.4–21.8 KB @2^18 / 22.9 KB @2^20** total with the 128-B GT codec (49.7 / 52.3 KB at 384 B). Single-digit KB requires leaving GT for the commitments (§5).

## 1. The Jolt-Dory verifier as equations (exact)

Setup (`D/setup.rs:47-80, 121-172`): Γ1 ∈ G1^{2^11}, Γ2 ∈ G2^{2^11} random (transparent URS, `scheme.rs:130-155`), H1, H2, HT=e(H1,H2); χ[k] = ⟨Γ1[..2^k], Γ2[..2^k]⟩ (k=0..σmax), Δ1L[k]=Δ2L[k]=χ[k−1], Δ1R[k]=⟨Γ1[2^{k−1}..2^k], Γ2[..2^{k−1}]⟩, Δ2R[k]=⟨Γ1[..2^{k−1}], Γ2[2^{k−1}..2^k]⟩. Verifier key = 3σ+2 distinct GT constants (χ[0..σ], Δ1R[1..σ], Δ2R[1..σ], HT) + Γ1_0, Γ2_0, H1, H2.

Commitment (`scheme.rs:163-180, 455-553`): poly with n vars as a 2^ν×2^σ matrix M; tier-1 rows T_i = Σ_j M_ij Γ1_j ∈ G1; tier-2 C_M = Σ_i e(T_i, Γ2_i) ∈ GT (AFGHO, paper §2.4). Jolt sends 41 such GT (`V/proof.rs:103-110`).

### 1.1 Stage-8 input (exact; `V/stages/stage8/verify.rs:120-244`, `scheme.rs:313-323`, `D/evaluation_proof.rs:129-156, 384-423`)
- ρ = γ_rlc, D1_init = Σ_{i<41} ρ^i·C_i — **41 GT exps + 40 GT mults** (`combine`, per-element generic pow). Joint claim y = Σ ρ^i·claim_i·scale_i (Fr).
- Prover VMV message: C_init = e(⟨T_pad, v⟩, Γ2_0), D2_init = e(⟨Γ1[..2^σ], v⟩, Γ2_0), E1_init = ⟨T, L⟩ ∈ G1, where v = Lᵀ M ∈ Fr^{2^σ}, L = ⊗(1−r_i, r_i) over the ν row coords, R over the σ column coords (`D/evaluation_proof.rs:128-156`; point reversed at `scheme.rs:290`).
- Verifier: E2_init = y·Γ2_0 (**1 G2 mul**, `:384-387`); s1_coords = point[..σ] (=R), s2_coords = point[σ..] zero-padded (=L) (`:406-411`); s1_acc = s2_acc = 1.
- Statement being reduced (paper §4, `L_{n,Γ1,Γ2}(s1,s2)`): ∃ v1 ∈ G1^{2^σ}, v2 ∈ G2^{2^σ}: D1 = ⟨v1,Γ2⟩, D2 = ⟨Γ1,v2⟩, C = ⟨v1,v2⟩, E1 = ⟨v1,s2⟩, E2 = ⟨s1,v2⟩ — honest v1 = T_pad, v2 = v⊗Γ2_0. Plus the VMV constraint D2_init = e(E1_init, Γ2_0) (paper §5 Σ-protocol on t; code batches it at the d² slot, `D/reduce_and_fold.rs:756-780`).

### 1.2 Reduce round j = 0..σ−1 (setup index k = σ−j; `D/reduce_and_fold.rs:675-737`)
Messages: D1L, D1R, D2L, D2R ∈ GT, E1β ∈ G1, E2β ∈ G2 → β_j; C+, C− ∈ GT, E1+, E1−, E2+, E2− → α_j (transcript order `D/evaluation_proof.rs:429-443`). Honest values (`D/reduce_and_fold.rs:214-268, 308-330`): D1L=⟨v1L,Γ2[..2^{k−1}]⟩, D1R=⟨v1R,Γ2[..2^{k−1}]⟩, D2L=⟨Γ1[..2^{k−1}],v2L⟩, D2R=⟨Γ1[..2^{k−1}],v2R⟩, E1β=⟨Γ1[..2^k],s2⟩, E2β=⟨s1,Γ2[..2^k]⟩; after v1 ← v1+βΓ1, v2 ← v2+β⁻¹Γ2 (`:279-283`): C+=⟨v1L,v2R⟩, C−=⟨v1R,v2L⟩, E1±=⟨v1L,s2R⟩/⟨v1R,s2L⟩, E2±=⟨s1L,v2R⟩/⟨s1R,v2L⟩.
```
C'  = C + χ[k] + β·D2 + β⁻¹·D1 + α·C+ + α⁻¹·C−                       (4 GT exp, 5 GT mult)   :695-700
D1' = α·D1L + D1R + αβ·Δ1L[k] + β·Δ1R[k]                              (3 GT exp, 3 GT mult)   :704-708
D2' = α⁻¹·D2L + D2R + α⁻¹β⁻¹·Δ2L[k] + β⁻¹·Δ2R[k]                      (3 GT exp, 3 GT mult)   :711-715
E1' = E1 + β·E1β + α·E1+ + α⁻¹·E1−                                    (3 G1 mul, 3 add)       :717-720
E2' = E2 + β⁻¹·E2β + α·E2+ + α⁻¹·E2−                                  (3 G2 mul, 3 add)       :723-726
s1_acc *= α·(1−y_t)+y_t ;  s2_acc *= α⁻¹·(1−x_t)+x_t   (y_t=s1_coords[k−1], x_t=s2_coords[k−1])   :729-733
```
Fr per round: 2 inversions (`:691-692`), ≈8 mults. Matches paper Fig. Dory-Reduce (p.17) and §4.2 (p.24) exactly, with Δ2L=Δ1L=χ[k−1].

### 1.3 Final check (transparent arm, `D/reduce_and_fold.rs:953-991`; challenges γ then d after absorbing final E1_fin, E2_fin)
```
e(E1_fin + d·Γ1_0 , E2_fin + d⁻¹·Γ2_0)                                   Pair 1  :969-970
· e(H1 , −γ·(E2_acc + d⁻¹·s1_acc·Γ2_0))                                  Pair 2  :973-974
· e(−γ⁻¹·(E1_acc + d·s2_acc·Γ1_0) , H2)                                  Pair 3  :977-979
· e(d²·E1_init , Γ2_0)                                                    Pair 4  :982-983   (deferred VMV)
= C_σ + (s1_acc·s2_acc)·HT + χ[0] + d·D2_σ + d⁻¹·D1_σ + d²·D2_init          RHS     :961-966   (4 GT exp, 5 mult)
```
One `multi_pair` = 4 Miller loops + 1 final exponentiation (`D/backends/arkworks/ark_pairing.rs:144-180`); one Fq12 equality. This is paper Scalar-Product (p.15) applied after Fold-Scalars (§4.1: C' = C + ⟨s1,s2⟩HT + γe(H1,E2) + γ⁻¹e(E1,H2); D1' = D1 + e(H1, γs1Γ2_0); D2' = D2 + e(γ⁻¹s2Γ1_0, H2)), with the e(H1,·)/e(·,H2) terms moved to the LHS and the VMV Σ-protocol replaced by the d²-slot pairing. Zero Fq12 inversions anywhere.

### 1.4 Closed form (paper §3.3 "Deferring V computation", p.20–21, specialised to the code)
Let u_j = β_{j+1}⁻¹ (j<σ−1), u_{σ−1} = d⁻¹; v_j = β_{j+1} (j<σ−1), v_{σ−1} = d. Then
```
RHS = C_init + Σ_i (β_0⁻¹ρ^i)·C_i + (β_0 + d²)·D2_init
    + Σ_j [ α_j·C+_j + α_j⁻¹·C−_j + u_j·(α_j·D1L_j + D1R_j) + v_j·(α_j⁻¹·D2L_j + D2R_j) ]
    + Σ_j [ (u_jα_jβ_j + v_jα_j⁻¹β_j⁻¹)·χ[σ−j−1] + u_jβ_j·Δ1R[σ−j] + v_jβ_j⁻¹·Δ2R[σ−j] ] + Σ_{k=0}^{σ} χ[k] + (s1s2)·HT
LHS = e(A,B)·e(H1,B')·e(A',H2)·e(A'',Γ2_0)  with
  E1_acc = E1_init + Σ_j (β_j·E1β_j + α_j·E1+_j + α_j⁻¹·E1−_j)   (G1 multi-exp, 3σ terms)
  E2_acc = y·Γ2_0  + Σ_j (β_j⁻¹·E2β_j + α_j·E2+_j + α_j⁻¹·E2−_j)  (G2 multi-exp, 3σ+1 terms)
  A = E1_fin + d·Γ1_0, B = E2_fin + d⁻¹·Γ2_0, B' = −γ·E2_acc − γd⁻¹s1·Γ2_0, A' = −γ⁻¹·E1_acc − γ⁻¹ds2·Γ1_0, A'' = d²·E1_init
```
Every X_k with its scalar s_k:

| X_k (count) | s_k | class |
|---|---|---|
| C_init (1) | 1 | proof GT |
| C_i (N=41) | β_0⁻¹·ρ^i | commitment GT |
| D2_init (1) | β_0 + d² | proof GT |
| C+_j, C−_j (2σ) | α_j, α_j⁻¹ | proof GT |
| D1L_j, D1R_j (2σ) | u_jα_j, u_j | proof GT |
| D2L_j, D2R_j (2σ) | v_jα_j⁻¹, v_j | proof GT |
| χ[k] (σ+1) | 1 + [k≤σ−1]·(u_jα_jβ_j + v_jα_j⁻¹β_j⁻¹), j=σ−k−1 | setup GT |
| Δ1R[k], Δ2R[k] (2σ) | u_jβ_j, v_jβ_j⁻¹ (k=σ−j) | setup GT |
| HT (1) | s1_acc·s2_acc = Π_j(α_j(1−y_j)+y_j)·Π_j(α_j⁻¹(1−x_j)+x_j) | setup GT |

**Counts (exact):** GT bases the verifier must hold = (6σ+2) + N + (3σ+2) = 9σ+N+4 = **144 @σ=11 / 153 @σ=12**; distinct non-unit scalars = 9σ+N+3 = **143 / 152**, all Laurent monomials/binomials in {α_j^{±1}, β_j^{±1}, ρ, d^{±1}, d², s1s2}; Fr inversions 2σ+2; G1 bases 3σ+4 (E1β/E1±/E1_init/E1_fin/Γ1_0... 37), G2 bases 3σ+4 (37); pairings 4+1 FE. Round-by-round as coded: GT exps 10σ+4+N = 155/165, GT mults 11σ+4+N = 166/177 (`verifier-structure.md §3.3`); deferred form: 9σ+N+3 exps. Deserialisation adds 6σ+2 GT `pow(r)` subgroup checks + N commitment checks (`types.rs:96` → `deserialize_compressed` with validation; `A/ec/src/pairing.rs:191-199`), each ≈ one 254-bit Fq12 exp.

Transcript per round (exact, `plan-relation.md:23,41`): 4 GT(5 compressions each)+G1(2)+G2(2)+squeeze β; 2 GT+2 G1+2 G2+squeeze α = 44 compressions; VMV 12; final 6 → **C_D = 44σ+18 = 502 / 546**; the 41 commitments 41×5 = 205 (absorbed natively before stage 1, `plan-relation.md:96`).

## 2. Linear vs nonlinear classification

(a) **Fr-native (cheap):** all s_k above, s1_acc/s2_acc (2σ mults), 2σ+2 inversions (1 constraint + hint each), ρ powers, joint claim — **≈ 300–600 constraints total (est.)**; G1/G2/pairing-input scalars included. Nothing else about Dory is Fr-native.

(b) **Group-linear with public scalars (a multi-exp):** RHS = Σ s_k X_k in GT (144 bases), E1_acc/E2_acc in G1/G2 (37 each), the four pairing inputs. Linear *in the group*, but the group law is Fq12 multiplication / EC addition over Fq, Fq2: the operands X_k must be held by whoever evaluates it, and each scalar application is an exponentiation.

(c) **Intrinsically nonlinear over Fq:** every GT exp (254-bit) — 254 Fq12 sqr + ~127 Fq12 mult as coded — on witness-dependent bases; every Fq12 mult; G1/G2 scalar muls (EC law over Fq/Fq2); 4 Miller loops + final exp; 6σ+2+N subgroup checks. No exponent is secret in Jolt-Dory (all scalars are public challenges); (c) is nonlinear because the *bases* are witness-dependent, not because the exponents are.

**Why RLC/batching cannot remove the Fq12 multiplications.** An Fr-R1CS with prover advice verifies relations that are R1CS-expressible over Fr; a GT exp Y = s·X unrolled is a chain X_{i+1} = X_i² (or X_i·X) of ~380 steps, each a *bilinear* map on 12 Fq coordinates (54 Fq mults, Karatsuba 2-3-2 tower), each Fq mult itself a limb product + quotient/remainder + range checks. Random batching Σ_i λ^i (Z_i − X_i·Y_i) = 0 collapses the *equality* checks (which are linear and cost nothing) but still requires every product X_i·Y_i to be formed — the residual is not computable without it, and if the prover supplies residuals unconstrained they are free variables. A Freivalds-type shortcut needs the verifier to natively evaluate a bilinear form over Fq — exactly the non-native operation being avoided. Batching helps only where operands are shared (Straus/Pippenger across bases, −60% mults) or where a relation is *linear* in committed Fr values (the sumcheck chain). Bilinearity of the pairing does move scalars between groups — e(sP, Q) = s·e(P, Q) — but only for bases with a known single-pair preimage (§3).

**Unit costs in Fr-R1CS, no lookups (est.; Fq mult = 1,000 plain / 2,716 measured `FqVar`, prover-stack-prior-art §3.1):**

| op | Fq mults | constraints @1,000 | @2,716 | × 2^18 budget |
|---|---|---|---|---|
| Fq12 mult (Karatsuba tower) | 54 | 54k | 147k | 0.21–0.56 |
| Fq12 squaring / cyclotomic squaring | 36 / 18 | 36k / 18k | 98k / 49k | 0.14–0.37 / 0.07–0.19 |
| **GT exp, 254-bit, as coded** (254 sqr + 127 mult) | 16,002 | **16.0M** | **43.5M** | **61–166** |
| GT exp, best in-circuit (cyclotomic sqr + 4-bit window: 254·18 + 64·54 + 14·54) | 8,784 | 8.8M | 23.9M | 34–91 |
| G1 / G2 scalar mul (254-bit Jacobian) | ≈3k / ≈10.7k | 3M / 10.7M | 8M / 29M | 11–110 |
| 4-way multi-pairing (4 ML ≈ 7.7k each + FE ≈ 5.2k) | ≈36k | 36M | 98M | 137–374 |

A single GT exponentiation is 34–166× the whole wrapper budget; a single GT multiplication is 21–56% of it.

## 3. Minimal-knowledge argument

**Claim.** Without a further succinct argument, the party evaluating the Dory accept predicate must hold, as GT elements, every C_i (or Σρ^iC_i together with the C_i needed to check it) and every one of the 6σ+2 proof GT elements except four.

*Proof.* (1) Each X_k enters the decision only through s_k·X_k in §1.4 with s_k a non-constant Laurent monomial in fresh challenges (β_0⁻¹ρ^i, α_j^{±1}, u_j, v_j, …); the map X_k ↦ RHS is injective for any fixed challenge vector, and the prover fixes X_k before its scalar's challenge is drawn, so the predicate is not evaluable from a proper subset of the X_k (Lemma 6 / Thm 2–3 extraction, paper p.10, 16–20, interpolate each coefficient separately). (2) The only cheaper representation of s·X than X itself is a pairing preimage: X = e(P,Q) ⇒ s·X = e(sP, Q), a G1 scalar mul + pairing with P (32 B). Since e(·, Γ2_0): G1 → GT is an isomorphism of prime-order groups, every X *has* a preimage P; computing it from X or from a multi-pair representation Σ_i e(a_i, b_i) with independent random b_i is the fixed-argument pairing-inversion problem (Galbraith–Hess–Vercauteren 2008), for which no efficient algorithm is known — and SXDH-based Dory soundness presupposes Lemma 3 (paper p.8), which pairing inversion would break. Honest D1L/D1R = Σ_i e(T_i (+βΓ1_i), Γ2_i), C± = Σ e(v1, v2) with v2 ∋ β⁻¹Γ2_i, D2L/D2R (j≥1) = Σ e(Γ1_i, v_iΓ2_0 + β⁻¹Γ2_i), and every commitment C_i = Σ_r e(T_{i,r}, Γ2_r) are such multi-pair sums over the independent generators — non-succinct (2^{k} terms) and uninvertible. ∎
*RLC of commitments:* D1_init = Σρ^iC_i sent alone is unsound (prover picks it after ρ) and its check IS a 41-term multi-exp needing the C_i; additionally state_in = Blake2b(preamble ‖ 41 × 384 B) must be recomputed by the native verifier (`plan-relation.md:96`) or the FS commit-then-challenge order breaks — so the 41 GT (or their bytes) are needed regardless (**5,248 B @128 B; 15,744 B @384 B — exact**).

**The four exceptions (exact, single-pair by construction, all against the fixed Γ2_0):** C_init = e(P_C, Γ2_0) with P_C = ⟨T_pad, v⟩ (`D/evaluation_proof.rs:143-144`); D2_init = e(P_D, Γ2_0), P_D = ⟨Γ1[..2^σ], v⟩ (`:147-150`); round-0 D2L, D2R = e(⟨Γ1[..2^{σ−1}], v_L/v_R⟩, Γ2_0) — the prover already computes them this way (`D/reduce_and_fold.rs:246-252`, `v2_scalars` path). Send P_C, P_D, P_L, P_R ∈ G1 (4 × 32 B) instead: the native verifier recomputes the 4 GT values with 4 pairings (~1.5 ms) and absorbs the *same 384-B serialisations* → **transcript bytes unchanged**, 4 subgroup checks saved. Savings: 4 × (128−32) = **384 B** compressed, 4 × 352 = **1,408 B** uncompressed. After round 0, v2 ← v2 + β⁻¹Γ2 mixes independent generators; no further element is single-pair. E1β_j, E2β_j are functions of public data (`:263-266`) and *could* be recomputed by the verifier for σ×96 = 1,056 B saved, at 2^{σ+1} G1 + 2^{σ+1} G2 muls (≈0.5 s est.) — not worth it.

**Byte floor of natively-held Dory data @2^18 (exact):** (6σ−2+N)·128 + 4·32 + (3σ+2)·32 + (3σ+1)·64 + ~17 framing = 105·128 + 128 + 1,120 + 2,176 + 17 = **16,881 B** (17,265 without the preimage trick; 43,761 / 45,169 at 384 B). @2^20: 111·128 + 128 + 1,216 + 2,368 + 17 = **17,937 B**. The wrapper itself is 3,616 B + ≈0.9 KB public IO (`plan-relation.md:106`).

## 4. Candidate designs with numbers

Common: wrapper Spartan+HyperKZG proof = 3,616 B @m=n=2^18 (`plan-relation.md:106`) + ≈896 B public IO (stage-8 boundary: 32-B transcript state, n-coordinate point, joint claim); Dory-side bytes from §3; "single-digit KB" read as < 10,000 B. Budget 262,144 constraints; ≤ 92k of it is what a Poseidon-transcript wrapper would leave for Dory (`plan-relation.md`, planner) — under Blake2b the whole wrapper is already 106M (BLOCKER #1).

### (i) Scalars in-circuit (or nowhere), GT multi-exp + pairings native, all elements sent — "deferred-native Dory"
- **Circuit:** 0–600 constraints. Minimal form: the circuit exports the 32-B transcript state after the RLC-γ squeeze, the point and the joint claim; the native verifier re-derives ρ (already in-circuit), β_j, α_j, γ, d over the public Dory bytes (502/546 compressions natively, µs), computes the 143 scalars, one 144-base GT multi-exp (or the round-by-round 155/165 exps as coded), two 37-base G1/G2 multi-exps, one 4-way multi-pairing. Optionally the circuit computes the scalars (≈500 constraints) and exports them (143 Fr = 4.6 KB IO — worse bytes; don't).
- **Native verifier:** exactly today's stage 8 + deserialisation subgroup checks; bounded by the measured whole-Jolt verify **67.8 ms @2^18 / 79.7 ms @2^20** (`proof-size.md:19-20`) + wrapper verify (HyperKZG 1.56 ms measured, `hyperkzg-perf.md:23`).
- **Bytes (exact, 128-B GT codec / 384-B):** @2^18: Dory 12,017 (11,633 with §3 preimage trick) + commitments 5,248 + wrapper 3,616 + IO 896 = **21,777 (21,393) B** / 29,425 + 15,744 + 4,512 = **49,681 B**. @2^20: 13,073 (12,689) + 5,248 + 4,576 = **22,897 (22,513) B** / 52,337 B. Miss vs 10,000 B: 2.1–2.3×.
- **Prover overhead:** none (existing Dory proof; codec compression of 109 GT is µs). **Soundness:** identical to production Jolt — byte-identical transcript, the native verifier is the production verifier with stages 1–7 replaced by the wrapper proof; it must recompute state_in from preamble + 41 commitments (`plan-relation.md:96`) and continue the Blake2b chain from the exported state; exporting challenges without the state is unsound (prover picks incompatible challenges/elements).
- Verdict: **the only Dory-kept design under 1 s and ≪100k; fails the byte target.** This is the "Fr-only + native Dory 23–27 KB" the user rejected, now with exact numbers (21.4–22.9 KB).

### (ii) Prover advice for GT exps/pairings, verified in-circuit via non-native Fq
Work as coded @2^20: 165 GT exps × 16,002 + 177 × 54 + 40 G1 × 3k + 40 G2 × 10.7k + multi-pairing 36k ≈ **3.23M Fq mults → 3.2B–8.8B constraints (est.)**; best in-circuit (cyclotomic + windows) 2.0B–5.5B; Straus over the 153 GT bases (254 shared cyclotomic squarings + 153×64 window mults + tables) 649k + 10k + 548k + 36k ≈ **1.24M Fq mults → 1.24B–3.4B**. Ratio to 2^18: **4,700×–33,600×**. Bytes could drop to the 4.5-KB wrapper only if the GT elements are witness — which forces their Blake2b absorption in-circuit (+26.3M). Prover: HyperKZG scaling from 0.47 s @2^18 gives hours and >100 GB of matrices. Sound if fully constrained; infeasible by 3.5–4.5 orders of magnitude.

### (iii) Advice + batching (one random linear combination of the GT relations)
Relations: Z_i = X_i·Y_i (Fq12) for each of the ≈62.9k exp-chain steps (as coded) or ≈12k (Straus), plus the final Σ s_kX_k = LHS equality. Batched residual Σ λ^i (Z_i − X_i·Y_i) = 0 still requires forming every X_i·Y_i (bilinear, 54 non-native Fq mults each); the saved equality checks are linear (free already). The final GT equality and the 4-pairing check are already one Fq12 comparison natively. **Non-native count floor = (#Fq12 mults in the exp chains) × 54 ≥ 0.65M Fq mults ≥ 0.65B constraints (est.)** — batching removes < 1%. Setup-constant exps (Δ, χ, HT: 3σ+2 fixed bases) could use precomputed tables, but authenticating a 254-bit-indexed table lookup without a lookup argument costs ≈ the exp. No.

### (iv) Fq-native sub-proof (Spartan over Fq, Grumpkin-Pedersen/Hyrax/IPA) of the GT multi-exp + pairing, verified by the Fr circuit
- **Fq circuit:** 1.24M (Straus) – 3.23M (as coded) Fq mults ≈ 2^20.2–2^21.6 constraints (+5M with subgroup checks). **Fq prover @63k–440k constraints/s: 2.8–51 s** (≫ 1 s on its own).
- **Fr-circuit verification:** (a) Spartan-Fq sumchecks: 2·log S ≈ 42 rounds × ≈4 non-native Fq mults ≈ 170 Fq mults → **170k–460k**; its transcript: Blake2b over ≈130 Fq + 44 squeezes ≈ 200 compressions ≈ 10.5M, or a *new* Poseidon-over-Fr hash of the Fq limbs (≈20k; Fr ⊂ [0,q) so digests are valid Fq challenges) — a new component, not the Jolt transcript. (b) PCS: Hyrax over Grumpkin (Grumpkin coordinates are Fr-native, scalars are Fq): √S = 2^10.5–2^11 row commitments → in-circuit MSM: 2,048 × 254 scalar-bit booleans = 520k + Straus adds ≈131k × ~6 = **0.8M**, plus the combined-row inner product 2^11 non-native Fq mults = **2–5.5M**; IPA instead: 2·21 G1 (1.3 KB) but the folded-generator MSM is 2^21 Grumpkin muls — in-circuit impossible (5B), natively 0.1–0.4 s (est.) and then the Fq sumcheck may as well be native too, at which point the GT elements must be public for the Blake2b transcript → sent → ≥17 KB (circular). **Total in-circuit ≥ 3.5–7M Fr constraints (13–27× budget)**; the oracle's 150k–500k counts only (a).
- **Bytes:** only if the GT elements are hidden, i.e. the Fq circuit also proves Blake2b over their 44.7 KB (502+205 compressions ≈ 37M Fq bit-constraints → +85–590 s prover). Otherwise ≥ 21 KB as (i).
- Verdict: dominated on constraints, prover time and (under the production transcript) bytes. Its historical role (June "Dory-assist" branches, `prover-stack-prior-art.md §1`) was RISC-V cycles for in-guest recursion, with a native O(√N) Hyrax verifier and no byte target.

### (v) Dory layout changes
- Code facts: matrix 2^ν rows × 2^σ cols; `nu > sigma` rejected (`D/evaluation_proof.rs:340-345`); `num_rounds = sigma` (`:391`); rows/scalars zero-padded to 2^σ (`:158-161, 179-185`); the verifier folds the point scalar-by-scalar and never materialises a 2^ν or 2^σ vector (`:406-411`, `D/reduce_and_fold.rs:729-733`). Jolt fixes n = log_k_chunk + L, σ = ⌈n/2⌉ (`crates/jolt-prover/src/config.rs:162-183`, `scheme.rs:242-243`).
- **Rounds = max(ν, σ); GT = 6·max(ν,σ)+2 ≥ 6⌈n/2⌉+2.** The square layout is the minimum; a 2^k × 2^{n−k} layout with k ≠ n/2 *adds* elements (k=4, n=22: 110 GT). There is no O(2^k) verifier MSM in any layout to trade against — the VMV step already reduces the row dimension for the verifier at O(1) (one pairing at the d² slot); the column dimension costs 6 GT per halving, intrinsically (paper p.17: 6 is already the optimised count vs the naive 8). Jolt-level knob: log_k_chunk 4→8 (N 41→23, n→L+8, σ→13): GT 109→103, +6 G1 +6 G2 → net −192 B. Irrelevant.
- **Extreme n2 = 1** (one row of 2^n): commitment e(T, Γ2_0) with T = Σ_j M_j Γ1_j needs a 2^n-element G1 basis (2^22 × 64 B = 268 MB, transparent; Dory's URS is 2^11 G1 + 2^11 G2) and the opening is no longer Dory-Reduce: it is a G1 Pedersen vector commitment opened with a Bulletproofs IPA — 2n G1 + 2 Fr = **1,472 B @n=22** (1,600 @24); the verifier does an **O(2^n) G1 MSM natively (2^22 points: est. 0.15–0.4 s on 10 cores)** + 2n G1 muls + 2^n Fr products for the folded-generator coefficients (≈10 ms); the circuit needs nothing (deferred). Commitments become G1 (send T_i; 41 × 32 = 1,312 B). Total ≈ 3,616 + 896 + 1,312 + 1,472 = **7,296 B ✓, 0 in-circuit ✓, transparent ✓** — but it is not Dory in the codebase's sense (no AFGHO tier-2, no GT, no χ/Δ, different setup, O(N) verifier — the property Dory exists to provide) and the IPA prover adds ≈2^{n+1} G1 muls (est. +1–3 s @2^22).

### (vi) Alternative opening over the SAME Dory/AFGHO commitments with a succinct verifier
- **TIPP** (Bünz–Maller–Mishra–Tyagi–Vesely 2019/1177): commitment (⟨A,v⟩, ⟨w,B⟩, ⟨A,B⟩) ∈ GT³ with structured v, w; GIPA cross-terms **6 GT per round**, verifier folds them with **6 GT exps per round** + O(1) KZG key checks (2 G1/G2, ≈4 pairings). Same GT count as Dory-Reduce; only the χ/Δ constants disappear. **No gain in bytes or exps.**
- **MIPP_k** (public structured b — exactly Jolt's row fold E1 = Σ_i L_i T_i): commitment ⟨T, Γ2⟩ with Γ2_i structured (trusted τ, size 2^ν — still O(√N)); per round 2 GT + 2 G1 (commitment cross-terms + G1 cross inner products); proof 2ν GT + 2ν G1 + G1 + 2 G2 ≈ 22×160 + 160 = **3,680 B @ν=11**; verifier **2ν = 22 GT exps + 22 G1 muls + 3 pairings**. Column side (E1 = Σ_j v_j Γ1_j Pedersen, ⟨v,R⟩ = y): Γ1 random → Bulletproofs IPA 2σ G1 + 2 Fr = 768 B with a native 2^σ = 2,048-point MSM (≈5–10 ms); Γ1 structured → PST/KZG σ G1 = 352 B + σ+1 pairings. **Opening ≈ 4.0–4.4 KB, native ≈ 15–25 ms (est.), in-circuit 0 (deferred).** Total ≈ 3,616 + 896 + 5,248 + 4,432 = **14,192 B @2^18 (est.)** — the 41 GT commitments remain: with structured Γ2, C_M = e(Σ_i τ^i T_i, H2) but the prover cannot form Σ_i [τ^i]T_i without [τ^i]Γ1_j for all i, j = a 2^n G1 SRS, which is PST/HyperKZG. Requirements: new opening protocol (not in dory-pcs; MIPP_k security under q-type assumptions + IPA/DL), trusted structured G2 setup, commitment *algorithm* unchanged but *values* change (new Γ2), transcript changes. "Dory with KZG-committed generator vectors" only saves verifier-key bytes (χ/Δ), not GT exps or proof GT.
- Consequence: **any opening whose commitments are AFGHO/GT-valued sends ≥ 2 GT per halving of the folded dimension and needs ≥ 2 log GT exps by the verifier; the 41-commitment floor (5,248 B) + wrapper (4,512 B) = 9,760 B already leaves < 240 B for the opening.** Single-digit KB with Jolt's 41 committed polynomials ⇒ commitments must leave GT ⇒ O(N)-size G1 SRS (HyperKZG/PST, trusted) or Pedersen+IPA (transparent, O(N) native verifier).

## 5. Verdict, Pareto frontier, recommendation

**Answer to the gate question: no.** With the production Blake2b transcript and Dory kept, the natively-held data floor is 16.9 KB @2^18 (§3) and the wrapper adds 4.5 KB → ≥ 21.4 KB; every way of hiding the GT elements costs ≥ 26M constraints (their Blake2b bytes) plus either ≥ 0.65B non-native constraints (GT exps in Fr) or an Fq sub-proof with ≥ 3.5M Fr verification constraints and 3–50 s prover. The circuit cost of *deferred* Dory is ~0, so "offloading" in the user's sense is trivially achievable — the casualty is bytes, not constraints, and it is information-theoretic (the verifier must hold 105 uninvertible GT values) plus transcript-binding (their bytes are hashed), not an artefact of the current code.

| # | design | in-circuit Dory | bytes @2^18 / @2^20 | native verifier | Jolt-prover Δ | Dory kept |
|---|---|---|---|---|---|---|
| P1 | (i) deferred-native, 128-B GT codec (+§3 preimage trick) | 0–0.6k | **21.4 / 22.5 KB** (21.8 / 22.9 without trick) | ≤ 68–80 ms (measured) | 0 | ✓ same proof, byte-identical transcript |
| P1′ | (i) with 384-B GT | 0–0.6k | 49.7 / 52.3 KB | same | 0 | ✓ |
| P6 | (vi) MIPP_k + IPA over Dory-form commitments, structured Γ2 | 0 | ≈14.2 KB (est.) | ≈15–25 ms (est.) | new opening prover | commitment form ✓, opening/setup ✗ |
| P5 | HyperKZG/PST inner PCS (user-rejected) | 0 | ≈8.6 KB (1,312 + 2,816 + 4,512) | ≈3–5 ms | commit/open change; 2^n trusted SRS | ✗ |
| P4 | (v) Pedersen+IPA, n2=1 | 0 | ≈7.3 KB | 0.15–0.4 s (est., 2^22 MSM) | +1–3 s (est.) | ✗ (transparent, O(N) verifier) |
| P3 | (iv) Fq sub-proof | ≥ 3.5M (≥ 26M to hide GT bytes) | 4.5 KB only with the 37M-Fq-constraint Blake2b proof, else ≥ 21 KB | small | +3–50 s (+85–590 s) | ✓ |
| P2 | (ii)/(iii) non-native in-circuit | 1.2–8.8 B | 4.5 KB (theoretical) | small | infeasible | ✓ |

**Recommendation.**
1. If "Dory kept" is the binding constraint: **P1**. Relax "single-digit KB" to ≈ 21–23 KB (128-B codec, lane G4). Zero Dory constraints, zero protocol change (wire format only if the preimage trick is used), production soundness. Everything else in this study is dominated.
2. If "single-digit KB" is binding: relax "Dory kept" — **P5** (8.6 KB, fastest verifier, trusted 2^n SRS) or **P4** (7.3 KB, transparent, 0.15–0.4 s native verify, +1–3 s prover). P6 (14 KB) is the only Dory-*form* middle ground and still misses the target.
3. Independently, BLOCKER #1 stands: the Blake2b transcript alone is 106M constraints @2^18 (`blake2b-gadget.md`); the Dory decision only matters once the transcript question is settled.

**Oracle cross-check (`oracle-dory-offload.md`):** agrees on the algebra (§1), the floors, (i)–(v), the four-preimage trick and the layout optimum. Differences: (a) it assumes a 6,144-B wrapper → 23.4 KB; the planner's exact 3,616 B + 896 B IO gives 21.4–21.8 KB; (b) its §4.6 "TIPP/MIPP ≈ 7.8–10.3 KB with an authenticated aggregate commitment" — TIPP has the same 6 GT/round as Dory, MIPP_k 2 GT/round, and the aggregate cannot be authenticated without the 41 C_i (or a 41-exp proof) *and* their bytes are needed by the transcript regardless, so the (vi) floor is ≈ 14 KB; (c) its (iv) estimate 150k–500k omits the Grumpkin PCS in-circuit cost (≥ 0.8M MSM + 2–5.5M combined-row Fq mults) and the Blake2b-of-GT-bytes floor; (d) unit GT-exp cost 21–57M (its 55–150k per Fq12 mult) vs 16–43.5M here (54 Fq mults per Fq12 mult, 36 per squaring) — same order.

**Exactness ledger.** Exact from code/paper: all equations in §1, element/scalar/base counts, transcript compression counts, byte tables at 128/384 B, layout constraints, the four single-pair elements, wrapper 3,616 B, measured verifier 67.8/79.7 ms, Blake2b 52,416/compression. Estimates: Fq-mult constraint cost (1,000–2,716), Fq12/G1/G2/pairing mult counts, Straus totals, Fq prover throughput, Grumpkin MSM costs, native MSM/IPA timings, P4/P6 prover deltas, MIPP_k/IPA byte counts (structured from BMMTV §5–6 without re-deriving the exact element lists).
