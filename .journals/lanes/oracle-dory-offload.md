## Executive conclusion

**No design based only on “advice + random linear combination” satisfies all four goals simultaneously for the current Jolt/Dory protocol:**

1. Dory retained as the inner PCS,
2. Dory contribution below roughly 100k Fr-R1CS constraints,
3. total wrapper proof in the single-digit-kilobyte range,
4. wrapper proving below one second at at most \(2^{18}\) constraints.

The obstruction is not the final pairing. It is the roughly \(N+10\sigma+4\) target-group scalar multiplications, plus subgroup checks and transcript binding. A random linear combination batches equality checks but does not turn non-native multiplication in \(\mathbb F_{q^{12}}\) into native \(\mathbb F_r\) arithmetic.

There are only three viable directions:

* keep Dory verification native and send its elements: about **23.4 KB at \(L=18\)** and **24.4 KB at \(L=20\)** including a 6 KB wrapper, assuming 128-byte GT compression;
* replace the Dory opening by a different structured/TIPP-style argument and change the setup/protocol;
* recursively prove the Dory computation with another proof system, accepting substantially more prover time and/or more than \(2^{18}\) wrapper work.

The current Blake2b Fiat–Shamir transcript is independently fatal to an Fr-native wrapper. It must also be replaced by an algebraic transcript or handled by another proof.

---

# 1. Closed-form verifier algebra

I use additive notation for \(G_T\), matching the Rust `Group` abstraction:

\[
X+Y \quad\text{means multiplication in the conventional multiplicative notation,}
\]
\[
[s]X \quad\text{means }X^s.
\]

Let:

* \(n=L+4\) for current Jolt parameters;
* \(\sigma=\lceil n/2\rceil\);
* \(\rho\) be the stage-8 randomizer used to combine the \(N=41\) commitments;
* \((\beta_t,\alpha_t)\), \(t=1,\ldots,\sigma\), be the Dory round challenges;
* \(g\) be the final fold-scalars challenge called `gamma` in the code;
* \(d\) be the final batching challenge;
* \(k_t=\sigma-t+1\) be the setup level used in round \(t\).

All inverses below are Fr inverses. Thus the coefficients are more precisely Laurent polynomials in the challenges, or ordinary polynomials after adjoining inverse variables constrained by \(x x^{-1}=1\).

## 1.1 Initial state

For commitments \(K_0,\ldots,K_{N-1}\in G_T\),

\[
D_{1,0}=K_{\rm joint}:=\sum_{i=0}^{N-1}\rho^i K_i.
\]

From the VMV message,

\[
C_0=C_{\rm vmv},\qquad
D_{2,0}=D_{2,\rm vmv},\qquad
E_{1,0}=E_{1,\rm vmv}.
\]

In transparent mode,

\[
E_{2,0}=[y]\Gamma_{2,0}.
\]

The initial folded-scalar accumulators are

\[
s_{1,0}=s_{2,0}=1.
\]

The point coordinates are stored as \(y_t\) on the \(s_1\) side and \(x_t\) on the \(s_2\) side, in the code's MSB-first consumption order.

## 1.2 Per-round messages

For round \(t\), abbreviate:

\[
A_t=D_{1L,t},\quad A'_t=D_{1R,t},
\]
\[
B_t=D_{2L,t},\quad B'_t=D_{2R,t},
\]
\[
P_t=C_{+,t},\quad M_t=C_{-,t}.
\]

The G1 messages are

\[
U_{\beta,t}=E_{1\beta,t},\quad
U_{+,t}=E_{1+,t},\quad
U_{-,t}=E_{1-,t},
\]

and the G2 messages are

\[
V_{\beta,t}=E_{2\beta,t},\quad
V_{+,t}=E_{2+,t},\quad
V_{-,t}=E_{2-,t}.
\]

The exact verifier updates are:

\[
\boxed{
C_t=C_{t-1}+\chi_{k_t}
+[\beta_t]D_{2,t-1}
+[\beta_t^{-1}]D_{1,t-1}
+[\alpha_t]P_t
+[\alpha_t^{-1}]M_t
}
\]

\[
\boxed{
D_{1,t}
=[\alpha_t]A_t+A'_t
+[\alpha_t\beta_t]\Delta_{1L,k_t}
+[\beta_t]\Delta_{1R,k_t}
}
\]

\[
\boxed{
D_{2,t}
=[\alpha_t^{-1}]B_t+B'_t
+[\alpha_t^{-1}\beta_t^{-1}]\Delta_{2L,k_t}
+[\beta_t^{-1}]\Delta_{2R,k_t}
}
\]

\[
\boxed{
E_{1,t}=E_{1,t-1}
+[\beta_t]U_{\beta,t}
+[\alpha_t]U_{+,t}
+[\alpha_t^{-1}]U_{-,t}
}
\]

\[
\boxed{
E_{2,t}=E_{2,t-1}
+[\beta_t^{-1}]V_{\beta,t}
+[\alpha_t]V_{+,t}
+[\alpha_t^{-1}]V_{-,t}
}
\]

and

\[
\boxed{
s_{1,t}
=s_{1,t-1}\bigl(\alpha_t(1-y_t)+y_t\bigr)
}
\]

\[
\boxed{
s_{2,t}
=s_{2,t-1}\bigl(\alpha_t^{-1}(1-x_t)+x_t\bigr).
}
\]

Consequently,

\[
s_1:=s_{1,\sigma}
=\prod_{t=1}^{\sigma}\bigl(\alpha_t(1-y_t)+y_t\bigr),
\]

\[
s_2:=s_{2,\sigma}
=\prod_{t=1}^{\sigma}\bigl(\alpha_t^{-1}(1-x_t)+x_t\bigr).
\]

The accumulated side elements have the closed forms

\[
E_1=
E_{1,0}
+\sum_{t=1}^{\sigma}
\left(
[\beta_t]U_{\beta,t}
+[\alpha_t]U_{+,t}
+[\alpha_t^{-1}]U_{-,t}
\right),
\]

\[
E_2=
E_{2,0}
+\sum_{t=1}^{\sigma}
\left(
[\beta_t^{-1}]V_{\beta,t}
+[\alpha_t]V_{+,t}
+[\alpha_t^{-1}]V_{-,t}
\right).
\]

## 1.3 Transparent final check

Let the final transparent message be

\[
(F_1,F_2)\in G_1\times G_2.
\]

The code verifies

\[
\begin{aligned}
& e(F_1+[d]\Gamma_{1,0},\,F_2+[d^{-1}]\Gamma_{2,0})\\
&\quad
+e\!\left(H_1,\,
[-g]\left(E_2+[d^{-1}s_1]\Gamma_{2,0}\right)\right)\\
&\quad
+e\!\left(
[-g^{-1}]\left(E_1+[d s_2]\Gamma_{1,0}\right),H_2
\right)\\
&\quad
+e([d^2]E_{1,0},\Gamma_{2,0})\\
&=
C_\sigma+[s_1s_2]H_T+\chi_0
+[d]D_{2,\sigma}
+[d^{-1}]D_{1,\sigma}
+[d^2]D_{2,0}.
\end{aligned}
\]

Here

\[
H_T=e(H_1,H_2).
\]

This is exactly the one four-way multipairing in `verify_final`.

## 1.4 Fully expanded target-group right-hand side

Define, for each round \(q\),

\[
u_q=
\begin{cases}
\beta_{q+1}^{-1}, & q<\sigma,\\
d^{-1}, & q=\sigma,
\end{cases}
\]

\[
v_q=
\begin{cases}
\beta_{q+1}, & q<\sigma,\\
d, & q=\sigma.
\end{cases}
\]

These are the coefficients with which \(D_{1,q}\) and \(D_{2,q}\) enter the eventual right-hand side: through the next round's \(C\)-update, or through the final \(d^{-1}D_1+dD_2\) slots.

The complete RHS is therefore:

\[
\begin{aligned}
R={}&
C_0
+[\beta_1^{-1}]D_{1,0}
+[\beta_1+d^2]D_{2,0}\\
&+\sum_{t=1}^{\sigma}
\left(
\chi_{k_t}
+[\alpha_t]P_t
+[\alpha_t^{-1}]M_t
\right)
+\chi_0+[s_1s_2]H_T\\
&+\sum_{q=1}^{\sigma}
u_q\left(
[\alpha_q]A_q+A'_q
+[\alpha_q\beta_q]\Delta_{1L,k_q}
+[\beta_q]\Delta_{1R,k_q}
\right)\\
&+\sum_{q=1}^{\sigma}
v_q\left(
[\alpha_q^{-1}]B_q+B'_q
+[\alpha_q^{-1}\beta_q^{-1}]\Delta_{2L,k_q}
+[\beta_q^{-1}]\Delta_{2R,k_q}
\right).
\end{aligned}
\]

Expanding \(D_{1,0}\),

\[
[\beta_1^{-1}]D_{1,0}
=
\sum_{i=0}^{N-1}
[\beta_1^{-1}\rho^i]K_i.
\]

Thus every target-group input \(X_k\) and coefficient \(s_k\) is:

| Target-group element \(X_k\) | Coefficient \(s_k\) |
|---|---:|
| \(C_{\rm vmv}\) | \(1\) |
| commitment \(K_i\) | \(\beta_1^{-1}\rho^i\) |
| \(D_{2,\rm vmv}\) | \(\beta_1+d^2\) |
| \(C_{+,t}\) | \(\alpha_t\) |
| \(C_{-,t}\) | \(\alpha_t^{-1}\) |
| \(D_{1L,t}\) | \(u_t\alpha_t\) |
| \(D_{1R,t}\) | \(u_t\) |
| \(D_{2L,t}\) | \(v_t\alpha_t^{-1}\) |
| \(D_{2R,t}\) | \(v_t\) |
| \(\Delta_{1L,k_t}\) | \(u_t\alpha_t\beta_t\) |
| \(\Delta_{1R,k_t}\) | \(u_t\beta_t\) |
| \(\Delta_{2L,k_t}\) | \(v_t\alpha_t^{-1}\beta_t^{-1}\) |
| \(\Delta_{2R,k_t}\) | \(v_t\beta_t^{-1}\) |
| \(\chi_{k_t}\) | \(1\) |
| \(\chi_0\) | \(1\) |
| \(H_T\) | \(s_1s_2\) |

Physical setup aliases must be combined. In particular,

\[
\Delta_{2L,k}=\Delta_{1L,k}=\chi_{k-1}.
\]

So the coefficient of a physical setup element is the sum of all rows above referring to that element.

This expansion shows directly that, except with negligible probability of a challenge-induced zero coefficient, **all \(6\sigma+2\) proof GT elements and all \(N\) commitments participate**.

## 1.5 Counts of held values

For transparent Dory:

* proof GT: \(6\sigma+2\);
* proof G1: \(3\sigma+2\);
* proof G2: \(3\sigma+1\);
* commitments before RLC: \(N\);
* or one joint commitment after a separately authenticated RLC.

Distinct setup GT values actually needed, accounting for
\(\Delta_{2L}=\Delta_{1L}\) and \(\Delta_{1L,k}=\chi_{k-1}\), are at most

\[
(\sigma+1)\ \chi\text{'s}
+\sigma\ \Delta_{1R}
+\sigma\ \Delta_{2R}
+1\ H_T
=3\sigma+2.
\]

Therefore a conventional verifier holding all proof data, commitments and distinct setup GT values holds

\[
N+(6\sigma+2)+(3\sigma+2)
=
N+9\sigma+4
\]

distinct GT values, excluding temporary accumulators.

For \(N=41,\sigma=11\), this is 144 GT values, of which 68 are proof elements and 41 are commitments.

The independent Dory transcript challenges are:

* \(2\sigma\) values \((\alpha_t,\beta_t)\);
* fold challenge \(g\);
* final challenge \(d\);
* stage-8 commitment RLC challenge \(\rho\).

Thus \(2\sigma+3\) challenge scalars, plus their inverses and derived products. The opening point contributes \(L+4\) Fr coordinates, and there is one evaluation scalar.

---

# 2. Linear versus nonlinear work

## 2.1 Fr-native work

Cheap native operations include:

* challenge powers \(\rho^i\);
* all coefficients in the table above;
* \(\alpha^{-1},\beta^{-1},g^{-1},d^{-1}\), each enforced by one witness inverse and one multiplication;
* \(s_1,s_2\);
* all sumcheck arithmetic;
* formation of scalar coefficients for G1, G2 and GT linear combinations;
* batching coefficients and equality of Fr digests.

This part is at most a few thousand constraints for Dory itself.

## 2.2 Abstract group-linear work

The following are linear in the corresponding prime-order group:

* \(D_{1,0}=\sum_i \rho^i K_i\) in \(G_T\);
* all GT state updates;
* all G1 and G2 accumulators;
* construction of the four pairing inputs.

Abstractly these are MSMs.

But they are cheap only for an entity that can natively manipulate the group. In an Fr circuit, BN254's:

* \(G_1\) coordinates are in \(\mathbb F_q\);
* \(G_2\) coordinates are in \(\mathbb F_{q^2}\);
* \(G_T\) coordinates are in \(\mathbb F_{q^{12}}\).

Therefore “group-linear” does not imply “Fr-circuit-linear.”

## 2.3 Intrinsically non-native operations

In a plain Fr R1CS, the expensive operations are:

* multiplication and squaring in \(\mathbb F_q\);
* multiplication in \(\mathbb F_{q^{12}}\);
* scalar multiplication of a witness-dependent BN254 point;
* target-group exponentiation of a witness-dependent GT element;
* pairing/Miller-loop arithmetic;
* subgroup checks;
* decompression and on-curve checks.

A GT exponentiation

\[
Y=[s]X
\]

is abstractly a scalar multiplication in a cyclic group, but on coordinates it is a sequence of witness-dependent \(\mathbb F_{q^{12}}\) multiplications. With the implementation's generic exponentiation, it is about 254 squarings plus 127 multiplies, or roughly 380 multiplication-like steps.

At the supplied costs:

\[
380\cdot(55\text{k--}150\text{k})
\approx 21\text{M--}57\text{M}
\]

constraints per generic GT exponentiation.

## 2.4 What random-linear-combination batching does

Suppose the prover gives claimed multiplication triples

\[
z_i=x_i y_i.
\]

A random combination can replace many equality outputs by

\[
\sum_i r^i(z_i-x_iy_i)=0.
\]

But every product \(x_i y_i\) still has to be computed or constrained. The batching saves equality gates and sometimes commitment openings; it does not save the multiplication constraints.

The same applies to a chain

\[
X_{j+1}=X_j^2
\quad\text{or}\quad
X_{j+1}=X_jX.
\]

One can batch the residuals, but each residual contains a separate non-native multiplication of witness-dependent operands. Omitting those products makes the advice unconstrained.

This is analogous to Freivalds' check for matrix multiplication: it reduces a cubic verification to matrix-vector products because the verifier can still natively multiply field elements. Here the wrapper cannot natively multiply \(\mathbb F_{q^{12}}\) elements.

## 2.5 Why the proposed algebraic shortcuts do not solve it

### Challenge scalars are public

This permits precomputation only when the base is fixed. Most GT bases here are proof or commitment elements. The circuit must check exponentiation of witness-dependent bases by challenge-dependent scalars.

Fixed-base setup exponentiations could be table-precomputed outside the circuit, but the circuit would still need to authenticate the selected result. Without lookups or a separate argument, this is not free.

### Cyclotomic or torus structure

Cyclotomic squaring and torus compression reduce constants and communication. They do not move the computation into \(\mathbb F_r\).

Even a factor-of-two or factor-of-four improvement is irrelevant against a gap of several orders of magnitude.

### “All GT values arise as pairings”

Not all are supplied with succinct preimages under the actual random-generator setup. For example,

\[
D_{1L}=\sum_i e(v_{1L,i},\Gamma_{2,i})
\]

is a multipairing over a vector of unrelated \(\Gamma_{2,i}\). There is generally no efficiently computable \(P\in G_1,Q\in G_2\) such that the prover can send only \((P,Q)\) and the verifier can infer this exact sum.

Existence is not enough: because \(G_T\) is prime order, every GT element has a mathematical preimage under \(P\mapsto e(P,\Gamma_{2,0})\), but finding that preimage from a generic GT value requires solving a discrete-log-type problem.

### TIPP/MIPP or KZG-authenticated vectors

These can yield succinct verification only after adding a structured SRS and an additional argument authenticating vector relations or inner pairing products.

That is a valid redesign, but it is not “advice plus an Fr RLC” applied to unchanged Dory. It is a new succinct argument layered on top of, or replacing, the current Dory opening.

### Constant native pairings on sent succinct elements

A constant number of native pairings can verify a constant number of pairing-product equations over sent G1/G2 elements. It cannot certify arbitrary coefficients of \(6\sigma+2+N\) independent GT values unless those values have:

1. succinct, authenticated G1/G2 representations, or
2. a separate proof that they satisfy the multi-exponentiation.

The current Dory setup supplies neither.

---

# 3. Minimal-knowledge argument

## 3.1 Commitments

The verifier need not conceptually hold all \(N\) commitments if it is given a correctly authenticated joint commitment

\[
K_{\rm joint}=\sum_i\rho^iK_i.
\]

But establishing that equality requires one of:

* all \(K_i\), so the native verifier computes it;
* a commitment scheme or proof for the vector \((K_i)\);
* changing the inner protocol so only the aggregate commitment is ever committed/transcript-bound;
* treating the commitments as separately authenticated public preprocessing.

Thus the strongest correct statement is:

> The Dory verifier needs either all \(N\) commitments or an authenticated fixed RLC of them.

Merely letting the prover send \(K_{\rm joint}\) is unsound because the prover could choose it after seeing the opening claim and Dory challenges.

## 3.2 Dory proof GT elements

The expanded equation assigns a generically nonzero coefficient to every one of:

* VMV \(C\) and \(D_2\);
* \(D_{1L},D_{1R},D_{2L},D_{2R}\) in each round;
* \(C_+,C_-\) in each round.

That is exactly

\[
2+4\sigma+2\sigma=6\sigma+2.
\]

Without another proof system, every such value must be available to the party evaluating the GT linear combination.

A verifier can receive them directly, or receive a succinct authenticated representation from which they can be reconstructed. Current Dory provides only the former.

## 3.3 Which proof GT elements have succinct preimages?

In transparent mode there are four immediate exceptions arising from the initial fixed-\(\Gamma_{2,0}\) structure.

The VMV elements are:

\[
C_{\rm vmv}=e(P_C,\Gamma_{2,0}),
\quad
P_C=\operatorname{MSM}(\text{row commitments},v),
\]

\[
D_{2,\rm vmv}=e(P_D,\Gamma_{2,0}),
\quad
P_D=\operatorname{MSM}(\Gamma_1,v).
\]

Likewise, before the first \(\beta\)-update,

\[
D_{2L,1}=e(P_L,\Gamma_{2,0}),\qquad
D_{2R,1}=e(P_R,\Gamma_{2,0}),
\]

because initially \(v_{2,i}=v_i\Gamma_{2,0}\).

These four GT values can therefore be replaced by four G1 values and reconstructed by native pairings.

At 128 bytes per GT and 32 bytes per G1, this saves

\[
4(128-32)=384\text{ bytes},
\]

at the cost of up to four additional Miller-loop terms, possibly batchable with other native pairing work.

This requires the transcript to absorb the reconstructed GT encodings if compatibility with the current transcript is desired.

After the first update,

\[
v_2\leftarrow v_2+\beta^{-1}\Gamma_2,
\]

and subsequent folds mix unrelated G2 setup generators. The analogous succinct G1 preimages are no longer computable from the public random-generator setup.

No analogous compression exists for generic:

* commitments;
* \(D_1\) messages;
* \(C_+\) or \(C_-\);
* later-round \(D_2\) messages.

## 3.4 Byte floor

At \(L=18\), \(\sigma=11\):

* proof GT: 68;
* commitments: 41.

Even with 128-byte GT encoding, the commitments alone are

\[
41\cdot128=5{,}248\text{ bytes}.
\]

A 6 KB wrapper leaves less than approximately 4 KB for all remaining material. Therefore:

> Any design that sends all 41 commitments cannot achieve a total proof below 10 KB, even if the Dory opening itself were free.

This is an unconditional communication floor under the stated byte sizes.

---

# 4. Candidate designs

The following totals omit a few dozen to a few hundred bytes of framing and public scalar metadata. “6 KB wrapper” is taken as approximately 6,144 bytes.

## 4.1 (i) Fr circuit computes scalars; native verifier performs Dory

The circuit computes:

* all Dory challenges or a transcript state;
* inverses and derived coefficients;
* \(s_1,s_2\);
* the coefficient table for the native GT MSM;
* the four pairing-input scalar coefficients.

The outer verifier receives all GT/G1/G2 elements and evaluates the group equation natively.

### Dory circuit cost

Ignoring transcript binding:

* about 500–2,000 Fr constraints;
* no non-native arithmetic.

With a field-native transcript, Dory-specific transcript work is still modest. With the existing Blake2b byte transcript, it is infeasible.

### Native verifier work

Exactly the current online work:

| \(L\) | \(\sigma\) | GT exps incl. 41 RLC | GT mults | G1 muls | G2 muls |
|---:|---:|---:|---:|---:|---:|
| 18 | 11 | 155 | 166 | 37 | 37 |
| 20 | 12 | 165 | 177 | 40 | 40 |

plus one four-way multipairing and subgroup/decompression checks.

A single native GT multi-exponentiation algorithm could improve constants relative to 155 independent calls, but not communication.

### Proof bytes

Compressed Dory data:

\[
(6\sigma+2)128+(3\sigma+2)32+(3\sigma+1)64.
\]

| \(L\) | \(\sigma\) | Dory opening | 41 commitments | wrapper | total |
|---:|---:|---:|---:|---:|---:|
| 18 | 11 | 12,000 B | 5,248 B | 6,144 B | **23,392 B** |
| 20 | 12 | 13,056 B | 5,248 B | 6,144 B | **24,448 B** |

The four-preimage optimization reduces each total by 384 bytes.

### Soundness condition

This split verifier is sound only if the wrapper relation binds:

* the exact Dory elements;
* the exact commitments or their authenticated RLC;
* the Fiat–Shamir challenges;
* the Jolt claims passed into Dory.

Simply exposing challenge scalars without binding the group elements that were hashed before them is unsound.

### Verdict

Constraint-feasible, but not single-digit KB and not literally “Dory verified inside the R1CS.”

---

## 4.2 (ii) Advice for GT intermediates, checked non-natively

The prover supplies all powers and products used in the GT exponentiations. The circuit verifies each multiplication.

At \(L=20\):

* 165 GT exponentiations including commitment RLC;
* about 380 Fq12 multiplication-like steps per exponentiation;
* 177 additional Fq12 products.

This is approximately

\[
165\cdot380+177 \approx 62{,}877
\]

Fq12 multiplication-like checks.

At 54 Fq multiplications per Fq12 multiplication:

\[
62{,}877\cdot54\approx3.40\text{ million Fq multiplications}.
\]

At 1,000–2,700 constraints per Fq multiplication:

\[
\boxed{3.4\text{--}9.2\text{ billion constraints}}
\]

before pairings, G1/G2 operations or subgroup checks.

If the GT elements are private wrapper witnesses, the transmitted proof may remain near 6 KB, but the R1CS is billions of constraints and the prover time is entirely unacceptable.

### Verdict

Cryptographically sound if fully constrained; operationally impossible.

---

## 4.3 (iii) Advice plus one random batching equation

Suppose each non-native relation has residual \(R_i\), and the circuit checks

\[
\sum_i \lambda^i R_i=0.
\]

To compute \(R_i\), it still needs the claimed product:

\[
R_i=Z_i-X_iY_i.
\]

Therefore every \(X_iY_i\) remains a non-native multiplication. Batching saves only the final equality comparisons and possibly some limb normalization.

If instead the prover supplies \(R_i\) without the circuit computing \(X_iY_i\), it can set every residual to zero independently of the alleged products.

Thus the lower-order benefit is perhaps a few percent, while the count remains approximately 3.2–3.4 million Fq multiplications online.

### Verdict

No asymptotic or budget-relevant saving. The proposed offload does not work by itself.

---

## 4.4 (iv) Fq-native proof using Grumpkin commitments

The idea is logically sound:

1. express the BN254 GT arithmetic and pairing as a circuit over \(\mathbb F_q\);
2. prove that circuit using Spartan over \(\mathbb F_q\);
3. use Grumpkin, whose scalar field is \(\mathbb F_q\), for the Fq proof commitment;
4. verify that secondary proof inside the main Fr circuit, where Grumpkin coordinates are native Fr values.

There are two costs.

### Secondary Fq prover

The Dory verifier relation is approximately:

* 3.2–3.4 million Fq multiplications without deserialization checks;
* around 5 million if subgroup/deserialization checks are included.

A Spartan prover for that relation is far beyond the sub-second target. A realistic CPU estimate is multiple seconds to tens of seconds, depending on implementation and parallelism.

### Fr verification of the Fq proof

The sumcheck scalar field is \(\mathbb F_q\), so its arithmetic is non-native in the Fr circuit. A verifier with, for example, 20–25 rounds and degree-2/3 round polynomials requires on the order of 60–150 Fq multiplications after accounting for interpolation, claim updates and transcript arithmetic.

That alone costs approximately:

\[
60\text{--}150 \times 1{,}000\text{--}2{,}700
=
60\text{k--}405\text{k constraints}.
\]

The Grumpkin opening adds group work. At the supplied estimate of roughly 2,500 constraints per 254-bit scalar multiplication:

* 20 scalar multiplications already cost about 50k;
* a logarithmic IPA/Hyrax opening generally needs more than that.

A reasonable estimate for the whole in-circuit secondary verifier is:

\[
\boxed{150\text{k--}500\text{k constraints}}
\]

depending strongly on the commitment and transcript design.

This is not safely below 100k, and likely exceeds the entire \(2^{18}\) wrapper once the Jolt verifier and transcript are included.

### Proof bytes

If the Fq proof is only a private witness to the outer wrapper, final communication can remain around 6 KB. This is the one candidate that can hide all Dory elements while preserving soundness.

### Verdict

Potentially succinct in bytes, but not within the constraint or sub-second prover budgets.

---

## 4.5 (v) Different Dory matrix layouts

Let the polynomial have \(m=L+4\) variables and be arranged as

\[
2^k\times 2^{m-k}.
\]

For the symmetric Dory reduction generalized to unequal dimensions, the padded reduction length is

\[
r=\max(k,m-k).
\]

The proof sizes become approximately:

\[
6r+2\quad G_T,
\]
\[
3r+2\quad G_1,
\]
\[
3r+1\quad G_2.
\]

This is minimized when \(k\) is as close as possible to \(m/2\). Therefore the current square or near-square layout already minimizes the number of Dory rounds and proof group elements.

The attached verifier implementation is even more restrictive:

* it enforces \(\nu\le \sigma\);
* it sets `num_rounds = sigma`.

So arbitrary layout changes require code/protocol changes.

### Extreme one-row layout

With one row and \(2^m\) columns, the commitment becomes

\[
C=e\!\left(\sum_{i=0}^{2^m-1}a_i\Gamma_{1,i},\Gamma_{2,0}\right).
\]

The G1 value

\[
P=\sum_i a_i\Gamma_{1,i}
\]

is a Pedersen-style vector commitment, and the GT commitment is its pairing image.

One can then use a Bulletproofs-style inner-product argument in G1:

* about \(2m\) G1 elements;
* at \(m=22\), approximately \(44\cdot32=1,408\) bytes;
* at \(m=24\), approximately 1,536 bytes;
* plus a few scalars/final points.

But with an unstructured generator vector, the verifier must form the folded generator combination, requiring an \(O(2^m)\) native G1 MSM unless an enormous precomputed table or structured SRS is introduced.

It is not the same reduce-and-fold protocol implemented in `dory-pcs 0.4.2`. It preserves an AFGHO-shaped commitment in the degenerate one-row case, but replaces the opening argument by a G1 IPA.

If verified inside the Fr circuit, BN254 G1 arithmetic remains non-native and is far over 100k. If verified by the outer native verifier, the circuit only computes Fr coefficients, but the protocol again becomes split verification.

### Communication

If all 41 GT commitments are still sent:

| \(L\) | wrapper | commitments | approximate IPA | total |
|---:|---:|---:|---:|---:|
| 18 | 6,144 | 5,248 | 1,500–1,800 | **12.9–13.2 KB** |
| 20 | 6,144 | 5,248 | 1,600–1,900 | **13.0–13.3 KB** |

If only one authenticated aggregate commitment is needed, totals can fall near 8 KB. But authenticating that aggregate is exactly the missing additional argument.

### Verdict

Interesting if verifier time and setup are allowed to change radically; not a solution for current Dory.

---

## 4.6 (vi) TIPP/MIPP-style opening with structured generator vectors

A structured setup could choose or authenticate generator vectors so that inner pairing products admit succinct verification. Conceptually:

* commit to \(\Gamma_1,\Gamma_2\) under KZG-like commitments;
* prove the relevant vector folds and inner pairing products using TIPP/MIPP;
* reduce verification to \(O(\log n)\) sent group elements and a constant or logarithmic number of native pairing checks.

The likely communication scale is:

* \(O(\log n)\) G1/G2 elements, roughly 1.5–4 KB at \(m=22\)–24;
* a constant number of Fr scalars;
* one or several GT/commitment values.

The in-circuit work can then be almost entirely Fr scalar arithmetic if the outer verifier performs the pairing equations natively.

However:

1. the current Dory URS consists of independent random generators, not a KZG powers-of-\(\tau\) SRS;
2. KZG-committing the existing random generator arrays does not by itself reveal algebraic relations needed to compress arbitrary inner pairing products;
3. soundness requires a formal TIPP/MIPP argument under additional assumptions;
4. the opening protocol and transcript change;
5. setup generation becomes structured/trusted or requires an updatable ceremony.

### Estimated total bytes

With all 41 original commitments:

\[
6.1\text{ KB wrapper}
+5.25\text{ KB commitments}
+1.5\text{--}4\text{ KB opening}
=
12.9\text{--}15.4\text{ KB}.
\]

With one authenticated aggregate commitment:

\[
6.1+0.128+1.5\text{--}4
=
7.8\text{--}10.3\text{ KB}.
\]

The low end reaches the requested class, but only after solving commitment aggregation and replacing current Dory verification.

### Verdict

The most promising protocol redesign, but not an offload of unchanged Dory.

---

## 4.7 (vii) Prove the GT trace with HyperKZG/sumcheck

The Dory verifier trace is approximately:

* 3.2 million Fq multiplications;
* roughly 13 million 64-bit limb products.

Putting those products directly into the main R1CS requires at least one constraint per limb product, therefore on the order of

\[
\boxed{13\text{ million constraints}}
\]

before carry/range constraints. This is about \(2^{24}\), not \(2^{18}\).

At the reported HyperKZG timings:

* \(2^{18}\): about 470 ms;
* \(2^{19}\): already about 1.25 s total wrapper scale.

A \(2^{24}\)-scale commitment/opening and Spartan relation is not remotely sub-second. Extrapolation is implementation-dependent, but tens of seconds rather than milliseconds is the correct order.

A separate HyperKZG proof can reduce communication, but verifying that proof inside the same Fr circuit requires BN254 G1/KZG operations, again non-native. If the outer verifier checks it natively, it becomes a second externally verified proof and total communication is likely around 12 KB or more.

### Verdict

Sound in principle, but violates the prover-time and relation-size budgets by roughly two orders of magnitude.

---

# 5. Fiat–Shamir binding is a separate blocker

The proposed split must not overlook that the current Jolt transcript absorbs:

* all 41 commitments;
* all Dory GT/G1/G2 messages;
* all sumcheck polynomials and claims.

A circuit that exposes only challenge scalars but does not prove they were derived from those exact group encodings allows the prover to choose incompatible challenges and group elements.

There are only three sound ways to bridge the circuit and native Dory verifier:

1. **Replay the exact transcript in-circuit.**  
   Blake2b costs approximately 63 million constraints according to the supplied report, so this is impossible.

2. **Change Jolt to a field-native transcript.**  
   This is feasible cryptographically but changes the proof protocol. Even the supplied Poseidon estimates are about 140k–270k constraints for the whole Jolt transcript, before Dory.

3. **Use an authenticated digest boundary.**  
   The native verifier hashes the group elements and the circuit consumes that digest. But for equivalence to the original transcript, the circuit must prove or define how that digest enters the challenge derivation. This again changes the transcript unless the original Blake computation is proved.

Therefore design (i) is practical only with a redesigned algebraic transcript and careful domain-separated digest bridge.

---

# 6. Summary table

Estimates marked with “est.”

| Design | Dory-related Fr constraints | Native outer work | Total bytes \(L=18\) | Total bytes \(L=20\) | Extra prover cost | Status |
|---|---:|---|---:|---:|---|---|
| (i) Scalar circuit, native Dory, all data sent | 0.5k–2k excluding transcript | 155/165 GT exps, 4-pair check | **23.4 KB** | **24.4 KB** | small | Sound with transcript binding; misses byte goal |
| (ii) Full non-native advice verification | 3.4–9.2B | negligible | ≈6 KB possible | ≈6 KB possible | prohibitive | Sound, infeasible |
| (iii) Batched non-native residuals | still billions | negligible | ≈6 KB possible | ≈6 KB possible | prohibitive | Batching does not remove products |
| (iv) Fq Spartan + Grumpkin recursion | 150k–500k est. | small | ≈6 KB | ≈6 KB | multi-second Fq proof est. | Misses constraints/time |
| (v) Degenerate G1 IPA, all commitments | Fr-only if checked outside | huge \(O(2^{L+4})\) G1 MSM | 12.9–13.2 KB | 13.0–13.3 KB | very high verifier/prover work | Not current Dory |
| (vi) TIPP/MIPP, all commitments | likely \(<100\)k with external pairings | constant/log pairings | 12.9–15.4 KB est. | similar | protocol redesign | Promising redesign |
| (vi), authenticated aggregate commitment | likely \(<100\)k | constant/log pairings | **7.8–10.3 KB est.** | similar | new SRS/argument | Can approach target, not unchanged Dory |
| (vii) GT trace in main wrapper | \(\ge13\)M | small | ≈6 KB | ≈6 KB | tens of seconds est. | Infeasible |

---

# 7. Verdict and Pareto frontier

## Final verdict

**No**, there is no design satisfying all stated requirements while keeping the current Jolt Dory commitment/opening protocol and relying only on advice, Fr random-linear-combination checks, and a constant number of native outer pairings.

Proof sketch:

1. The expanded verifier equation contains every one of the \(6\sigma+2\) proof GT elements and all commitments or their authenticated RLC with generically nonzero coefficients.
2. For the random-generator AFGHO setup, most of these GT values do not have efficiently computable succinct G1/G2 preimages.
3. Therefore either:
   * those GT values are sent and combined natively, creating a communication floor above 17 KB before the wrapper; or
   * their multi-exponentiation is proved.
4. A plain Fr R1CS proof of that multi-exponentiation requires about 3.2–3.4 million non-native Fq multiplications, i.e. billions of constraints at the supplied costs.
5. Random batching does not eliminate those multiplications.
6. A recursive Fq proof can hide them but exceeds the 100k Dory budget and sub-second prover budget.
7. Independently, the existing Blake2b transcript cannot be replayed in a \(2^{18}\)-constraint wrapper.

## Concrete Pareto points

### Point A — practical, larger proof

* Dory native outside;
* all GT elements sent with 128-byte codec;
* field-native transcript bridge;
* wrapper relation contains only Jolt Fr algebra.

Results:

* Dory constraints: under 2k;
* total proof: **23.4 KB at \(L=18\)**, **24.4 KB at \(L=20\)**;
* wrapper prover can plausibly remain near the target if the complete transcript fits;
* native verifier performs current Dory work.

This is the only near-term engineering option.

### Point B — slightly smaller native proof

Additionally replace the four initial fixed-\(\Gamma_{2,0}\) GT elements by G1 preimages.

Results:

* save exactly 384 bytes;
* total approximately **23.0 KB / 24.1 KB**;
* up to four extra native pairing terms;
* protocol serialization/transcript changes required.

Useful but not transformative.

### Point C — succinct bytes, slower recursive proof

Use an Fq-native proof of the Dory computation and verify it via Grumpkin inside the wrapper.

Results:

* final proof near 6 KB;
* Dory subcircuit verifier: estimated **150k–500k** constraints;
* secondary Fq prover handles 3–5 million Fq multiplication constraints;
* overall prover not sub-second.

Choose this only if proof size dominates latency.

### Point D — redesign opening/setup

Use structured generators and a formal TIPP/MIPP-style argument, preferably with an authenticated aggregate commitment.

Expected results:

* proof approximately **8–10 KB** at the favorable end;
* Fr circuit mostly scalar-only;
* constant/logarithmic native pairing verification;
* new structured SRS, assumptions, transcript and opening protocol;
* no compatibility with `dory-pcs 0.4.2` proofs.

This is the only credible route toward all size and circuit goals, but it relaxes “Dory kept” to “AFGHO/Dory commitments retained while the opening argument is replaced.”

## Recommendation

For an immediate implementation, choose Point A and accept roughly 24 KB. If single-digit kilobytes is non-negotiable, do not invest in GT advice/RLC gadgets: redesign stage 8 around a structured TIPP/MIPP-style opening and aggregate the 41 commitments before they become independently transcript-bound. The current Dory opening cannot be offloaded into the stated wrapper budget.