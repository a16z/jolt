# Akita committed programs with native field-inline instructions

With `akita,field-inline`, the verifier checks a program's field-register accesses from the trusted bytecode commitment. For `FIELD_MUL f3, f1, f2`, the arithmetic constraint checks the product. The committed bytecode identifies the instruction's two sources and destination at the trace PC.

Use `preprocess_committed` with the prover's `ProverConfig`, or `preprocess_committed_with_advice` with the two advice-presence flags. The verifier receives trusted program metadata, the bytecode and image commitments, the verifier setup, public inputs and any trusted advice commitment. The prover retains the rows and opening hints. Dory field-inline requires full program preprocessing.

## Trusted row encoding

After the RAF flag, append three 16-lane blocks in order: field Rd, Rs1 and Rs2. Set one lane per present operand, including f0. An absent operand contributes sixteen zeros. The encoder uses `row.integer_operands()` and `row.field_operands()` for their respective register files. PC, immediate, flags and lookup selection come from the original row.

| Decoded row | Integer selectors | Field selectors |
|---|---|---|
| `FIELD_MUL f3, f1, f2` | None | Rd 3, Rs1 1, Rs2 2 |
| Field assert-equal | None | Rs1 and Rs2, no Rd |
| Field load immediate | None | Rd only |
| Accumulate from register | Rs1 names the integer source | Rd and Rs1 both name the field accumulator |
| Accumulate from memory | Rs1 names the base, Rd the integer scratch destination | Encoded rs2 supplies both field Rd and implicit field Rs1 |
| Advice limb | Rd names the integer limb destination | Encoded rs1 supplies the field read, encoded rs2 the field write |
| Ordinary instruction or padding | Existing integer projection | All three blocks zero |

Akita checks the rows again at the committed-field constructor, including prebuilt program inputs. Preprocessing checks the instruction profile, field operand shapes and store/Rd disjointness. The grid encoder checks the projected integer and field indices against their selector block widths before writing coefficients. The entry index must select a real instruction at the declared address and the padded size must match the rows and be a power of two. Legal encoding is a property of trusted preprocessing. The proof binds execution to the resulting commitment.

## Stage 6

At bytecode address `a`, let `D(a,k)`, `S1(a,k)` and `S2(a,k)` be the field selector evaluations for register `k`. Split the field read/write and value-evaluation points into register and cycle coordinates `(u4,t4)` and `(u5,t5)`. The stage-4 and stage-5 batching challenges are `g4` and `g5`. Stage 6a keeps the five ordinary values and the Akita store selector at wire 5, then appends two address-only evaluations.

```text
F4(a) = Σ_k eq(u4,k) · (g4³ D(a,k) + g4⁴ S1(a,k) + g4⁵ S2(a,k))
F5(a) = Σ_k eq(u5,k) · g5^(2 + lookup_table_count) · D(a,k)
```

Wire 6 carries F4 and wire 7 carries F5. Both wires are zero when the static bytecode has no field-register operands. The address fold includes rows the execution did not visit. At cycle point `t`, stage 6b adds the following to the committed read-RAF output, using the existing outer challenge `G` and product of bytecode address-chunk openings `RA`.

```text
RA · (G³ · eq(t,t4) · F4(a) + G⁴ · eq(t,t5) · F5(a))
```

The bytecode claim reduction draws its existing challenge η after the stage-6a values are fixed. It adds η⁶ F4(a) + η⁷ F5(a) to its input and incorporates the corresponding selector coefficients into the public lane weights. The dot product of each row and those weights equals the η-weighted sum of that row's eight staged values. The final opening of `ProgramBytecode` authenticates this sum. The cycle equalities enter only the read-RAF term. [Field-register lane weights](../crates/jolt-claims/src/protocols/composed/geometry/committed_bytecode.rs)

## Akita packing and identity

For N padded bytecode rows the field profile uses 1,024 lanes per row and logical arity 10 + log₂ N. Ordinary committed Akita uses 512 lanes. The extra variable doubles the coefficient count and adds one active bytecode reduction round, including for ordinary-only guests. Akita uses CycleMajor and zero-prefix embedding, with a physical arity floor of 14 and a cap of 34. The structural ceiling is 2²⁴ rows for this profile and 2²⁵ for ordinary committed Akita. Memory and catalog coverage may impose lower limits.

Setup, stage-0 hint assembly and stage-8 opening use the same group order. The Akita adapter checks strictly ascending auxiliary sort keys and passes the groups to the batch prover and verifier by position. The main group follows the auxiliaries.

| Present group | Auxiliary sort key | Source |
|---|---:|---|
| Untrusted advice, if present | 0 | Bounded advice object |
| Trusted advice, if present | 1 | Bounded advice object |
| `FieldInc` | 2 | Full-width field-register delta, reduced in stage 6b |
| `ProgramBytecode` | 3 | Bounded direct object from trusted preprocessing |
| `ProgramImageInit` | 4 | Bounded direct object from trusted preprocessing |
| `FusedInc` | 5 | Bounded execution delta, consumed by read-RAF |
| `OneHotTrace` | Main group, last | Canonical one-hot trace columns |

The adapter resolves the final trace shape in the trusted catalog. Its combined schedule admits one full-width field increment, three bounded program/image/fused producers, and up to two leading bounded advice producers. The verifier validates the selected row and binds the opening points, commitments and roles.

Full program artifacts use tag 0. Ordinary committed artifacts use tag 1. Akita committed-field artifacts use tag 2 and reject tag 1 on decode. Both Akita field-inline views hash the tagged serialization under `jolt/program-preprocessing/akita-whole-bytecode/field-inline/v3`. The bytecode packing label `program-bytecode-field-selectors-v1` distinguishes layouts even when the physical arities match.

## Verification and measurements

The committed guest matrix exercises field arithmetic, limb bridges and ordinary RV execution, including advice and independent transport of the verifier setup and proof. [Acceptance matrix](../crates/jolt-prover/tests/e2e_matrix.rs)

The November 19, 2026 measurements used a development build based on `c97e93004`, using the optimized CPU backend, K=16, CycleMajor, single-chunk (1, 0) and no advice. The first experiment used the same 4,096-cycle arithmetic execution and varied static NoOp padding. RSS is the process high-water mark through preprocessing. Verification starts after artifact decoding.

| Program view | Static rows | Preprocess, ms | Peak RSS through preprocess, MiB | Stage 6a / 6b, ms | Stage 8, ms | Verify, ms | Proof, bytes | Verifier artifact, bytes |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Full | 4,096 | 199 | 130 | 3 / 14 | 240 | 7 | 98,658 | 6,783,435 |
| Committed | 4,096 | 264 | 309 | 3 / 41 | 447 | 72 | 106,232 | 6,688,959 |
| Full | 16,384 | 194 | 129 | 9 / 14 | 215 | 7 | 98,774 | 6,894,027 |
| Committed | 16,384 | 284 | 743 | 8 / 110 | 724 | 74 | 119,007 | 6,688,975 |

The second experiment varied the input length of a single 8,192-row arithmetic program. Each guest checked its native field result against an independently calculated residue. Every serialized proof verified.

| Input pairs | Input bytes | Executed cycles | Padded cycles | Field events | Preprocess, ms | Prove, ms | Verify, ms | Proof, bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 99 | 1,937 | 4,096 | 83 | 263 | 757 | 72 | 113,718 |
| 64 | 1,239 | 14,302 | 16,384 | 683 | 260 | 858 | 71 | 114,611 |
| 256 | 4,875 | 53,696 | 65,536 | 2,603 | 263 | 990 | 73 | 104,151 |
| 1,024 | 19,493 | 212,259 | 262,144 | 10,283 | 259 | 1,432 | 73 | 105,150 |
| 4,096 | 77,810 | 844,492 | 1,048,576 | 41,003 | 34,247 | 3,287 | 203 | 108,448 |
