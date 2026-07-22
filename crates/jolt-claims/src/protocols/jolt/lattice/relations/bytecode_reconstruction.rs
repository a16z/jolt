//! Bytecode chunk virtualization: in lattice mode the committed
//! `BytecodeChunk(chunk)` polynomials are decomposed into per-lane
//! polynomials (register selectors, flags, lookup selector, pc/imm bytes),
//! so the chunk claims produced by the base `BytecodeClaimReduction` (all
//! chunks at one shared `(r_lane ‖ r_row)` point) are settled against the
//! lane polynomials by one γ-batched sumcheck — the chunk polynomials are
//! never PCS-opened. The lane polynomials are precommitted public data, so
//! their structural validity is checked offline; only the value identity is
//! spent in-protocol.
//!
//! The identity, per chunk: `BytecodeChunk(r_lane ‖ r_row) = Σ_lane
//! eq(r_lane, lane) · lane_value(r_row)`, where a lane value is a direct
//! lane-polynomial evaluation (0/1 flag lanes), a one-hot selector decode
//! (register/lookup selector lanes: the 2^k lane-eq weights of the block form
//! a k-variable multilinear), or a byte decode (pc/imm lanes:
//! `value(byte) · 256^place`, see
//! [`byte_decode_weight`](super::super::geometry::byte_decode_weight)).
//!
//! One sumcheck over the widest byte-lane `(byte ‖ place)` variables, row
//! point fixed at `r_row`; the narrower legs bind only their own suffix
//! rounds and the flag lanes none at all (mixed-count legs are precedented by the
//! lattice booleanity's msb). Every leg is at most a product of two
//! multilinears per bound variable, hence degree 2.

use jolt_field::RingCore;
use jolt_poly::math::Math;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
use serde::{Deserialize, Serialize};

use crate::protocols::jolt::geometry::claim_reductions::bytecode::{
    final_bytecode_chunk_opening, BYTECODE_LANE_LAYOUT,
};
use crate::protocols::jolt::{
    BytecodeChunkReconstructionChallenge, BytecodeChunkReconstructionPublic, BytecodeRegisterLane,
    JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltRelationId,
};
use crate::{
    challenge, derived, opening, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};

use super::super::geometry::{BYTE_BITS, WORD_BYTES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BytecodeReconstructionDimensions {
    pub chunks: usize,
    /// Byte places of one immediate lane (the field's canonical byte width).
    pub imm_byte_width: usize,
}

/// The consumed chunk claims: the base bytecode reduction's terminus, all
/// chunks at one shared point.
#[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
pub struct BytecodeChunkReconstructionInputClaims<C> {
    #[opening(committed = BytecodeChunk, from = BytecodeClaimReduction)]
    pub chunks: Vec<C>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, SumcheckChallenges)]
pub struct BytecodeChunkReconstructionChallenges<F> {
    #[challenge(BytecodeChunkReconstructionChallenge::Gamma)]
    pub gamma: F,
}

/// The lane-polynomial openings the packed opening consumes — one final
/// claim per packed slot. Families are chunk-major (`register_selectors` additionally
/// lane-minor: `[c0·rs1, c0·rs2, c0·rd, c1·rs1, …]`).
///
/// The `(chunk, lane/flag)` families are two-index, which the
/// `#[derive(OutputClaims)]` `Vec` convention (single `usize` index) cannot
/// express — the trait impl is hand-written below in the same
/// field-declaration order the derive would use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: serde::Serialize",
    deserialize = "C: serde::Deserialize<'de>"
))]
pub struct BytecodeChunkReconstructionOutputClaims<C> {
    pub register_selectors: Vec<C>,
    pub circuit_flags: Vec<C>,
    pub instruction_flags: Vec<C>,
    pub lookup_selectors: Vec<C>,
    pub raf_flags: Vec<C>,
    pub pc_bytes: Vec<C>,
    pub imm_bytes: Vec<C>,
}

impl<C> BytecodeChunkReconstructionOutputClaims<C> {
    /// Output-claim ids in field-declaration order for `chunks` bytecode
    /// chunks — the canonical order of the hand-written [`OutputClaims`]
    /// impl.
    pub fn opening_order(chunks: usize) -> Vec<JoltOpeningId> {
        let mut order = Vec::with_capacity(
            chunks
                * (BytecodeRegisterLane::ALL.len() + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTION_FLAGS + 4),
        );
        for chunk in 0..chunks {
            for lane in BytecodeRegisterLane::ALL {
                order.push(bytecode_register_selector_opening(chunk, lane));
            }
        }
        for chunk in 0..chunks {
            for flag in 0..NUM_CIRCUIT_FLAGS {
                order.push(bytecode_circuit_flag_opening(chunk, flag));
            }
        }
        for chunk in 0..chunks {
            for flag in 0..NUM_INSTRUCTION_FLAGS {
                order.push(bytecode_instruction_flag_opening(chunk, flag));
            }
        }
        order.extend((0..chunks).map(bytecode_lookup_selector_opening));
        order.extend((0..chunks).map(bytecode_raf_flag_opening));
        order.extend((0..chunks).map(bytecode_unexpanded_pc_bytes_opening));
        order.extend((0..chunks).map(bytecode_imm_bytes_opening));
        order
    }

    /// Cells chained in field-declaration order — [`Self::opening_order`]
    /// pairs one-for-one with this by construction (one lookup selector per
    /// chunk fixes the chunk count; the assert keeps a malformed family from
    /// silently shifting every later id).
    pub fn leaves(&self) -> impl Iterator<Item = (JoltOpeningId, &C)> {
        let chunks = self.lookup_selectors.len();
        debug_assert_eq!(
            self.register_selectors.len(),
            chunks * BytecodeRegisterLane::ALL.len()
        );
        debug_assert_eq!(self.circuit_flags.len(), chunks * NUM_CIRCUIT_FLAGS);
        debug_assert_eq!(self.instruction_flags.len(), chunks * NUM_INSTRUCTION_FLAGS);
        debug_assert_eq!(self.raf_flags.len(), chunks);
        debug_assert_eq!(self.pc_bytes.len(), chunks);
        debug_assert_eq!(self.imm_bytes.len(), chunks);

        let cells = self
            .register_selectors
            .iter()
            .chain(&self.circuit_flags)
            .chain(&self.instruction_flags)
            .chain(&self.lookup_selectors)
            .chain(&self.raf_flags)
            .chain(&self.pc_bytes)
            .chain(&self.imm_bytes);
        Self::opening_order(chunks).into_iter().zip(cells)
    }
}

impl<F: jolt_field::Field> OutputClaims<F> for BytecodeChunkReconstructionOutputClaims<F> {
    fn canonical_order(&self) -> Vec<JoltOpeningId> {
        self.leaves().map(|(id, _)| id).collect()
    }

    fn from_opening_values(
        mut resolve: impl FnMut(&JoltOpeningId) -> Option<F>,
    ) -> Result<Self, crate::MissingOpeningValue<JoltOpeningId>> {
        // The chunk count is instance data; probe it through the
        // one-per-chunk lookup-selector family, then size every other family
        // from it (the same invariant `leaves()` debug-asserts).
        let mut chunks = 0usize;
        while resolve(&bytecode_lookup_selector_opening(chunks)).is_some() {
            chunks += 1;
        }
        let mut get = |id: JoltOpeningId| resolve(&id).ok_or(crate::MissingOpeningValue { id });
        Ok(Self {
            register_selectors: (0..chunks)
                .flat_map(|chunk| {
                    BytecodeRegisterLane::ALL
                        .map(|lane| bytecode_register_selector_opening(chunk, lane))
                })
                .map(&mut get)
                .collect::<Result<_, _>>()?,
            circuit_flags: (0..chunks)
                .flat_map(|chunk| {
                    (0..NUM_CIRCUIT_FLAGS)
                        .map(move |flag| bytecode_circuit_flag_opening(chunk, flag))
                })
                .map(&mut get)
                .collect::<Result<_, _>>()?,
            instruction_flags: (0..chunks)
                .flat_map(|chunk| {
                    (0..NUM_INSTRUCTION_FLAGS)
                        .map(move |flag| bytecode_instruction_flag_opening(chunk, flag))
                })
                .map(&mut get)
                .collect::<Result<_, _>>()?,
            lookup_selectors: (0..chunks)
                .map(bytecode_lookup_selector_opening)
                .map(&mut get)
                .collect::<Result<_, _>>()?,
            raf_flags: (0..chunks)
                .map(bytecode_raf_flag_opening)
                .map(&mut get)
                .collect::<Result<_, _>>()?,
            pc_bytes: (0..chunks)
                .map(bytecode_unexpanded_pc_bytes_opening)
                .map(&mut get)
                .collect::<Result<_, _>>()?,
            imm_bytes: (0..chunks)
                .map(bytecode_imm_bytes_opening)
                .map(&mut get)
                .collect::<Result<_, _>>()?,
        })
    }

    /// One pass instead of the default's per-id linear re-resolution — this
    /// is the largest claim struct in the system.
    fn opening_values(&self) -> Vec<F> {
        self.leaves().map(|(_, cell)| *cell).collect()
    }

    fn resolve_output(&self, id: &JoltOpeningId) -> Option<F> {
        self.leaves()
            .find(|(candidate, _)| candidate == id)
            .map(|(_, cell)| *cell)
    }
}

pub struct BytecodeChunkReconstruction {
    shape: BytecodeReconstructionDimensions,
}

impl SymbolicSumcheck for BytecodeChunkReconstruction {
    type RelationId = JoltRelationId;
    type OpeningId = JoltOpeningId;
    type DerivedId = crate::protocols::jolt::JoltDerivedId;
    type ChallengeId = crate::protocols::jolt::JoltChallengeId;
    type Shape = BytecodeReconstructionDimensions;
    type Challenges<F> = BytecodeChunkReconstructionChallenges<F>;
    type Inputs<C> = BytecodeChunkReconstructionInputClaims<C>;
    type Outputs<C> = BytecodeChunkReconstructionOutputClaims<C>;

    fn new(shape: BytecodeReconstructionDimensions) -> Self {
        Self { shape }
    }

    fn id() -> JoltRelationId {
        JoltRelationId::BytecodeChunkReconstruction
    }

    /// The widest byte lane's `(byte ‖ place)` variable count; narrower legs join in
    /// their suffix rounds, the flag legs only at the final claim.
    fn rounds(&self) -> usize {
        BYTE_BITS + WORD_BYTES.log_2().max(self.shape.imm_byte_width.log_2())
    }

    fn degree(&self) -> usize {
        2
    }

    fn input_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(BytecodeChunkReconstructionChallenge::Gamma);
        (0..self.shape.chunks).fold(JoltExpr::zero(), |acc, chunk| {
            acc + gamma.clone().pow(chunk) * opening(final_bytecode_chunk_opening(chunk))
        })
    }

    fn output_expression<F: RingCore>(&self) -> JoltExpr<F> {
        let gamma = challenge(BytecodeChunkReconstructionChallenge::Gamma);
        let layout = BYTECODE_LANE_LAYOUT;
        let mut output = JoltExpr::<F>::zero();

        for chunk in 0..self.shape.chunks {
            let scale = gamma.clone().pow(chunk);
            for lane in BytecodeRegisterLane::ALL {
                output = output
                    + scale.clone()
                        * derived(BytecodeChunkReconstructionPublic::RegisterSelectorWeight(
                            lane,
                        ))
                        * opening(bytecode_register_selector_opening(chunk, lane));
            }
            for flag in 0..NUM_CIRCUIT_FLAGS {
                output = output
                    + scale.clone()
                        * derived(BytecodeChunkReconstructionPublic::LaneWeight(
                            layout.circuit_start + flag,
                        ))
                        * opening(bytecode_circuit_flag_opening(chunk, flag));
            }
            for flag in 0..NUM_INSTRUCTION_FLAGS {
                output = output
                    + scale.clone()
                        * derived(BytecodeChunkReconstructionPublic::LaneWeight(
                            layout.instr_start + flag,
                        ))
                        * opening(bytecode_instruction_flag_opening(chunk, flag));
            }
            output = output
                + scale.clone()
                    * derived(BytecodeChunkReconstructionPublic::LookupSelectorWeight)
                    * opening(bytecode_lookup_selector_opening(chunk))
                + scale.clone()
                    * derived(BytecodeChunkReconstructionPublic::LaneWeight(
                        layout.raf_flag_idx,
                    ))
                    * opening(bytecode_raf_flag_opening(chunk))
                + scale.clone()
                    * derived(BytecodeChunkReconstructionPublic::PcByteDecode)
                    * opening(bytecode_unexpanded_pc_bytes_opening(chunk))
                + scale
                    * derived(BytecodeChunkReconstructionPublic::ImmByteDecode)
                    * opening(bytecode_imm_bytes_opening(chunk));
        }
        output
    }
}

pub fn bytecode_register_selector_opening(
    chunk: usize,
    lane: BytecodeRegisterLane,
) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeRegisterSelector { chunk, lane },
        JoltRelationId::BytecodeChunkReconstruction,
    )
}

pub fn bytecode_circuit_flag_opening(chunk: usize, flag: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeCircuitFlag { chunk, flag },
        JoltRelationId::BytecodeChunkReconstruction,
    )
}

pub fn bytecode_instruction_flag_opening(chunk: usize, flag: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeInstructionFlag { chunk, flag },
        JoltRelationId::BytecodeChunkReconstruction,
    )
}

pub fn bytecode_lookup_selector_opening(chunk: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeLookupSelector { chunk },
        JoltRelationId::BytecodeChunkReconstruction,
    )
}

pub fn bytecode_raf_flag_opening(chunk: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeRafFlag { chunk },
        JoltRelationId::BytecodeChunkReconstruction,
    )
}

pub fn bytecode_unexpanded_pc_bytes_opening(chunk: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeUnexpandedPcBytes { chunk },
        JoltRelationId::BytecodeChunkReconstruction,
    )
}

pub fn bytecode_imm_bytes_opening(chunk: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeImmBytes { chunk },
        JoltRelationId::BytecodeChunkReconstruction,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::jolt::JoltDerivedId;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> BytecodeReconstructionDimensions {
        BytecodeReconstructionDimensions {
            chunks: 2,
            imm_byte_width: 16,
        }
    }

    fn pow(base: Fr, exponent: usize) -> Fr {
        (0..exponent).fold(Fr::from_u64(1), |acc, _| acc * base)
    }

    #[test]
    fn bytecode_reconstruction_evaluates_like_core_formula() {
        let relation = BytecodeChunkReconstruction::new(dimensions());
        let layout = BYTECODE_LANE_LAYOUT;

        let gamma = Fr::from_u64(37);
        let zero = Fr::from_u64(0);
        let chunk_claim = |chunk: usize| Fr::from_u64(3 + chunk as u64);

        let input = relation.input_expression::<Fr>().evaluate(
            |id| {
                (0..2)
                    .find(|&chunk| *id == final_bytecode_chunk_opening(chunk))
                    .map_or(zero, chunk_claim)
            },
            |_| gamma,
            |_| zero,
        );
        assert_eq!(input, chunk_claim(0) + gamma * chunk_claim(1));

        // Distinct primes per opening and per derived pin the bilinear form.
        let opening_value = |id: &JoltOpeningId| {
            BytecodeChunkReconstructionOutputClaims::<Fr>::opening_order(2)
                .iter()
                .position(|candidate| candidate == id)
                .map_or(zero, |position| Fr::from_u64(100 + position as u64))
        };
        let derived_value = |id: &JoltDerivedId| match *id {
            JoltDerivedId::BytecodeChunkReconstruction(public) => match public {
                BytecodeChunkReconstructionPublic::RegisterSelectorWeight(lane) => {
                    Fr::from_u64(1000 + lane as u64)
                }
                BytecodeChunkReconstructionPublic::LaneWeight(lane) => {
                    Fr::from_u64(2000 + lane as u64)
                }
                BytecodeChunkReconstructionPublic::LookupSelectorWeight => Fr::from_u64(3000),
                BytecodeChunkReconstructionPublic::PcByteDecode => Fr::from_u64(4000),
                BytecodeChunkReconstructionPublic::ImmByteDecode => Fr::from_u64(5000),
            },
            _ => zero,
        };

        let output =
            relation
                .output_expression::<Fr>()
                .evaluate(opening_value, |_| gamma, derived_value);

        let mut expected = Fr::from_u64(0);
        for chunk in 0..2 {
            let scale = pow(gamma, chunk);
            for lane in BytecodeRegisterLane::ALL {
                expected += scale
                    * Fr::from_u64(1000 + lane as u64)
                    * opening_value(&bytecode_register_selector_opening(chunk, lane));
            }
            for flag in 0..NUM_CIRCUIT_FLAGS {
                expected += scale
                    * Fr::from_u64(2000 + (layout.circuit_start + flag) as u64)
                    * opening_value(&bytecode_circuit_flag_opening(chunk, flag));
            }
            for flag in 0..NUM_INSTRUCTION_FLAGS {
                expected += scale
                    * Fr::from_u64(2000 + (layout.instr_start + flag) as u64)
                    * opening_value(&bytecode_instruction_flag_opening(chunk, flag));
            }
            expected += scale
                * Fr::from_u64(3000)
                * opening_value(&bytecode_lookup_selector_opening(chunk));
            expected += scale
                * Fr::from_u64(2000 + layout.raf_flag_idx as u64)
                * opening_value(&bytecode_raf_flag_opening(chunk));
            expected += scale
                * Fr::from_u64(4000)
                * opening_value(&bytecode_unexpanded_pc_bytes_opening(chunk));
            expected +=
                scale * Fr::from_u64(5000) * opening_value(&bytecode_imm_bytes_opening(chunk));
        }
        assert_eq!(output, expected);
    }

    #[test]
    fn bytecode_reconstruction_exposes_expected_dependencies() {
        let relation = BytecodeChunkReconstruction::new(dimensions());

        assert_eq!(
            BytecodeChunkReconstruction::id(),
            JoltRelationId::BytecodeChunkReconstruction
        );
        // Widest byte lane: 16-byte immediates → 8 + 4 rounds.
        assert_eq!(relation.rounds(), 8 + 4);
        assert_eq!(relation.degree(), 2);
    }

    /// The output expression's leaves and the hand-written claim struct's
    /// canonical order cover exactly the same id set (orders differ: the
    /// expression iterates chunk-major, the struct field-major).
    #[test]
    fn output_expression_covers_exactly_the_canonical_leaves() {
        let relation = BytecodeChunkReconstruction::new(dimensions());

        let from_expression = relation.expected_output_openings::<Fr>();
        let from_struct = BytecodeChunkReconstructionOutputClaims::<Fr>::opening_order(2)
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(from_expression, from_struct);
    }

    #[test]
    fn hand_written_claims_resolve_in_canonical_order() {
        let chunks = 2;
        let order = BytecodeChunkReconstructionOutputClaims::<Fr>::opening_order(chunks);

        let claims = BytecodeChunkReconstructionOutputClaims::<Fr> {
            register_selectors: (0..chunks * BytecodeRegisterLane::ALL.len())
                .map(|index| Fr::from_u64(10 + index as u64))
                .collect(),
            circuit_flags: (0..chunks * NUM_CIRCUIT_FLAGS)
                .map(|index| Fr::from_u64(1000 + index as u64))
                .collect(),
            instruction_flags: (0..chunks * NUM_INSTRUCTION_FLAGS)
                .map(|index| Fr::from_u64(3000 + index as u64))
                .collect(),
            lookup_selectors: (0..chunks).map(|c| Fr::from_u64(7000 + c as u64)).collect(),
            raf_flags: (0..chunks).map(|c| Fr::from_u64(7100 + c as u64)).collect(),
            pc_bytes: (0..chunks).map(|c| Fr::from_u64(7200 + c as u64)).collect(),
            imm_bytes: (0..chunks).map(|c| Fr::from_u64(7300 + c as u64)).collect(),
        };

        assert_eq!(OutputClaims::<Fr>::canonical_order(&claims), order);
        // Spot-check two-index resolution: chunk 1, second circuit flag.
        assert_eq!(
            OutputClaims::<Fr>::resolve_output(&claims, &bytecode_circuit_flag_opening(1, 1)),
            Some(Fr::from_u64(1000 + (NUM_CIRCUIT_FLAGS + 1) as u64))
        );
        assert_eq!(
            OutputClaims::<Fr>::resolve_output(
                &claims,
                &bytecode_register_selector_opening(1, BytecodeRegisterLane::Rd)
            ),
            Some(Fr::from_u64(10 + 5))
        );
        // Values line up with the canonical order one-for-one.
        let values = OutputClaims::<Fr>::opening_values(&claims);
        assert_eq!(values.len(), order.len());
    }
}
