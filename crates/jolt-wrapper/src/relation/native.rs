//! Assign-mode drift guard: every challenge and derived wire the relation
//! evaluates is recomputed through its native owner — the verifier's own
//! `ConcreteSumcheck` instances, rebuilt from the replayed stage outputs the
//! way each `stageN::verify` builds them — and compared with the gadget value.
//! The native challenges are re-drawn from the recorded squeezes, so a wire
//! registered under the wrong id, a gadget that drifts from its formula, or a
//! point wired to the wrong stage all surface as a `NativeMismatch`.

use std::collections::BTreeSet;

use jolt_claims::protocols::jolt::geometry::bytecode::{
    read_raf_committed_public_values, BytecodeReadRafCommittedEvaluationInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS};
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::spartan::{
    SpartanOuterDimensions, SpartanProductDimensions,
};
use jolt_claims::protocols::jolt::{
    BytecodeReadRafChallenge, BytecodeReadRafPublic, JoltAdviceKind, JoltChallengeId,
    JoltDerivedId, JoltExpr, JoltRelationId,
};
use jolt_claims::{
    InputClaims, NoChallenges, OutputClaims, Source, SumcheckChallenges, SymbolicSumcheck,
};
use jolt_field::{Fr, Zero};
use jolt_program::preprocess::PublicIoMemory;
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_verifier::stages::stage1::outputs::Stage1BatchSumchecks;
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_verifier::stages::stage2::outputs::Stage2BatchSumchecks;
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::stages::stage2::product_uniskip::ProductUniskip;
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_verifier::stages::stage3::instruction_input::InstructionInput;
use jolt_verifier::stages::stage3::outputs::Stage3Sumchecks;
use jolt_verifier::stages::stage3::registers_claim_reduction::RegistersClaimReduction;
use jolt_verifier::stages::stage3::spartan_shift::SpartanShift;
use jolt_verifier::stages::stage4::outputs::Stage4Sumchecks;
use jolt_verifier::stages::stage4::ram_val_check::RamValCheck;
use jolt_verifier::stages::stage4::registers_read_write_checking::RegistersReadWriteChecking;
use jolt_verifier::stages::stage4::{
    public_initial_ram_evaluation, ram_val_check_init_structure, stage4_input_points_from_upstream,
};
use jolt_verifier::stages::stage5::outputs::Stage5Sumchecks;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_verifier::stages::stage5::{
    stage5_input_points_from_upstream, InstructionReadRaf, RegistersValEvaluation,
};
use jolt_verifier::stages::stage6a::batch::Stage6aBuildParts;
use jolt_verifier::stages::stage6a::outputs::Stage6aSumchecks;
use jolt_verifier::stages::stage6b::batch::{Stage6bBuildParts, Stage6bDraws};
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::advice_reference_point_from_upstream;
use jolt_verifier::stages::stage6b::outputs::Stage6bSumchecks;
use jolt_verifier::stages::stage6b::stage6b_input_points_from_upstream;
use jolt_verifier::stages::stage7::build_stage7_sumchecks;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::hamming_weight_claim_reduction_dimensions;
use jolt_verifier::stages::uniskip::{draw_spartan_outer_tau, draw_spartan_product_tau_high};

use super::ctx::{Ctx, Lc};
use super::replay::NativeReplay;
use super::wiring::Wires;
use super::{Preprocessing, Proof, RelationError};

/// How many distinct ids the guard compared against their native owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeParity {
    pub derived: usize,
    pub challenges: usize,
}

/// Replays one stage's recorded squeezes from the point where the native
/// verifier began drawing; appends are ignored, so the native draw functions
/// reproduce exactly the values the recording saw.
#[derive(Default)]
struct Squeezes {
    values: Vec<Fr>,
    position: usize,
}

impl Squeezes {
    fn stage(replay: &NativeReplay, stage: usize) -> Self {
        Self {
            values: replay.squeezes[replay.stage_squeezes[stage]..].to_vec(),
            position: 0,
        }
    }
}

impl Transcript for Squeezes {
    type Challenge = Fr;

    fn new(_label: &'static [u8]) -> Self {
        Self::default()
    }

    fn append_bytes(&mut self, _bytes: &[u8]) {}

    fn challenge(&mut self) -> Fr {
        let value = self
            .values
            .get(self.position)
            .copied()
            .unwrap_or_else(Fr::zero);
        self.position += 1;
        value
    }

    fn state(&self) -> [u8; 32] {
        [0; 32]
    }
}

fn sources(expr: &JoltExpr<Fr>) -> (BTreeSet<JoltChallengeId>, BTreeSet<JoltDerivedId>) {
    let mut challenges = BTreeSet::new();
    let mut deriveds = BTreeSet::new();
    for factor in expr.terms.iter().flat_map(|term| term.factors.iter()) {
        match factor {
            Source::Challenge(id) => {
                let _ = challenges.insert(*id);
            }
            Source::Derived(id) => {
                let _ = deriveds.insert(*id);
            }
            Source::Opening(_) => {}
        }
    }
    (challenges, deriveds)
}

struct Checker<'a> {
    ctx: &'a Ctx,
    wires: &'a Wires,
    derived: BTreeSet<JoltDerivedId>,
    challenges: BTreeSet<JoltChallengeId>,
}

impl Checker<'_> {
    fn compare(
        kind: &'static str,
        id: impl std::fmt::Debug,
        gadget: Option<Fr>,
        native: Fr,
    ) -> Result<(), RelationError> {
        let gadget = gadget.ok_or_else(|| RelationError::MissingSource {
            kind,
            id: format!("{id:?}"),
        })?;
        if gadget != native {
            return Err(RelationError::NativeMismatch {
                kind,
                id: format!("{id:?}"),
                gadget,
                native,
            });
        }
        Ok(())
    }

    fn value(&self, lc: Option<&Lc>) -> Option<Fr> {
        lc.and_then(|lc| self.ctx.value(lc))
    }

    fn derived(&mut self, id: JoltDerivedId, native: Fr) -> Result<(), RelationError> {
        let gadget = self.value(self.wires.sources.deriveds.get(&id));
        Self::compare("derived", id, gadget, native)?;
        let _ = self.derived.insert(id);
        Ok(())
    }

    fn challenge(&mut self, id: JoltChallengeId, native: Fr) -> Result<(), RelationError> {
        let gadget = self.value(self.wires.sources.challenges.get(&id));
        Self::compare("challenge", id, gadget, native)?;
        let _ = self.challenges.insert(id);
        Ok(())
    }

    fn member<S>(
        &mut self,
        member: &S,
        input_points: &SumcheckInputPoints<Fr, S>,
        output_points: &SumcheckOutputPoints<Fr, S>,
        challenges: &ConcreteSumcheckChallenges<Fr, S>,
    ) -> Result<(), RelationError>
    where
        S: ConcreteSumcheck<Fr>,
        SumcheckInputClaims<Fr, S>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, S>: OutputClaims<Fr>,
        ConcreteSumcheckChallenges<Fr, S>: SumcheckChallenges<Fr>,
    {
        let symbolic = member.symbolic();
        let (input_challenges, input_deriveds) = sources(&symbolic.input_expression::<Fr>());
        let (output_challenges, output_deriveds) = sources(&symbolic.output_expression::<Fr>());
        for id in input_challenges.union(&output_challenges) {
            let native =
                challenges
                    .resolve_challenge(id)
                    .ok_or_else(|| RelationError::MissingSource {
                        kind: "native challenge",
                        id: format!("{id:?}"),
                    })?;
            self.challenge(*id, native)?;
        }
        for id in input_deriveds {
            let native = member.derive_input_term(&id, challenges)?;
            self.derived(id, native)?;
        }
        for id in output_deriveds {
            let native = member.derive_output_term(&id, input_points, output_points, challenges)?;
            self.derived(id, native)?;
        }
        Ok(())
    }

    /// A uni-skip relation: verified outside the batch, so only its input
    /// expression's sources have native owners here.
    fn input_terms<S>(
        &mut self,
        member: &S,
        challenges: &ConcreteSumcheckChallenges<Fr, S>,
    ) -> Result<(), RelationError>
    where
        S: ConcreteSumcheck<Fr>,
        SumcheckInputClaims<Fr, S>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, S>: OutputClaims<Fr>,
        ConcreteSumcheckChallenges<Fr, S>: SumcheckChallenges<Fr>,
    {
        let (input_challenges, input_deriveds) =
            sources(&member.symbolic().input_expression::<Fr>());
        for id in input_challenges {
            let native =
                challenges
                    .resolve_challenge(&id)
                    .ok_or_else(|| RelationError::MissingSource {
                        kind: "native challenge",
                        id: format!("{id:?}"),
                    })?;
            self.challenge(id, native)?;
        }
        for id in input_deriveds {
            let native = member.derive_input_term(&id, challenges)?;
            self.derived(id, native)?;
        }
        Ok(())
    }

    /// An optional batch member: present in all four projections or in none.
    fn optional<S>(
        &mut self,
        member: Option<&S>,
        input_points: Option<&SumcheckInputPoints<Fr, S>>,
        output_points: Option<&SumcheckOutputPoints<Fr, S>>,
        challenges: Option<&ConcreteSumcheckChallenges<Fr, S>>,
    ) -> Result<(), RelationError>
    where
        S: ConcreteSumcheck<Fr>,
        SumcheckInputClaims<Fr, S>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, S>: OutputClaims<Fr>,
        ConcreteSumcheckChallenges<Fr, S>: SumcheckChallenges<Fr>,
    {
        match (member, input_points, output_points, challenges) {
            (Some(member), Some(input), Some(output), Some(challenges)) => {
                self.member(member, input, output, challenges)
            }
            (None, None, None, None) => Ok(()),
            _ => Err(RelationError::Geometry(format!(
                "optional member {:?} present in only some native projections",
                S::Symbolic::id()
            ))),
        }
    }
}

macro_rules! members {
    ($checker:expr, $sumchecks:expr, $inputs:expr, $outputs:expr, $challenges:expr; $($member:ident),* $(; $($optional:ident),*)?) => {
        $($checker.member(&$sumchecks.$member, &$inputs.$member, &$outputs.$member, &$challenges.$member)?;)*
        $($($checker.optional(
            $sumchecks.$optional.as_ref(),
            $inputs.$optional.as_ref(),
            $outputs.$optional.as_ref(),
            $challenges.$optional.as_ref(),
        )?;)*)?
    };
}

fn geometry(error: impl std::fmt::Display) -> RelationError {
    RelationError::Geometry(error.to_string())
}

/// Rebuilds every stage's native sumchecks and compares each challenge and
/// derived wire the relation uses with the native owner's value; every
/// registered wire must be covered by some owner.
pub(crate) fn check(
    ctx: &Ctx,
    wires: &Wires,
    preprocessing: &Preprocessing,
    proof: &Proof,
    replay: &NativeReplay,
) -> Result<NativeParity, RelationError> {
    let checked = &replay.checked;
    let formula_dimensions = &replay.formula_dimensions;
    let log_t = checked.trace_length.ilog2() as usize;
    let log_k = checked.ram_K.ilog2() as usize;
    let trace = TraceDimensions::new(log_t);
    let mut checker = Checker {
        ctx,
        wires,
        derived: BTreeSet::new(),
        challenges: BTreeSet::new(),
    };

    let stage1 = replay.stage1.clear()?;
    let stage2 = replay.stage2.clear()?;
    let stage3 = replay.stage3.clear()?;
    let stage4 = replay.stage4.clear()?;
    let stage5 = replay.stage5.clear()?;
    let stage6a = replay.stage6a.clear()?;
    let stage6b = replay.stage6b.clear()?;
    let stage7 = replay.stage7.clear()?;

    // Stage 1: tau, the uni-skip challenge, then the batch draws.
    let mut transcript = Squeezes::stage(replay, 0);
    let tau = draw_spartan_outer_tau(&mut transcript, log_t);
    let uniskip_challenge = transcript.challenge();
    let sumchecks = Stage1BatchSumchecks {
        outer_remainder: OuterRemainder::new(
            SpartanOuterDimensions::rv64(log_t),
            tau,
            uniskip_challenge,
        ),
    };
    let challenges = sumchecks.draw_challenges(&mut transcript)?;
    let inputs = sumchecks.empty_input_points();
    // Binds the remainder point the outer relation evaluates its output terms at.
    let points = sumchecks.derive_opening_points(&stage1.remainder_point(), &inputs)?;
    if points != stage1.output_points {
        return Err(RelationError::Geometry(
            "stage-1 opening points do not re-derive from the remainder point".to_string(),
        ));
    }
    members!(checker, sumchecks, inputs, points, challenges; outer_remainder);

    // Stage 2: tau_high, the product uni-skip challenge, then the batch draws.
    let mut transcript = Squeezes::stage(replay, 1);
    let tau_high = draw_spartan_product_tau_high(&mut transcript);
    let uniskip_challenge = transcript.challenge();
    let product_dimensions = SpartanProductDimensions::new(log_t);
    checker.input_terms(
        &ProductUniskip::new(product_dimensions, tau_high),
        &NoChallenges::default(),
    )?;
    let tau_low = stage2.product_tau_low.clone();
    let read_write = proof.rw_config.ram_dimensions(log_t, log_k);
    let raf = RamRafEvaluationDimensions::try_from(read_write).map_err(geometry)?;
    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(geometry)?;
    let sumchecks = Stage2BatchSumchecks {
        ram_read_write: RamReadWriteChecking::new(read_write, log_k, tau_low.clone()),
        product_remainder: ProductRemainder::new(
            product_dimensions,
            uniskip_challenge,
            tau_high,
            tau_low.clone(),
        ),
        instruction_claim_reduction: InstructionClaimReduction::new(trace, tau_low.clone()),
        ram_raf_evaluation: RamRafEvaluation::new(
            read_write,
            raf,
            log_k,
            checked.public_io.memory_layout.get_lowest_address(),
            tau_low.clone(),
        ),
        ram_output_check: RamOutputCheck::new(read_write, public_memory),
    };
    let challenges = sumchecks.draw_challenges(&mut transcript)?;
    let inputs = sumchecks.empty_input_points();
    members!(
        checker, sumchecks, inputs, stage2.output_points, challenges;
        ram_read_write, product_remainder, instruction_claim_reduction, ram_raf_evaluation,
        ram_output_check
    );

    // Stage 3.
    let mut transcript = Squeezes::stage(replay, 2);
    let product_remainder_point = stage2.output_points.product_remainder_point().to_vec();
    let sumchecks = Stage3Sumchecks {
        shift: SpartanShift::new(trace, tau_low.clone(), product_remainder_point.clone()),
        instruction_input: InstructionInput::new(trace, product_remainder_point),
        registers_claim_reduction: RegistersClaimReduction::new(trace, tau_low),
    };
    let challenges = sumchecks.draw_challenges(&mut transcript)?;
    let inputs = sumchecks.empty_input_points();
    members!(
        checker, sumchecks, inputs, stage3.output_points, challenges;
        shift, instruction_input, registers_claim_reduction
    );

    // Stage 4.
    let mut transcript = Squeezes::stage(replay, 3);
    let r_address = &stage2.output_points.ram_read_write_point()[..log_k];
    let public_eval = public_initial_ram_evaluation(checked, preprocessing, r_address)?;
    let init_structure = ram_val_check_init_structure(
        checked,
        proof.untrusted_advice_commitment.is_some(),
        r_address,
        public_eval,
    )?;
    let sumchecks = Stage4Sumchecks {
        registers_read_write: RegistersReadWriteChecking::new(
            proof
                .rw_config
                .register_dimensions(log_t, REGISTER_ADDRESS_BITS),
        ),
        ram_val_check: RamValCheck::new(trace, log_k, init_structure.decomposition()),
    };
    let challenges = sumchecks.draw_challenges(&mut transcript)?;
    let inputs = stage4_input_points_from_upstream(
        &stage2.output_points,
        &stage3.output_points,
        &init_structure,
    );
    members!(
        checker, sumchecks, inputs, stage4.output_points, challenges;
        registers_read_write, ram_val_check
    );

    // Stage 5.
    let mut transcript = Squeezes::stage(replay, 4);
    let sumchecks = Stage5Sumchecks {
        instruction_read_raf: InstructionReadRaf::new(formula_dimensions.instruction_read_raf),
        ram_ra_claim_reduction: RamRaClaimReduction::new(formula_dimensions.trace, log_k),
        registers_val_evaluation: RegistersValEvaluation::new(formula_dimensions.trace),
    };
    let challenges = sumchecks.draw_challenges(&mut transcript)?;
    let inputs = stage5_input_points_from_upstream(&stage2.output_points, &stage4.output_points);
    members!(
        checker, sumchecks, inputs, stage5.output_points, challenges;
        instruction_read_raf, ram_ra_claim_reduction, registers_val_evaluation
    );

    // Stage 6a.
    let mut transcript = Squeezes::stage(replay, 5);
    let committed_program = checked.precommitted.bytecode.is_some();
    let committed_chunk_bits = proof.one_hot_config.committed_chunk_bits();
    let stage1_cycle_binding = replay
        .stage1
        .cycle_binding_checked(JoltRelationId::BytecodeReadRaf)?;
    let entry_bytecode_index = preprocessing
        .program
        .entry_bytecode_index_checked(JoltRelationId::BytecodeReadRaf)?;
    let sumchecks = Stage6aSumchecks::build_from_parts(Stage6aBuildParts {
        formula_dimensions,
        committed_chunk_bits,
        committed_program,
        entry_bytecode_index,
        stage1_cycle_binding: &stage1_cycle_binding,
        stage2_points: &stage2.output_points,
        stage3_points: &stage3.output_points,
        stage4_points: &stage4.output_points,
        stage5_points: &stage5.output_points,
    })?;
    let challenges = sumchecks.draw_challenges(&mut transcript)?;
    let inputs = sumchecks.empty_input_points();
    members!(
        checker, sumchecks, inputs, stage6a.output_points, challenges;
        bytecode_read_raf, booleanity
    );

    // Stage 6b: the stage-level draws precede the batch; member challenges are
    // carried from 6a or taken from those draws.
    let mut transcript = Squeezes::stage(replay, 6);
    let draws = Stage6bDraws::draw(&mut transcript, committed_program);
    let bytecode_table_rows = if committed_program {
        None
    } else {
        Some(
            preprocessing
                .program
                .as_full()
                .ok_or(RelationError::Unsupported("full bytecode table"))?
                .bytecode
                .bytecode
                .as_slice(),
        )
    };
    let sumchecks = Stage6bSumchecks::build_from_parts(Stage6bBuildParts {
        formula_dimensions,
        ram_log_k: log_k,
        committed_chunk_bits,
        precommitted: &checked.precommitted,
        entry_bytecode_index,
        bytecode_table_rows,
        carried: &stage6a.challenges,
        eta: draws.eta,
        stage1_cycle_binding,
        stage2_points: &stage2.output_points,
        stage3_points: &stage3.output_points,
        stage4_points: &stage4.output_points,
        stage5_points: &stage5.output_points,
        stage6a_points: &stage6a.output_points,
        address_val_stages: stage6a.output_values.bytecode_read_raf.val_stages.clone(),
        trusted_advice_reference_point: advice_reference_point_from_upstream(
            &stage4.ram_val_check_init,
            JoltAdviceKind::Trusted,
        ),
        untrusted_advice_reference_point: advice_reference_point_from_upstream(
            &stage4.ram_val_check_init,
            JoltAdviceKind::Untrusted,
        ),
    })?;
    let challenges = sumchecks.cycle_challenges(&stage6a.challenges, &draws);
    let inputs = stage6b_input_points_from_upstream(
        &sumchecks,
        &stage2.output_points,
        &stage4.output_points,
        &stage5.output_points,
    );
    members!(
        checker, sumchecks, inputs, stage6b.output_points, challenges;
        booleanity, ram_hamming_booleanity, ram_ra_virtualization,
        instruction_ra_virtualization, inc_claim_reduction;
        trusted_advice, untrusted_advice, bytecode_reduction, program_image_reduction
    );
    // The bytecode owner folds its publics inside `expected_output` rather than
    // exposing them per id; the public functions it calls give the same values.
    let bytecode = &sumchecks.bytecode_read_raf;
    checker.challenge(
        BytecodeReadRafChallenge::Gamma.into(),
        challenges.bytecode_read_raf.gamma,
    )?;
    let opening_point = stage6b
        .output_points
        .bytecode_read_raf
        .bytecode_ra
        .first()
        .ok_or(RelationError::Unsupported(
            "bytecode cycle produced no openings",
        ))?;
    let r_cycle = opening_point
        .get(opening_point.len().saturating_sub(log_t)..)
        .ok_or(RelationError::Unsupported("bytecode cycle opening point"))?;
    let publics = read_raf_committed_public_values(BytecodeReadRafCommittedEvaluationInputs {
        r_address: bytecode.r_address(),
        r_cycle,
        stage_cycle_points: bytecode.stage_cycle_points().each_ref().map(Vec::as_slice),
        entry_bytecode_index: bytecode.entry_bytecode_index(),
    });
    let stage_values = bytecode.stage_values_at_r_address()?;
    for (stage, (value, cycle_eq)) in stage_values
        .iter()
        .zip(&publics.stage_cycle_eqs)
        .enumerate()
    {
        checker.derived(BytecodeReadRafPublic::StageCycleEq(stage).into(), *cycle_eq)?;
        checker.derived(
            BytecodeReadRafPublic::StageValue(stage).into(),
            *value * *cycle_eq,
        )?;
    }
    checker.derived(
        BytecodeReadRafPublic::SpartanOuterRaf.into(),
        publics.spartan_outer_raf,
    )?;
    checker.derived(
        BytecodeReadRafPublic::SpartanShiftRaf.into(),
        publics.spartan_shift_raf,
    )?;
    checker.derived(BytecodeReadRafPublic::Entry.into(), publics.entry)?;

    // Stage 7.
    let mut transcript = Squeezes::stage(replay, 7);
    let hamming_dimensions = hamming_weight_claim_reduction_dimensions(
        formula_dimensions.ra_layout,
        committed_chunk_bits,
    )?;
    let sumchecks = build_stage7_sumchecks(
        hamming_dimensions,
        &checked.precommitted,
        &stage6b.output_points,
        Some((stage4, stage6b)),
    )?;
    let challenges = sumchecks.draw_challenges(&mut transcript)?;
    let inputs = sumchecks.empty_input_points();
    members!(
        checker, sumchecks, inputs, stage7.output_points, challenges;
        hamming_weight_claim_reduction;
        trusted_advice, untrusted_advice, bytecode_address_phase, program_image_address_phase
    );

    if let Some(id) = wires
        .sources
        .deriveds
        .keys()
        .find(|id| !checker.derived.contains(id))
    {
        return Err(RelationError::MissingSource {
            kind: "native owner of derived",
            id: format!("{id:?}"),
        });
    }
    if let Some(id) = wires
        .sources
        .challenges
        .keys()
        .find(|id| !checker.challenges.contains(id))
    {
        return Err(RelationError::MissingSource {
            kind: "native owner of challenge",
            id: format!("{id:?}"),
        });
    }
    Ok(NativeParity {
        derived: checker.derived.len(),
        challenges: checker.challenges.len(),
    })
}
