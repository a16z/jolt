use jolt_claims::protocols::jolt::{
    formulas::{
        booleanity::{self, BooleanityDimensions},
        bytecode::{self, BytecodeReadRafDimensions},
        claim_reductions::{
            advice,
            bytecode::{self as bytecode_reduction, BytecodeLaneWeightInputs},
            increments, program_image,
        },
        dimensions::{committed_address_chunks, JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
        instruction::{self, InstructionRaVirtualizationDimensions},
        ram,
    },
    AdviceClaimReductionLayout, BytecodeClaimReductionChallenge, BytecodeClaimReductionLayout,
    BytecodeReadRafChallenge, JoltAdviceKind, JoltChallengeId, JoltPublicId, JoltRelationClaims,
    JoltRelationId, PrecommittedClaimReduction, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
use jolt_openings::CommitmentScheme;
use jolt_riscv::NUM_CIRCUIT_FLAGS;
use jolt_sumcheck::{
    BatchedCommittedSumcheckConsistency, BatchedSumcheckVerifier, SumcheckClaim, SumcheckStatement,
};
use jolt_transcript::Transcript;
use num_traits::{One, Zero};

use super::{
    batch::{Stage6Relations, Stage6RelationsParams},
    booleanity::{
        BooleanityAddressPhase, BooleanityAddressPhaseInputClaims,
        BooleanityAddressPhaseOutputClaims,
    },
    bytecode_read_raf::{
        BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseInputClaims,
        BytecodeReadRafAddressPhaseOutputClaims,
    },
    committed_reduction_cycle_phase::{
        AdviceCyclePhase, AdviceCyclePhaseInputClaims, AdviceCyclePhaseOutputClaims,
        BytecodeReductionCyclePhase, BytecodeReductionCyclePhaseInputClaims,
        BytecodeReductionCyclePhaseOutputClaims, ProgramImageReductionCyclePhase,
        ProgramImageReductionCyclePhaseInputClaims, ProgramImageReductionCyclePhaseOutputClaims,
    },
    outputs::{
        AdviceCyclePhaseOutputClaim, BooleanityOutputClaims, BytecodeCyclePhaseOutputClaims,
        BytecodeReadRafOutputClaims, BytecodeReductionWeights, IncClaimReductionOutputClaims,
        InstructionRaVirtualizationOutputClaims, ProgramImageCyclePhaseOutputClaim,
        RamHammingBooleanityOutputClaims, RamRaVirtualizationOutputClaims,
        Stage6AddressPhaseClaims, Stage6AdviceCyclePhaseClaims, Stage6ClearOutput, Stage6Output,
        Stage6OutputClaims, Stage6PublicOutput, Stage6ZkOutput,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::{
            check_relation_boolean_hypercube, zip_openings, ConcreteSumcheck, OpeningClaim,
            OutputClaims,
        },
        stage1::{Stage1ClearOutput, Stage1Output},
        stage2::{Stage2ClearOutput, Stage2Output},
        stage3::{Stage3ClearOutput, Stage3Output},
        stage4::{Stage4ClearOutput, Stage4Output},
        stage5::{Stage5ClearOutput, Stage5Output},
        zk::{committed, outputs::CommittedOutputClaimOutput},
    },
    verifier::CheckedInputs,
    VerifierError,
};

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 6 consumes all five prior clear-stage outputs directly; bundling them would reintroduce the removed `Deps` indirection."
)]
pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage1: &Stage1Output<PCS::Field, VC::Output>,
    stage2: &Stage2Output<PCS::Field, VC::Output>,
    stage3: &Stage3Output<PCS::Field, VC::Output>,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
    stage5: &Stage5Output<PCS::Field, VC::Output>,
) -> Result<Stage6Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let log_t = formula_dimensions.trace.log_t();
    let log_k = checked.ram_K.ilog2() as usize;
    let trace_dimensions = formula_dimensions.trace;

    let bytecode_reduction_layout = checked.precommitted.bytecode.as_ref();
    let program_image_reduction_layout = checked.precommitted.program_image.as_ref();
    let committed_program = bytecode_reduction_layout.is_some();

    let bytecode_address_claims =
        bytecode::read_raf_address_phase::<PCS::Field>(formula_dimensions.bytecode_read_raf);
    let bytecode_claims = if committed_program {
        bytecode::read_raf_cycle_phase_committed::<PCS::Field>(formula_dimensions.bytecode_read_raf)
    } else {
        bytecode::read_raf_cycle_phase::<PCS::Field>(formula_dimensions.bytecode_read_raf)
    };
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_address_claims =
        booleanity::booleanity_address_phase::<PCS::Field>(booleanity_dimensions);
    let booleanity_claims = booleanity::booleanity_cycle_phase::<PCS::Field>(booleanity_dimensions);
    let ram_hamming_claims = ram::hamming_booleanity::<PCS::Field>(trace_dimensions);
    let ram_ra_claims =
        ram::ra_virtualization::<PCS::Field>(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_claims = instruction::ra_virtualization::<PCS::Field>(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_claims = increments::claim_reduction::<PCS::Field>(trace_dimensions);

    let trusted_advice_layout = checked.precommitted.trusted_advice.as_ref();
    let untrusted_advice_layout = checked.precommitted.untrusted_advice.as_ref();
    let trusted_advice_claims = trusted_advice_layout.map(|layout| {
        advice::cycle_phase::<PCS::Field>(JoltAdviceKind::Trusted, layout.dimensions())
    });
    let untrusted_advice_claims = untrusted_advice_layout.map(|layout| {
        advice::cycle_phase::<PCS::Field>(JoltAdviceKind::Untrusted, layout.dimensions())
    });
    let bytecode_reduction_claims = bytecode_reduction_layout.map(|layout| {
        bytecode_reduction::cycle_phase::<PCS::Field>(layout.dimensions(), layout.chunk_count())
    });
    let program_image_reduction_claims = program_image_reduction_layout
        .map(|layout| program_image::cycle_phase::<PCS::Field>(layout.dimensions()));

    for claim in [
        &bytecode_address_claims,
        &bytecode_claims,
        &booleanity_address_claims,
        &booleanity_claims,
        &ram_hamming_claims,
        &ram_ra_claims,
        &instruction_ra_claims,
        &inc_claims,
    ] {
        check_relation_boolean_hypercube(claim)?;
    }
    for claim in [
        &trusted_advice_claims,
        &untrusted_advice_claims,
        &bytecode_reduction_claims,
        &program_image_reduction_claims,
    ]
    .into_iter()
    .flatten()
    {
        check_relation_boolean_hypercube(claim)?;
    }

    let bytecode_gamma_powers = transcript.challenge_scalar_powers(8);
    let bytecode_gamma = bytecode_gamma_powers[1];
    let stage1_gammas = transcript.challenge_scalar_powers(2 + NUM_CIRCUIT_FLAGS);
    let stage2_gammas = transcript.challenge_scalar_powers(4);
    let stage3_gammas = transcript.challenge_scalar_powers(9);
    let stage4_gammas = transcript.challenge_scalar_powers(3);
    let stage5_gammas =
        transcript.challenge_scalar_powers(2 + LookupTableKind::<RISCV_XLEN>::COUNT);

    let (stage5_instruction_address, stage5_instruction_cycle) = match stage5 {
        Stage5Output::Clear(stage5) => (
            stage5.instruction_r_address.as_slice(),
            stage5.instruction_r_cycle(),
        ),
        Stage5Output::Zk(stage5) => (
            stage5.instruction_r_address.as_slice(),
            stage5.output_points.instruction_r_cycle(),
        ),
    };
    let mut booleanity_reference_address = stage5_instruction_address.to_vec();
    booleanity_reference_address.reverse();
    if booleanity_reference_address.len() < proof.one_hot_config.committed_chunk_bits() {
        let missing =
            proof.one_hot_config.committed_chunk_bits() - booleanity_reference_address.len();
        booleanity_reference_address.extend(transcript.challenge_vector(missing));
    } else {
        booleanity_reference_address = booleanity_reference_address
            [booleanity_reference_address.len() - proof.one_hot_config.committed_chunk_bits()..]
            .to_vec();
    }
    let mut booleanity_reference_cycle = stage5_instruction_cycle.to_vec();
    booleanity_reference_cycle.reverse();
    let mut booleanity_gamma = transcript.challenge();
    if booleanity_gamma.is_zero() {
        booleanity_gamma = PCS::Field::one();
    }

    let public = |instruction_ra_gamma_powers: Vec<PCS::Field>,
                  inc_gamma: PCS::Field,
                  eta: Option<PCS::Field>| Stage6PublicOutput {
        bytecode_gamma_powers: bytecode_gamma_powers.clone(),
        stage1_gammas: stage1_gammas.clone(),
        stage2_gammas: stage2_gammas.clone(),
        stage3_gammas: stage3_gammas.clone(),
        stage4_gammas: stage4_gammas.clone(),
        stage5_gammas: stage5_gammas.clone(),
        booleanity_reference_address: booleanity_reference_address.clone(),
        booleanity_reference_cycle: booleanity_reference_cycle.clone(),
        booleanity_gamma,
        instruction_ra_gamma_powers: instruction_ra_gamma_powers.clone(),
        inc_gamma,
        bytecode_reduction_eta: eta,
    };

    if checked.zk {
        let stage5 = stage5.zk()?;
        let stage6a = verify_zk(
            checked,
            proof,
            transcript,
            &bytecode_address_claims,
            &booleanity_address_claims,
        )?;

        let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
            formula_dimensions
                .instruction_ra_virtualization
                .num_virtual_ra_polys(),
        );
        let inc_gamma = transcript.challenge_scalar();
        let eta = committed_program.then(|| transcript.challenge_scalar());

        let mut statements = vec![
            SumcheckStatement::new(
                bytecode_claims.sumcheck.rounds,
                bytecode_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                booleanity_claims.sumcheck.rounds,
                booleanity_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(
                ram_hamming_claims.sumcheck.rounds,
                ram_hamming_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(ram_ra_claims.sumcheck.rounds, ram_ra_claims.sumcheck.degree),
            SumcheckStatement::new(
                instruction_ra_claims.sumcheck.rounds,
                instruction_ra_claims.sumcheck.degree,
            ),
            SumcheckStatement::new(inc_claims.sumcheck.rounds, inc_claims.sumcheck.degree),
        ];
        if let Some(claim) = &trusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &untrusted_advice_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &bytecode_reduction_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        if let Some(claim) = &program_image_reduction_claims {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }
        let consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage6b_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;

        let bytecode_point = consistency
            .try_instance_point(bytecode_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::BytecodeReadRaf,
                reason: error.to_string(),
            })?;
        let bytecode_r_cycle = bytecode_point.iter().rev().copied().collect::<Vec<_>>();
        let bytecode_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(&stage6a.bytecode_r_address)
            .into_iter()
            .map(|r_address_chunk| {
                [r_address_chunk.as_slice(), bytecode_r_cycle.as_slice()].concat()
            })
            .collect::<Vec<_>>();

        let booleanity_point = consistency
            .try_instance_point(booleanity_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::Booleanity,
                reason: error.to_string(),
            })?;
        let booleanity_r_cycle = booleanity_point.iter().rev().copied().collect::<Vec<_>>();
        let booleanity_opening_point = [
            stage6a.booleanity_r_address.as_slice(),
            booleanity_r_cycle.as_slice(),
        ]
        .concat();

        let ram_hamming_point = consistency
            .try_instance_point(ram_hamming_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            })?;
        let ram_hamming_opening_point = trace_dimensions
            .cycle_opening_point(&ram_hamming_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            })?;

        let ram_ra_point = consistency
            .try_instance_point(ram_ra_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: error.to_string(),
            })?;
        let ram_ra_cycle = trace_dimensions
            .cycle_opening_point(&ram_ra_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: error.to_string(),
            })?;
        let ram_reduced_opening_point = stage5.output_points.ram_reduced_opening_point();
        if ram_reduced_opening_point.len() != log_k + log_t {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::RamRaVirtualization,
                reason: format!(
                    "RAM RA reduction opening point length mismatch: expected {}, got {}",
                    log_k + log_t,
                    ram_reduced_opening_point.len()
                ),
            });
        }
        let (ram_reduced_address, _) = ram_reduced_opening_point.split_at(log_k);
        let ram_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(ram_reduced_address)
            .into_iter()
            .map(|r_address_chunk| [r_address_chunk.as_slice(), ram_ra_cycle.as_slice()].concat())
            .collect::<Vec<_>>();

        let instruction_ra_point = consistency
            .try_instance_point(instruction_ra_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::InstructionRaVirtualization,
                reason: error.to_string(),
            })?;
        let instruction_ra_cycle = trace_dimensions
            .cycle_opening_point(&instruction_ra_point)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::InstructionRaVirtualization,
                reason: error.to_string(),
            })?;
        let instruction_ra_opening_points = proof
            .one_hot_config
            .committed_address_chunks(&stage5.instruction_r_address)
            .into_iter()
            .map(|r_address_chunk| {
                [r_address_chunk.as_slice(), instruction_ra_cycle.as_slice()].concat()
            })
            .collect::<Vec<_>>();

        let inc_point = consistency
            .try_instance_point(inc_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::IncClaimReduction,
                reason: error.to_string(),
            })?;
        let inc_opening_point =
            trace_dimensions
                .cycle_opening_point(&inc_point)
                .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::IncClaimReduction,
                    reason: error.to_string(),
                })?;

        let trusted_advice = if let (Some(layout), Some(claim)) =
            (trusted_advice_layout, trusted_advice_claims.as_ref())
        {
            Some(advice_cycle_phase_opening_point(
                &consistency,
                claim,
                layout,
            )?)
        } else {
            None
        };
        let untrusted_advice = if let (Some(layout), Some(claim)) =
            (untrusted_advice_layout, untrusted_advice_claims.as_ref())
        {
            Some(advice_cycle_phase_opening_point(
                &consistency,
                claim,
                layout,
            )?)
        } else {
            None
        };
        let bytecode_cycle_phase = if let (Some(layout), Some(claim)) = (
            bytecode_reduction_layout,
            bytecode_reduction_claims.as_ref(),
        ) {
            Some(committed_reduction_cycle_phase_opening_point(
                &consistency,
                claim,
                layout.precommitted(),
                JoltRelationId::BytecodeClaimReductionCyclePhase,
            )?)
        } else {
            None
        };
        let program_image_cycle_phase = if let (Some(layout), Some(claim)) = (
            program_image_reduction_layout,
            program_image_reduction_claims.as_ref(),
        ) {
            Some(committed_reduction_cycle_phase_opening_point(
                &consistency,
                claim,
                layout.precommitted(),
                JoltRelationId::ProgramImageClaimReductionCyclePhase,
            )?)
        } else {
            None
        };

        let bytecode_output_openings =
            bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
        let booleanity_output_openings =
            booleanity::booleanity_output_openings(formula_dimensions.ra_layout);
        let ram_ra_output_openings =
            ram::ra_virtualization_output_openings(formula_dimensions.ram_ra_virtualization);
        let instruction_ra_output_openings = instruction::ra_virtualization_output_openings(
            formula_dimensions.instruction_ra_virtualization,
        );
        let flat_instruction_ra_output_openings = instruction_ra_output_openings.all();
        let aliased_bytecode_ra_openings = aliased_booleanity_bytecode_openings(
            &bytecode_ra_opening_points,
            &booleanity_opening_point,
        );
        let bytecode_reduction_output_claims = bytecode_reduction_layout.map_or(0, |layout| {
            bytecode_reduction::cycle_phase_output_openings(
                layout.dimensions(),
                layout.chunk_count(),
            )
            .len()
        });
        let program_image_reduction_output_claims = program_image_reduction_layout
            .map_or(0, |layout| {
                program_image::cycle_phase_output_openings(layout.dimensions()).len()
            });
        let committed_output_claims = bytecode_output_openings.bytecode_ra.len()
            + booleanity_output_openings.len()
            - aliased_bytecode_ra_openings
            + 1
            + ram_ra_output_openings.len()
            + flat_instruction_ra_output_openings.len()
            + 2
            + usize::from(trusted_advice_claims.is_some())
            + usize::from(untrusted_advice_claims.is_some())
            + bytecode_reduction_output_claims
            + program_image_reduction_output_claims;
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage6b_sumcheck_proof,
                proof_label: "stage6b_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::BytecodeReadRaf,
            })?;

        let booleanity_layout = booleanity_dimensions.layout;
        let booleanity_points = |count: usize| vec![booleanity_opening_point.clone(); count];
        let output_points = Stage6OutputClaims {
            address_phase: Stage6AddressPhaseClaims {
                bytecode_read_raf: stage6a.bytecode_r_address.clone(),
                booleanity: stage6a.booleanity_r_address.clone(),
                bytecode_val_stages: committed_program
                    .then(|| core::array::from_fn(|_| stage6a.bytecode_r_address.clone())),
            },
            bytecode_read_raf: BytecodeReadRafOutputClaims {
                bytecode_ra: bytecode_ra_opening_points,
            },
            booleanity: BooleanityOutputClaims {
                instruction_ra: booleanity_points(booleanity_layout.instruction()),
                bytecode_ra: booleanity_points(booleanity_layout.bytecode()),
                ram_ra: booleanity_points(booleanity_layout.ram()),
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                ram_hamming_weight: ram_hamming_opening_point,
            },
            ram_ra_virtualization: RamRaVirtualizationOutputClaims {
                ram_ra: ram_ra_opening_points,
            },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra: instruction_ra_opening_points,
            },
            inc_claim_reduction: IncClaimReductionOutputClaims {
                ram_inc: inc_opening_point.clone(),
                rd_inc: inc_opening_point,
            },
            advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
                trusted: trusted_advice
                    .map(|opening_claim| AdviceCyclePhaseOutputClaim { opening_claim }),
                untrusted: untrusted_advice
                    .map(|opening_claim| AdviceCyclePhaseOutputClaim { opening_claim }),
            },
            bytecode_claim_reduction: match (bytecode_reduction_layout, bytecode_cycle_phase) {
                (Some(layout), Some(opening_point)) => {
                    Some(if layout.dimensions().has_address_phase() {
                        BytecodeCyclePhaseOutputClaims::Intermediate(opening_point)
                    } else {
                        BytecodeCyclePhaseOutputClaims::Chunks(vec![
                            opening_point;
                            layout.chunk_count()
                        ])
                    })
                }
                _ => None,
            },
            program_image_claim_reduction: program_image_cycle_phase
                .map(|opening_claim| ProgramImageCyclePhaseOutputClaim { opening_claim }),
        };

        return Ok(Stage6Output::Zk(Stage6ZkOutput {
            public: public(instruction_ra_gamma_powers, inc_gamma, eta),
            batch_consistency: consistency,
            batch_output_claims,
            address_phase_consistency: stage6a.address_phase_consistency,
            address_phase_output_claims: stage6a.address_phase_output_claims,
            output_points,
        }));
    }

    let stage1 = stage1.clear()?;
    let stage2 = stage2.clear()?;
    let stage3 = stage3.clear()?;
    let stage4 = stage4.clear()?;
    let stage5 = stage5.clear()?;
    let claims = &proof.clear_claims()?.stage6;
    if committed_program != claims.address_phase.bytecode_val_stages.is_some() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode Val-stage claims presence ({}) does not match committed program mode ({committed_program})",
                claims.address_phase.bytecode_val_stages.is_some()
            ),
        });
    }

    // The bytecode address-phase input claim is the gamma-folded bind of every
    // prior clear stage opening; the relation evaluates it through its input `Expr`
    // from these wired openings + the per-stage folding gammas.
    let bytecode_address_inputs = BytecodeReadRafAddressPhaseInputClaims::from_upstream(
        stage1, stage2, stage3, stage4, stage5,
    )?;
    let num_bytecode_val_stages = if committed_program {
        bytecode_reduction::NUM_BYTECODE_VAL_STAGES
    } else {
        0
    };
    let bytecode_address_relation = BytecodeReadRafAddressPhase::new(
        formula_dimensions.bytecode_read_raf,
        bytecode_gamma,
        [
            stage1_gammas[1],
            stage2_gammas[1],
            stage3_gammas[1],
            stage4_gammas[1],
            stage5_gammas[1],
        ],
        num_bytecode_val_stages,
    );
    let booleanity_address_relation = BooleanityAddressPhase::new(booleanity_dimensions);

    if trusted_advice_claims.is_none() && claims.advice_cycle_phase.trusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if untrusted_advice_claims.is_none() && claims.advice_cycle_phase.untrusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted),
        });
    }
    if bytecode_reduction_claims.is_none() && claims.bytecode_claim_reduction.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        });
    }
    if program_image_reduction_claims.is_none() && claims.program_image_claim_reduction.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        });
    }

    let stage6a = verify_clear(
        proof,
        transcript,
        claims,
        &bytecode_address_relation,
        &bytecode_address_inputs,
        &booleanity_address_relation,
    )?;
    let bytecode_r_address = stage6a.bytecode_r_address.clone();
    let booleanity_r_address = stage6a.booleanity_r_address.clone();

    let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
        formula_dimensions
            .instruction_ra_virtualization
            .num_virtual_ra_polys(),
    );
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(PCS::Field::one);
    let inc_gamma = transcript.challenge_scalar();
    let eta = committed_program.then(|| transcript.challenge_scalar());

    // Build the stage-6b cycle-phase relation bundle shared with the prover. Its
    // construction inputs (per-stage cycle bindings, reduced points, reduction
    // weights) are derived here from the prior-stage clear outputs and the
    // stage-6a results; the post-sumcheck section recomputes the sumcheck-point
    // dependent openings against these same values.
    // Per-stage cycle bindings, the reduced RAM point, the instruction RA point,
    // and the per-source increment cycle suffixes, single-sourced through the same
    // prover-facing helpers the prover builds its backend state requests from, so
    // the bundle algebra cannot drift between the two sides.
    let stage1_cycle_binding = stage6_stage1_cycle_binding(stage1)?;
    let stage_cycle_points = stage6_bytecode_cycle_points(stage1, stage2, stage3, stage4, stage5)?;
    let register_points = stage6_bytecode_register_points(stage4, stage5)?;
    let ram_reduced = stage6_stage5_ram_reduced_opening_point(stage5, log_k, log_t)?;
    let instruction_read_raf = stage6_instruction_read_raf_point(stage5);
    let inc_cycle_points = stage6_inc_claim_reduction_cycle_points(stage2, stage4, stage5, log_k)?;
    let entry_bytecode_index = preprocessing
        .program
        .entry_bytecode_index()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "entry address was not found in bytecode preprocessing".to_string(),
        })?;
    let bytecode_table = if committed_program {
        None
    } else {
        Some(
            preprocessing
                .program
                .as_full()
                .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                    stage: JoltRelationId::BytecodeReadRaf,
                    reason: "full bytecode table is unavailable".to_string(),
                })?
                .bytecode
                .bytecode
                .as_slice(),
        )
    };
    let cycle_bytecode_reduction_weights = match (bytecode_reduction_layout, eta) {
        (Some(layout), Some(eta)) => Some(bytecode_reduction_weights(
            layout,
            BytecodeReductionWeightInputs {
                eta,
                stage1_gammas: &stage1_gammas,
                stage2_gammas: &stage2_gammas,
                stage3_gammas: &stage3_gammas,
                stage4_gammas: &stage4_gammas,
                stage5_gammas: &stage5_gammas,
                register_read_write_point: register_points.register_read_write_address,
                register_val_evaluation_point: register_points.register_val_evaluation_address,
                bytecode_r_address: &bytecode_r_address,
            },
        )?),
        _ => None,
    };
    let relations = Stage6Relations::build(
        Stage6RelationsParams {
            bytecode_dimensions: formula_dimensions.bytecode_read_raf,
            booleanity_dimensions,
            trace_dimensions,
            ram_ra_dimensions: formula_dimensions.ram_ra_virtualization,
            instruction_ra_dimensions: formula_dimensions.instruction_ra_virtualization,
            committed_chunk_bits: proof.one_hot_config.committed_chunk_bits(),
            bytecode_table,
            entry_bytecode_index,
            bytecode_r_address: bytecode_r_address.clone(),
            booleanity_r_address: booleanity_r_address.clone(),
            address_bytecode_read_raf: claims.address_phase.bytecode_read_raf,
            address_booleanity: claims.address_phase.booleanity,
            address_val_stages: claims
                .address_phase
                .bytecode_val_stages
                .map_or_else(Vec::new, |stages| stages.to_vec()),
            bytecode_gamma,
            instruction_ra_gamma,
            inc_gamma,
            booleanity_gamma,
            eta,
            stage_cycle_points: stage_cycle_points.clone(),
            register_read_write_point: register_points.register_read_write_address.to_vec(),
            register_val_evaluation_point: register_points.register_val_evaluation_address.to_vec(),
            stage_gammas: [
                stage1_gammas.clone(),
                stage2_gammas.clone(),
                stage3_gammas.clone(),
                stage4_gammas.clone(),
                stage5_gammas.clone(),
            ],
            booleanity_reference_address: booleanity_reference_address.clone(),
            booleanity_reference_cycle: booleanity_reference_cycle.clone(),
            stage1_cycle_binding: stage1_cycle_binding.to_vec(),
            ram_reduced_address: ram_reduced.address.to_vec(),
            ram_reduced_cycle: ram_reduced.cycle.to_vec(),
            instruction_r_address: instruction_read_raf.address.to_vec(),
            instruction_r_cycle: instruction_read_raf.cycle.to_vec(),
            inc_cycle_points: [
                inc_cycle_points.ram_read_write_cycle.to_vec(),
                inc_cycle_points.ram_val_check_cycle.to_vec(),
                inc_cycle_points.registers_read_write_cycle.to_vec(),
                inc_cycle_points.registers_val_evaluation_cycle.to_vec(),
            ],
            trusted_advice_layout,
            untrusted_advice_layout,
            bytecode_reduction_layout,
            program_image_reduction_layout,
            bytecode_reduction_weights: cycle_bytecode_reduction_weights.clone(),
            program_image_r_addr_rw: stage4.output_claims.ram_val_check.ram_ra.point[..log_k]
                .to_vec(),
        },
        stage2,
        stage4,
        stage5,
    )?;
    // The per-instance batched-sumcheck claims (claimed sums), single-sourced
    // through the same bundle method the prover uses, in canonical batch order.
    let sumcheck_claims = relations.sumcheck_claims()?;

    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage6b_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let bytecode_point = batch
        .try_instance_point(bytecode_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
    let bytecode_r_cycle = bytecode_point.iter().rev().copied().collect::<Vec<_>>();
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    if claims.bytecode_read_raf.bytecode_ra.len() != bytecode_output_openings.bytecode_ra.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                bytecode_output_openings.bytecode_ra.len(),
                claims.bytecode_read_raf.bytecode_ra.len()
            ),
        });
    }

    let bytecode_points = relations
        .bytecode_read_raf
        .derive_opening_points(bytecode_point, &relations.bytecode_read_raf_inputs)?;
    let bytecode_outputs = zip_openings(&claims.bytecode_read_raf, &bytecode_points);
    let bytecode_output = relations
        .bytecode_read_raf
        .expected_output(&relations.bytecode_read_raf_inputs, &bytecode_outputs)?;
    let bytecode_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(&bytecode_r_address)
        .into_iter()
        .map(|r_address_chunk| [r_address_chunk.as_slice(), bytecode_r_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

    let booleanity_point = batch
        .try_instance_point(booleanity_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;
    let booleanity_r_cycle = booleanity_point.iter().rev().copied().collect::<Vec<_>>();
    let booleanity_opening_point = [
        booleanity_r_address.as_slice(),
        booleanity_r_cycle.as_slice(),
    ]
    .concat();
    let booleanity_points = relations
        .booleanity
        .derive_opening_points(booleanity_point, &relations.booleanity_inputs)?;
    let booleanity_outputs = zip_openings(&claims.booleanity, &booleanity_points);
    let booleanity_output = relations
        .booleanity
        .expected_output(&relations.booleanity_inputs, &booleanity_outputs)?;
    let ram_hamming_point = batch
        .try_instance_point(ram_hamming_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamHammingBooleanity,
            reason: error.to_string(),
        })?;
    let ram_hamming_points = relations
        .ram_hamming
        .derive_opening_points(ram_hamming_point, &relations.ram_hamming_inputs)?;
    let ram_hamming_outputs = zip_openings(&claims.ram_hamming_booleanity, &ram_hamming_points);
    let ram_hamming_output = relations
        .ram_hamming
        .expected_output(&relations.ram_hamming_inputs, &ram_hamming_outputs)?;

    let ram_ra_point = batch
        .try_instance_point(ram_ra_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        })?;
    let ram_ra_points = relations
        .ram_ra
        .derive_opening_points(ram_ra_point, &relations.ram_ra_inputs)?;
    let ram_ra_outputs = zip_openings(&claims.ram_ra_virtualization, &ram_ra_points);
    let ram_ra_output = relations
        .ram_ra
        .expected_output(&relations.ram_ra_inputs, &ram_ra_outputs)?;

    let instruction_ra_point = batch
        .try_instance_point(instruction_ra_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_ra_points = relations
        .instruction_ra
        .derive_opening_points(instruction_ra_point, &relations.instruction_ra_inputs)?;
    let instruction_ra_outputs = zip_openings(
        &claims.instruction_ra_virtualization,
        &instruction_ra_points,
    );
    let instruction_ra_output = relations
        .instruction_ra
        .expected_output(&relations.instruction_ra_inputs, &instruction_ra_outputs)?;

    let inc_point = batch
        .try_instance_point(inc_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let inc_points = relations
        .inc
        .derive_opening_points(inc_point, &relations.inc_inputs)?;
    let inc_outputs = zip_openings(&claims.inc_claim_reduction, &inc_points);
    let inc_output = relations
        .inc
        .expected_output(&relations.inc_inputs, &inc_outputs)?;

    let trusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        trusted_advice_layout,
        trusted_advice_claims.as_ref(),
        claims.advice_cycle_phase.trusted.as_ref(),
    ) {
        Some(verify_advice_cycle_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Trusted,
            opening_claim,
            stage4,
        )?)
    } else {
        None
    };
    let untrusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        untrusted_advice_layout,
        untrusted_advice_claims.as_ref(),
        claims.advice_cycle_phase.untrusted.as_ref(),
    ) {
        Some(verify_advice_cycle_phase(
            &batch,
            claim,
            layout,
            JoltAdviceKind::Untrusted,
            opening_claim,
            stage4,
        )?)
    } else {
        None
    };

    if let (Some(layout), Some(_)) = (trusted_advice_layout, trusted_advice_claims.as_ref()) {
        if trusted_advice.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_output_openings(
                    JoltAdviceKind::Trusted,
                    layout.dimensions(),
                )[0],
            });
        }
    }
    if let (Some(layout), Some(_)) = (untrusted_advice_layout, untrusted_advice_claims.as_ref()) {
        if untrusted_advice.is_none() {
            return Err(VerifierError::MissingOpeningClaim {
                id: advice::cycle_phase_output_openings(
                    JoltAdviceKind::Untrusted,
                    layout.dimensions(),
                )[0],
            });
        }
    }
    let bytecode_cycle_phase =
        if let (Some(layout), Some(claim)) = (
            bytecode_reduction_layout,
            bytecode_reduction_claims.as_ref(),
        ) {
            let output_claims = claims.bytecode_claim_reduction.as_ref().ok_or(
                VerifierError::MissingOpeningClaim {
                    id: bytecode_reduction::cycle_phase_output_openings(
                        layout.dimensions(),
                        layout.chunk_count(),
                    )[0],
                },
            )?;
            let eta = eta.ok_or(VerifierError::MissingStageClaimChallenge {
                id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
            })?;
            // Same inputs as the bundle's `cycle_bytecode_reduction_weights`
            // (both gated on `bytecode_reduction_layout` + `eta`), so reuse it
            // rather than recomputing the fold.
            let weights = cycle_bytecode_reduction_weights.clone().ok_or(
                VerifierError::MissingStageClaimChallenge {
                    id: JoltChallengeId::from(BytecodeClaimReductionChallenge::Eta),
                },
            )?;
            Some(verify_bytecode_cycle_phase(
                &batch,
                claim,
                layout,
                output_claims,
                weights,
                eta,
            )?)
        } else {
            None
        };
    let program_image_cycle_phase = if let (Some(layout), Some(claim)) = (
        program_image_reduction_layout,
        program_image_reduction_claims.as_ref(),
    ) {
        let output_claim = claims.program_image_claim_reduction.as_ref().ok_or(
            VerifierError::MissingOpeningClaim {
                id: program_image::cycle_phase_output_openings(layout.dimensions())[0],
            },
        )?;
        let r_addr_rw = &stage4.output_claims.ram_val_check.ram_ra.point[..log_k];
        let input_claim = match (
            &relations.program_image_reduction,
            &relations.program_image_reduction_inputs,
        ) {
            (Some(relation), Some(inputs)) => relation.input_claim(inputs)?,
            _ => {
                return Err(VerifierError::MissingOpeningClaim {
                    id: program_image::ram_val_check_contribution_opening(),
                })
            }
        };
        Some(verify_program_image_cycle_phase(
            &batch,
            claim,
            layout,
            output_claim,
            r_addr_rw,
            input_claim,
        )?)
    } else {
        None
    };

    // The per-instance expected output claims in canonical batch order: the six
    // base relations, then each present committed-mode reduction. The optional
    // tail mirrors `sumcheck_claims` and the batching-coefficient order.
    let mut expected_outputs_in_order = vec![
        bytecode_output,
        booleanity_output,
        ram_hamming_output,
        ram_ra_output,
        instruction_ra_output,
        inc_output,
    ];
    for output_claim in [
        trusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        untrusted_advice
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        bytecode_cycle_phase
            .as_ref()
            .map(|verified| verified.expected_output_claim),
        program_image_cycle_phase
            .as_ref()
            .map(|verified| verified.expected_output_claim),
    ]
    .into_iter()
    .flatten()
    {
        expected_outputs_in_order.push(output_claim);
    }
    if batch.batching_coefficients.len() != expected_outputs_in_order.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "Stage 6 batch verifier returned {} coefficients for {} instances",
                batch.batching_coefficients.len(),
                expected_outputs_in_order.len()
            ),
        });
    }
    let expected_final_claim = batch
        .batching_coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum();
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
        });
    }

    append_opening_claims(
        transcript,
        claims,
        &bytecode_ra_opening_points,
        &booleanity_opening_point,
    );

    let output_points = Stage6OutputClaims {
        address_phase: Stage6AddressPhaseClaims {
            bytecode_read_raf: bytecode_r_address.clone(),
            booleanity: booleanity_r_address.clone(),
            bytecode_val_stages: claims
                .address_phase
                .bytecode_val_stages
                .as_ref()
                .map(|_| core::array::from_fn(|_| bytecode_r_address.clone())),
        },
        bytecode_read_raf: bytecode_points,
        booleanity: booleanity_points,
        ram_hamming_booleanity: ram_hamming_points,
        ram_ra_virtualization: ram_ra_points,
        instruction_ra_virtualization: instruction_ra_points,
        inc_claim_reduction: inc_points,
        advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
            trusted: trusted_advice
                .as_ref()
                .map(|verified| AdviceCyclePhaseOutputClaim {
                    opening_claim: verified.opening_point.clone(),
                }),
            untrusted: untrusted_advice
                .as_ref()
                .map(|verified| AdviceCyclePhaseOutputClaim {
                    opening_claim: verified.opening_point.clone(),
                }),
        },
        bytecode_claim_reduction: bytecode_cycle_phase.as_ref().map(|verified| {
            match claims.bytecode_claim_reduction.as_ref() {
                Some(BytecodeCyclePhaseOutputClaims::Chunks(chunks)) => {
                    BytecodeCyclePhaseOutputClaims::Chunks(vec![
                        verified.opening_point.clone();
                        chunks.len()
                    ])
                }
                _ => BytecodeCyclePhaseOutputClaims::Intermediate(verified.opening_point.clone()),
            }
        }),
        program_image_claim_reduction: program_image_cycle_phase.as_ref().map(|verified| {
            ProgramImageCyclePhaseOutputClaim {
                opening_claim: verified.opening_point.clone(),
            }
        }),
    };

    Ok(Stage6Output::Clear(Stage6ClearOutput {
        public: public(instruction_ra_gamma_powers, inc_gamma, eta),
        output_claims: claims.clone(),
        output_points,
        bytecode_reduction_weights: cycle_bytecode_reduction_weights,
    }))
}

// ============================================================================
// Stage 6 prover/verifier shared helpers.
//
// The functions and types below are the public, prover-facing Stage 6 API
// re-exported by `mod.rs`. They are single-sourced extractions of the value
// derivations performed inline by the canonical `verify()` above: each helper
// computes exactly the same quantity (same formula, same Fiat-Shamir gamma
// counts, same instance/absorption order) so the prover's Stage 6 batch cannot
// drift from the verifier.
//
// NOTE on claim reductions: new 09 added bytecode and program-image claim
// reductions to the verifier's Stage 6 batch (the input claims come from
// `Stage6Relations::sumcheck_claims`, the expected outputs from each relation's
// `expected_output`, both shared with the prover via the bundle).
// The public structs below do NOT carry those reduction fields; the prover-side
// Stage 6 batch does not (yet) materialize them. The `stage6_input_claim_values`
// / `stage6_expected_output_claim_values` ordering therefore covers only the
// non-reduction instances that the public batch shares with `verify()`; the
// reductions are appended last by `verify()` itself, after every shared
// instance, preserving the shared prefix order.
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BatchInputClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    pub ram_hamming_booleanity: F,
    pub ram_ra_virtualization: F,
    pub instruction_ra_virtualization: F,
    pub inc_claim_reduction: F,
    pub trusted_advice_cycle_phase: Option<F>,
    pub untrusted_advice_cycle_phase: Option<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BooleanityReference<F: Field> {
    pub address: Vec<F>,
    pub cycle: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6TranscriptChallenges<F: Field> {
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference: Stage6BooleanityReference<F>,
    pub booleanity_gamma: F,
    pub instruction_ra_gamma_powers: Vec<F>,
    pub instruction_ra_gamma: F,
    pub inc_gamma: F,
}

pub fn stage6_stage1_cycle_binding<F: Field>(
    stage1: &Stage1ClearOutput<F>,
) -> Result<&[F], VerifierError> {
    let (_, cycle) = stage1
        .remainder
        .sumcheck_point
        .as_slice()
        .split_first()
        .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: "Stage 1 remainder point is empty".to_string(),
        })?;
    Ok(cycle)
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeRegisterPoints<'a, F: Field> {
    pub register_read_write_address: &'a [F],
    pub register_read_write_cycle: &'a [F],
    pub register_val_evaluation_address: &'a [F],
    pub register_val_evaluation_cycle: &'a [F],
}

pub fn stage6_bytecode_register_points<'a, F: Field>(
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
) -> Result<Stage6BytecodeRegisterPoints<'a, F>, VerifierError> {
    let (register_read_write_address, register_read_write_cycle) = stage6_checked_split(
        "Stage 6 stage4 register read-write opening",
        &stage4
            .output_claims
            .registers_read_write
            .registers_val
            .point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    let (register_val_evaluation_address, register_val_evaluation_cycle) = stage6_checked_split(
        "Stage 6 stage5 register value-evaluation opening",
        stage5.registers_opening_point(),
        REGISTER_ADDRESS_BITS,
        JoltRelationId::BytecodeReadRaf,
    )?;
    Ok(Stage6BytecodeRegisterPoints {
        register_read_write_address,
        register_read_write_cycle,
        register_val_evaluation_address,
        register_val_evaluation_cycle,
    })
}

pub fn stage6_bytecode_cycle_points<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<[Vec<F>; 5], VerifierError> {
    let stage1_cycle = stage6_stage1_cycle_binding(stage1)?
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let stage2_cycle = stage2.output_claims.product_remainder_point().to_vec();
    let stage3_cycle = stage3.output_claims.shift_opening_point().to_vec();
    let register_points = stage6_bytecode_register_points(stage4, stage5)?;
    Ok([
        stage1_cycle,
        stage2_cycle,
        stage3_cycle,
        register_points.register_read_write_cycle.to_vec(),
        register_points.register_val_evaluation_cycle.to_vec(),
    ])
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6RamReducedOpeningPoint<'a, F: Field> {
    pub full: &'a [F],
    pub address: &'a [F],
    pub cycle: &'a [F],
}

impl<F: Field> Stage6RamReducedOpeningPoint<'_, F> {
    pub fn committed_address_chunks(self, committed_chunk_bits: usize) -> Vec<Vec<F>> {
        committed_address_chunks(self.address, committed_chunk_bits)
    }

    pub fn opening_point(self, cycle: &[F]) -> Vec<F> {
        [self.address, cycle].concat()
    }

    pub fn committed_opening_points(self, cycle: &[F], committed_chunk_bits: usize) -> Vec<Vec<F>> {
        self.committed_address_chunks(committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), cycle].concat())
            .collect()
    }
}

fn stage6_ram_reduced_opening_point<F: Field>(
    point: &[F],
    log_k: usize,
    log_t: usize,
) -> Result<Stage6RamReducedOpeningPoint<'_, F>, VerifierError> {
    let (address, cycle) = stage6_checked_exact_split(
        "Stage 6 RAM RA reduction opening point",
        point,
        log_k,
        log_k + log_t,
        JoltRelationId::RamRaVirtualization,
    )?;
    Ok(Stage6RamReducedOpeningPoint {
        full: point,
        address,
        cycle,
    })
}

pub fn stage6_stage5_ram_reduced_opening_point<F: Field>(
    stage5: &Stage5ClearOutput<F>,
    log_k: usize,
    log_t: usize,
) -> Result<Stage6RamReducedOpeningPoint<'_, F>, VerifierError> {
    stage6_ram_reduced_opening_point(stage5.ram_reduced_opening_point(), log_k, log_t)
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6InstructionReadRafPoint<'a, F: Field> {
    pub address: &'a [F],
    pub cycle: &'a [F],
}

impl<F: Field> Stage6InstructionReadRafPoint<'_, F> {
    pub fn committed_address_chunks(self, committed_chunk_bits: usize) -> Vec<Vec<F>> {
        committed_address_chunks(self.address, committed_chunk_bits)
    }

    pub fn opening_point(self, cycle: &[F]) -> Vec<F> {
        [self.address, cycle].concat()
    }

    pub fn committed_opening_points(self, cycle: &[F], committed_chunk_bits: usize) -> Vec<Vec<F>> {
        self.committed_address_chunks(committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), cycle].concat())
            .collect()
    }
}

pub fn stage6_instruction_read_raf_point<F: Field>(
    stage5: &Stage5ClearOutput<F>,
) -> Stage6InstructionReadRafPoint<'_, F> {
    Stage6InstructionReadRafPoint {
        address: &stage5.instruction_r_address,
        cycle: stage5.instruction_r_cycle(),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6AdviceCyclePhaseReference<'a, F: Field> {
    pub opening_claim: F,
    pub opening_point: &'a [F],
}

pub fn stage6_advice_cycle_phase_reference<F: Field>(
    stage4: &Stage4ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<Stage6AdviceCyclePhaseReference<'_, F>, VerifierError> {
    let contribution = stage4
        .ram_val_check_init
        .advice_contributions
        .iter()
        .find(|contribution| contribution.kind == kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::ram_val_check_advice_opening(kind),
        })?;
    Ok(Stage6AdviceCyclePhaseReference {
        opening_claim: contribution.opening.value,
        opening_point: &contribution.opening.point,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6IncClaimReductionCyclePoints<'a, F: Field> {
    pub ram_read_write_cycle: &'a [F],
    pub ram_val_check_cycle: &'a [F],
    pub registers_read_write_cycle: &'a [F],
    pub registers_val_evaluation_cycle: &'a [F],
}

impl<F: Field> Stage6IncClaimReductionCyclePoints<'_, F> {
    pub fn reversed_cycles(&self) -> [Vec<F>; 4] {
        [
            self.ram_read_write_cycle.iter().rev().copied().collect(),
            self.ram_val_check_cycle.iter().rev().copied().collect(),
            self.registers_read_write_cycle
                .iter()
                .rev()
                .copied()
                .collect(),
            self.registers_val_evaluation_cycle
                .iter()
                .rev()
                .copied()
                .collect(),
        ]
    }
}

pub fn stage6_inc_claim_reduction_cycle_points<'a, F: Field>(
    stage2: &'a Stage2ClearOutput<F>,
    stage4: &'a Stage4ClearOutput<F>,
    stage5: &'a Stage5ClearOutput<F>,
    log_k: usize,
) -> Result<Stage6IncClaimReductionCyclePoints<'a, F>, VerifierError> {
    let (_, ram_read_write_cycle) = stage6_checked_split(
        "Stage 6 RAM read-write opening",
        stage2.output_claims.ram_read_write_point(),
        log_k,
        JoltRelationId::IncClaimReduction,
    )?;
    let (_, ram_val_check_cycle) = stage6_checked_split(
        "Stage 6 RAM value-check opening",
        &stage4.output_claims.ram_val_check.ram_ra.point,
        log_k,
        JoltRelationId::IncClaimReduction,
    )?;
    let (_, registers_read_write_cycle) = stage6_checked_split(
        "Stage 6 register read-write opening",
        &stage4
            .output_claims
            .registers_read_write
            .registers_val
            .point,
        REGISTER_ADDRESS_BITS,
        JoltRelationId::IncClaimReduction,
    )?;
    let (_, registers_val_evaluation_cycle) = stage6_checked_split(
        "Stage 6 register value-evaluation opening",
        stage5.registers_opening_point(),
        REGISTER_ADDRESS_BITS,
        JoltRelationId::IncClaimReduction,
    )?;
    Ok(Stage6IncClaimReductionCyclePoints {
        ram_read_write_cycle,
        ram_val_check_cycle,
        registers_read_write_cycle,
        registers_val_evaluation_cycle,
    })
}

fn stage6_checked_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
    stage: JoltRelationId,
) -> Result<(&'a [F], &'a [F]), VerifierError> {
    if point.len() < split_at {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: format!(
                "{label} has {} variables, expected at least {split_at}",
                point.len()
            ),
        });
    }
    Ok(point.split_at(split_at))
}

fn stage6_checked_exact_split<'a, F: Field>(
    label: &'static str,
    point: &'a [F],
    split_at: usize,
    expected_len: usize,
    stage: JoltRelationId,
) -> Result<(&'a [F], &'a [F]), VerifierError> {
    if point.len() != expected_len {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: format!(
                "{label} length mismatch: expected {expected_len}, got {}",
                point.len()
            ),
        });
    }
    Ok(point.split_at(split_at))
}

fn stage6_booleanity_reference<F, T>(
    instruction_address: &[F],
    instruction_cycle: &[F],
    committed_chunk_bits: usize,
    transcript: &mut T,
) -> Stage6BooleanityReference<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let mut address = instruction_address.to_vec();
    address.reverse();
    if address.len() < committed_chunk_bits {
        address.extend(transcript.challenge_vector(committed_chunk_bits - address.len()));
    } else {
        address = address[address.len() - committed_chunk_bits..].to_vec();
    }
    let mut cycle = instruction_cycle.to_vec();
    cycle.reverse();
    Stage6BooleanityReference { address, cycle }
}

/// The Stage 6 transcript challenges drawn BEFORE the stage 6a address-phase
/// sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PreAddressChallenges<F: Field> {
    pub bytecode_gamma_powers: Vec<F>,
    pub stage1_gammas: Vec<F>,
    pub stage2_gammas: Vec<F>,
    pub stage3_gammas: Vec<F>,
    pub stage4_gammas: Vec<F>,
    pub stage5_gammas: Vec<F>,
    pub booleanity_reference: Stage6BooleanityReference<F>,
    pub booleanity_gamma: F,
}

/// The Stage 6 transcript challenges drawn AFTER the stage 6a address-phase
/// sumcheck (and after its output openings are appended to the transcript).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6PostAddressChallenges<F: Field> {
    pub instruction_ra_gamma_powers: Vec<F>,
    pub instruction_ra_gamma: F,
    pub inc_gamma: F,
}

pub fn stage6_pre_address_transcript_challenges<F, T>(
    instruction_address: &[F],
    instruction_cycle: &[F],
    committed_chunk_bits: usize,
    lookup_table_flag_count: usize,
    transcript: &mut T,
) -> Stage6PreAddressChallenges<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let bytecode_gamma_powers = transcript.challenge_scalar_powers(stage6_bytecode_gamma_count());
    let stage1_gammas = transcript.challenge_scalar_powers(stage6_stage1_gamma_count());
    let stage2_gammas = transcript.challenge_scalar_powers(stage6_stage2_gamma_count());
    let stage3_gammas = transcript.challenge_scalar_powers(stage6_stage3_gamma_count());
    let stage4_gammas = transcript.challenge_scalar_powers(stage6_stage4_gamma_count());
    let stage5_gammas =
        transcript.challenge_scalar_powers(stage6_stage5_gamma_count(lookup_table_flag_count));
    let booleanity_reference = stage6_booleanity_reference(
        instruction_address,
        instruction_cycle,
        committed_chunk_bits,
        transcript,
    );
    let mut booleanity_gamma = transcript.challenge();
    if booleanity_gamma == F::zero() {
        booleanity_gamma = F::one();
    }
    Stage6PreAddressChallenges {
        bytecode_gamma_powers,
        stage1_gammas,
        stage2_gammas,
        stage3_gammas,
        stage4_gammas,
        stage5_gammas,
        booleanity_reference,
        booleanity_gamma,
    }
}

pub fn stage6_post_address_transcript_challenges<F, T>(
    instruction_ra_dimensions: InstructionRaVirtualizationDimensions,
    transcript: &mut T,
) -> Stage6PostAddressChallenges<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let instruction_ra_gamma_powers =
        transcript.challenge_scalar_powers(instruction_ra_dimensions.num_virtual_ra_polys());
    let instruction_ra_gamma = instruction_ra_gamma_powers
        .get(1)
        .copied()
        .unwrap_or_else(F::one);
    let inc_gamma = transcript.challenge_scalar();
    Stage6PostAddressChallenges {
        instruction_ra_gamma_powers,
        instruction_ra_gamma,
        inc_gamma,
    }
}

impl<F: Field> Stage6TranscriptChallenges<F> {
    /// Reassembles the full transcript-challenge record from the pre/post address
    /// halves, preserving the canonical field order.
    pub fn from_address_phases(
        pre: Stage6PreAddressChallenges<F>,
        post: Stage6PostAddressChallenges<F>,
    ) -> Self {
        Self {
            bytecode_gamma_powers: pre.bytecode_gamma_powers,
            stage1_gammas: pre.stage1_gammas,
            stage2_gammas: pre.stage2_gammas,
            stage3_gammas: pre.stage3_gammas,
            stage4_gammas: pre.stage4_gammas,
            stage5_gammas: pre.stage5_gammas,
            booleanity_reference: pre.booleanity_reference,
            booleanity_gamma: pre.booleanity_gamma,
            instruction_ra_gamma_powers: post.instruction_ra_gamma_powers,
            instruction_ra_gamma: post.instruction_ra_gamma,
            inc_gamma: post.inc_gamma,
        }
    }
}

pub fn stage6_public_output<F: Field>(
    transcript_challenges: &Stage6TranscriptChallenges<F>,
    bytecode_reduction_eta: Option<F>,
) -> Stage6PublicOutput<F> {
    Stage6PublicOutput {
        bytecode_gamma_powers: transcript_challenges.bytecode_gamma_powers.clone(),
        stage1_gammas: transcript_challenges.stage1_gammas.clone(),
        stage2_gammas: transcript_challenges.stage2_gammas.clone(),
        stage3_gammas: transcript_challenges.stage3_gammas.clone(),
        stage4_gammas: transcript_challenges.stage4_gammas.clone(),
        stage5_gammas: transcript_challenges.stage5_gammas.clone(),
        booleanity_reference_address: transcript_challenges.booleanity_reference.address.clone(),
        booleanity_reference_cycle: transcript_challenges.booleanity_reference.cycle.clone(),
        booleanity_gamma: transcript_challenges.booleanity_gamma,
        instruction_ra_gamma_powers: transcript_challenges.instruction_ra_gamma_powers.clone(),
        inc_gamma: transcript_challenges.inc_gamma,
        bytecode_reduction_eta,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage6BytecodeReadRafExpectedOutputInputs<'a, F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub public_values: &'a bytecode::BytecodeReadRafPublicValues<F>,
    pub bytecode_ra: &'a [F],
    pub gamma: F,
}

pub fn stage6_bytecode_read_raf_expected_output<F: Field>(
    inputs: Stage6BytecodeReadRafExpectedOutputInputs<'_, F>,
) -> Result<F, VerifierError> {
    let output_openings = bytecode::read_raf_output_openings(inputs.dimensions);
    if inputs.bytecode_ra.len() != output_openings.bytecode_ra.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                output_openings.bytecode_ra.len(),
                inputs.bytecode_ra.len()
            ),
        });
    }
    let claim = bytecode::read_raf::<F>(inputs.dimensions);
    claim.output.expression().try_evaluate(
        |id| {
            for (index, opening) in output_openings.bytecode_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(inputs.bytecode_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => Ok(inputs.gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltPublicId::BytecodeReadRaf(public_id) => inputs
                .public_values
                .value(*public_id)
                .ok_or(VerifierError::MissingStageClaimPublic { id: *id }),
            _ => Err(VerifierError::MissingStageClaimPublic { id: *id }),
        },
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage6BatchExpectedOutputClaims<F: Field> {
    pub bytecode_read_raf: F,
    pub booleanity: F,
    pub ram_hamming_booleanity: F,
    pub ram_ra_virtualization: F,
    pub instruction_ra_virtualization: F,
    pub inc_claim_reduction: F,
    pub trusted_advice_cycle_phase: Option<F>,
    pub untrusted_advice_cycle_phase: Option<F>,
}

pub fn stage6_expected_output_claim_values<F: Field>(
    expected_outputs: &Stage6BatchExpectedOutputClaims<F>,
) -> Vec<F> {
    let mut values = vec![
        expected_outputs.bytecode_read_raf,
        expected_outputs.booleanity,
        expected_outputs.ram_hamming_booleanity,
        expected_outputs.ram_ra_virtualization,
        expected_outputs.instruction_ra_virtualization,
        expected_outputs.inc_claim_reduction,
    ];
    if let Some(output_claim) = expected_outputs.trusted_advice_cycle_phase {
        values.push(output_claim);
    }
    if let Some(output_claim) = expected_outputs.untrusted_advice_cycle_phase {
        values.push(output_claim);
    }
    values
}

pub fn stage6_expected_final_claim<F: Field>(
    coefficients: &[F],
    expected_outputs: &Stage6BatchExpectedOutputClaims<F>,
) -> Result<F, VerifierError> {
    let expected_outputs_in_order = stage6_expected_output_claim_values(expected_outputs);
    if coefficients.len() != expected_outputs_in_order.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "Stage 6 batch verifier returned {} coefficients for {} instances",
                coefficients.len(),
                expected_outputs_in_order.len()
            ),
        });
    }
    Ok(coefficients
        .iter()
        .zip(expected_outputs_in_order)
        .map(|(coefficient, output)| *coefficient * output)
        .sum())
}

const fn stage6_bytecode_gamma_count() -> usize {
    8
}

const fn stage6_stage1_gamma_count() -> usize {
    2 + NUM_CIRCUIT_FLAGS
}

const fn stage6_stage2_gamma_count() -> usize {
    4
}

const fn stage6_stage3_gamma_count() -> usize {
    9
}

const fn stage6_stage4_gamma_count() -> usize {
    3
}

const fn stage6_stage5_gamma_count(lookup_table_flag_count: usize) -> usize {
    2 + lookup_table_flag_count
}

pub fn stage6_output_claim_values<F: Field>(
    claims: &Stage6OutputClaims<F>,
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> Vec<F> {
    let mut values = Vec::new();
    values.extend(claims.bytecode_read_raf.bytecode_ra.iter().copied());
    values.extend(claims.booleanity.instruction_ra.iter().copied());
    for (index, &claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_ra_opening_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_opening_point)
        {
            continue;
        }
        values.push(claim);
    }
    values.extend(claims.booleanity.ram_ra.iter().copied());
    values.push(claims.ram_hamming_booleanity.ram_hamming_weight);
    values.extend(claims.ram_ra_virtualization.ram_ra.iter().copied());
    values.extend(
        claims
            .instruction_ra_virtualization
            .committed_instruction_ra
            .iter()
            .copied(),
    );
    values.push(claims.inc_claim_reduction.ram_inc);
    values.push(claims.inc_claim_reduction.rd_inc);
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        values.push(opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        values.push(opening_claim.opening_claim);
    }
    values
}

pub fn stage6_input_claim_values<F: Field>(claims: &Stage6BatchInputClaims<F>) -> Vec<F> {
    let mut values = vec![
        claims.bytecode_read_raf,
        claims.booleanity,
        claims.ram_hamming_booleanity,
        claims.ram_ra_virtualization,
        claims.instruction_ra_virtualization,
        claims.inc_claim_reduction,
    ];
    if let Some(input_claim) = claims.trusted_advice_cycle_phase {
        values.push(input_claim);
    }
    if let Some(input_claim) = claims.untrusted_advice_cycle_phase {
        values.push(input_claim);
    }
    values
}
pub(super) struct Stage6AZkOutput<F: Field, C> {
    pub address_phase_consistency: BatchedCommittedSumcheckConsistency<F, C>,
    pub address_phase_output_claims: CommittedOutputClaimOutput<C>,
    pub bytecode_r_address: Vec<F>,
    pub booleanity_r_address: Vec<F>,
}

pub(super) struct Stage6AClearOutput<F: Field> {
    pub bytecode_r_address: Vec<F>,
    pub booleanity_r_address: Vec<F>,
}

pub(super) fn verify_zk<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    bytecode_address_claims: &JoltRelationClaims<PCS::Field>,
    booleanity_address_claims: &JoltRelationClaims<PCS::Field>,
) -> Result<Stage6AZkOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let address_statements = vec![
        SumcheckStatement::new(
            bytecode_address_claims.sumcheck.rounds,
            bytecode_address_claims.sumcheck.degree,
        ),
        SumcheckStatement::new(
            booleanity_address_claims.sumcheck.rounds,
            booleanity_address_claims.sumcheck.degree,
        ),
    ];
    let address_phase_consistency = BatchedSumcheckVerifier::verify_committed_consistency(
        &address_statements,
        &proof.stages.stage6a_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;
    let committed_program_claims = if checked.precommitted.bytecode.is_some() {
        jolt_claims::protocols::jolt::formulas::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES
    } else {
        0
    };
    let address_phase_output_claims =
        committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
            checked,
            proof: &proof.stages.stage6a_sumcheck_proof,
            proof_label: "stage6a_sumcheck_proof",
            output_claim_count: 2 + committed_program_claims,
            stage: JoltRelationId::BytecodeReadRaf,
        })?;

    let bytecode_address_point = address_phase_consistency
        .try_instance_point(bytecode_address_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?;
    let bytecode_r_address = bytecode_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let booleanity_address_point = address_phase_consistency
        .try_instance_point(booleanity_address_claims.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?;
    let booleanity_r_address = booleanity_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();

    Ok(Stage6AZkOutput {
        address_phase_consistency,
        address_phase_output_claims,
        bytecode_r_address,
        booleanity_r_address,
    })
}

pub(super) fn verify_clear<PCS, VC, T, ZkProof>(
    proof: &JoltProof<PCS, VC, ZkProof>,
    transcript: &mut T,
    claims: &Stage6OutputClaims<PCS::Field>,
    bytecode_relation: &BytecodeReadRafAddressPhase<PCS::Field>,
    bytecode_inputs: &BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<PCS::Field>>,
    booleanity_relation: &BooleanityAddressPhase<PCS::Field>,
) -> Result<Stage6AClearOutput<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let booleanity_inputs = BooleanityAddressPhaseInputClaims::from_upstream();
    let bytecode_read_raf_input = bytecode_relation.input_claim(bytecode_inputs)?;
    let booleanity_input = booleanity_relation.input_claim(&booleanity_inputs)?;
    let bytecode_spec = bytecode_relation.spec();
    let booleanity_spec = booleanity_relation.spec();
    let address_sumcheck_claims = vec![
        SumcheckClaim::new(
            bytecode_spec.rounds,
            bytecode_spec.degree,
            bytecode_read_raf_input,
        ),
        SumcheckClaim::new(
            booleanity_spec.rounds,
            booleanity_spec.degree,
            booleanity_input,
        ),
    ];
    let address_batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &address_sumcheck_claims,
        &proof.stages.stage6a_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: error.to_string(),
    })?;

    let bytecode_address_point = address_batch
        .try_instance_point(bytecode_spec.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        })?
        .to_vec();
    let bytecode_r_address = bytecode_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();
    let booleanity_address_point = address_batch
        .try_instance_point(booleanity_spec.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::Booleanity,
            reason: error.to_string(),
        })?
        .to_vec();
    let booleanity_r_address = booleanity_address_point
        .iter()
        .rev()
        .copied()
        .collect::<Vec<_>>();

    // Pair the produced address-phase openings (wire values + derived points) and
    // check the expected outputs through the relation objects.
    let bytecode_points =
        bytecode_relation.derive_opening_points(&bytecode_address_point, bytecode_inputs)?;
    let bytecode_values = BytecodeReadRafAddressPhaseOutputClaims {
        intermediate: claims.address_phase.bytecode_read_raf,
        val_stages: claims
            .address_phase
            .bytecode_val_stages
            .map_or_else(Vec::new, |stages| stages.to_vec()),
    };
    let bytecode_outputs = zip_openings(&bytecode_values, &bytecode_points);
    let booleanity_points =
        booleanity_relation.derive_opening_points(&booleanity_address_point, &booleanity_inputs)?;
    let booleanity_values = BooleanityAddressPhaseOutputClaims {
        intermediate: claims.address_phase.booleanity,
    };
    let booleanity_outputs = zip_openings(&booleanity_values, &booleanity_points);

    let address_expected_outputs = [
        bytecode_relation.expected_output(bytecode_inputs, &bytecode_outputs)?,
        booleanity_relation.expected_output(&booleanity_inputs, &booleanity_outputs)?,
    ];
    if address_batch.batching_coefficients.len() != address_expected_outputs.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "Stage 6 address batch verifier returned {} coefficients for {} instances",
                address_batch.batching_coefficients.len(),
                address_expected_outputs.len()
            ),
        });
    }
    let expected_final_claim = address_batch
        .batching_coefficients
        .iter()
        .zip(address_expected_outputs)
        .map(|(coefficient, output)| *coefficient * output)
        .sum();
    if address_batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::BytecodeReadRaf,
        });
    }

    append_address_phase_opening_claims(transcript, claims);

    Ok(Stage6AClearOutput {
        bytecode_r_address,
        booleanity_r_address,
    })
}

pub(super) fn append_address_phase_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6OutputClaims<F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append_labeled(b"opening_claim", &claims.address_phase.bytecode_read_raf);
    if let Some(stage_claims) = &claims.address_phase.bytecode_val_stages {
        for opening_claim in stage_claims {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
    }
    transcript.append_labeled(b"opening_claim", &claims.address_phase.booleanity);
}
pub(super) fn aliased_booleanity_bytecode_openings<F: Field>(
    bytecode_ra_opening_points: &[Vec<F>],
    booleanity_opening_point: &[F],
) -> usize {
    bytecode_ra_opening_points
        .iter()
        .filter(|point| point.as_slice() == booleanity_opening_point)
        .count()
}

/// A verified committed-reduction cycle phase: the produced opening point (which
/// stage 8 reads off `output_points` and stage 7 reverses into the cycle-phase
/// variables) and the expected output claim (checked against the batch final
/// claim). The retired `Verified*CyclePhase` structs also carried stage-local
/// sumcheck artifacts that nothing downstream consumed.
pub(super) struct CyclePhaseVerified<F: Field> {
    pub opening_point: Vec<F>,
    pub expected_output_claim: F,
}

pub(super) fn verify_advice_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    opening_claim: &AdviceCyclePhaseOutputClaim<F>,
    stage4: &Stage4ClearOutput<F>,
) -> Result<CyclePhaseVerified<F>, VerifierError> {
    let advice_point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let opening_point = layout
        .cycle_phase_opening_point(advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let contribution = stage4
        .ram_val_check_init
        .advice_contributions
        .iter()
        .find(|contribution| contribution.kind == kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::ram_val_check_advice_opening(kind),
        })?;
    let relation = AdviceCyclePhase::new(kind, layout, contribution.opening.point.clone());
    let inputs = AdviceCyclePhaseInputClaims {
        trusted: (kind == JoltAdviceKind::Trusted).then(|| OpeningClaim {
            point: Vec::new(),
            value: contribution.opening.value,
        }),
        untrusted: (kind == JoltAdviceKind::Untrusted).then(|| OpeningClaim {
            point: Vec::new(),
            value: contribution.opening.value,
        }),
    };
    let derived = relation.derive_opening_points(advice_point, &inputs)?;
    let values = AdviceCyclePhaseOutputClaims {
        trusted: (kind == JoltAdviceKind::Trusted).then_some(opening_claim.opening_claim),
        untrusted: (kind == JoltAdviceKind::Untrusted).then_some(opening_claim.opening_claim),
    };
    let outputs = zip_openings(&values, &derived);
    let expected_output_claim = relation.expected_output(&inputs, &outputs)?;

    Ok(CyclePhaseVerified {
        opening_point,
        expected_output_claim,
    })
}

/// The ZK advice cycle-phase produced opening point, for `Stage6ZkOutput`'s
/// `output_points`. Stage 7 / BlindFold recover `cycle_phase_variables` as
/// `reverse(opening_point)` from there.
pub(super) fn advice_cycle_phase_opening_point<F: Field, C>(
    batch: &jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
) -> Result<Vec<F>, VerifierError> {
    let advice_point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    layout
        .cycle_phase_opening_point(&advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        })
}

pub(crate) struct BytecodeReductionWeightInputs<'a, F: Field> {
    pub eta: F,
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    /// Full bytecode address point (the `BytecodeReadRafAddrClaim` opening).
    pub bytecode_r_address: &'a [F],
}

pub(crate) fn bytecode_reduction_weights<F: Field>(
    layout: &BytecodeClaimReductionLayout,
    inputs: BytecodeReductionWeightInputs<'_, F>,
) -> Result<BytecodeReductionWeights<F>, VerifierError> {
    let address_point = layout
        .split_address_point(inputs.bytecode_r_address)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
            reason: error.to_string(),
        })?;
    let lane_weights = bytecode_reduction::lane_weights(BytecodeLaneWeightInputs {
        eta: inputs.eta,
        stage1_gammas: inputs.stage1_gammas,
        stage2_gammas: inputs.stage2_gammas,
        stage3_gammas: inputs.stage3_gammas,
        stage4_gammas: inputs.stage4_gammas,
        stage5_gammas: inputs.stage5_gammas,
        register_read_write_point: inputs.register_read_write_point,
        register_val_evaluation_point: inputs.register_val_evaluation_point,
    })
    .map_err(|error| VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeClaimReductionCyclePhase,
        reason: error.to_string(),
    })?;
    Ok(BytecodeReductionWeights {
        r_bc: address_point.r_bc,
        chunk_rbc_weights: address_point.chunk_rbc_weights,
        lane_weights,
    })
}

pub(super) fn verify_bytecode_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &BytecodeClaimReductionLayout,
    output_claims: &BytecodeCyclePhaseOutputClaims<F>,
    weights: BytecodeReductionWeights<F>,
    eta: F,
) -> Result<CyclePhaseVerified<F>, VerifierError> {
    let stage = JoltRelationId::BytecodeClaimReductionCyclePhase;
    let point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    let opening_point = layout.cycle_phase_opening_point(point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        }
    })?;
    let has_address_phase = layout.dimensions().has_address_phase();
    let values = match (output_claims, has_address_phase) {
        (BytecodeCyclePhaseOutputClaims::Intermediate(value), true) => {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: Some(*value),
                chunks: Vec::new(),
            }
        }
        (BytecodeCyclePhaseOutputClaims::Chunks(chunks), false)
            if chunks.len() == layout.chunk_count() =>
        {
            BytecodeReductionCyclePhaseOutputClaims {
                intermediate: None,
                chunks: chunks.clone(),
            }
        }
        _ => {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage,
                reason: format!(
                    "bytecode reduction cycle output shape mismatch (address phase: {has_address_phase})"
                ),
            })
        }
    };
    let relation = BytecodeReductionCyclePhase::new(layout, eta, weights.clone());
    let inputs = BytecodeReductionCyclePhaseInputClaims::from_values(Vec::new());
    let derived = relation.derive_opening_points(point, &inputs)?;
    let outputs = zip_openings(&values, &derived);
    let expected_output_claim = relation.expected_output(&inputs, &outputs)?;

    Ok(CyclePhaseVerified {
        opening_point,
        expected_output_claim,
    })
}

/// The ZK committed-reduction (bytecode / program-image) cycle-phase produced
/// opening point, for `Stage6ZkOutput`'s `output_points`. `cycle_phase_variables`
/// are recovered downstream as `reverse(opening_point)`.
pub(super) fn committed_reduction_cycle_phase_opening_point<F: Field, C>(
    batch: &jolt_sumcheck::BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    precommitted: &PrecommittedClaimReduction,
    stage: JoltRelationId,
) -> Result<Vec<F>, VerifierError> {
    let point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    precommitted
        .cycle_phase_opening_point(&point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })
}

pub(super) fn verify_program_image_cycle_phase<F: Field>(
    batch: &jolt_sumcheck::BatchedEvaluationClaim<F>,
    claim: &JoltRelationClaims<F>,
    layout: &ProgramImageClaimReductionLayout,
    output_claim: &ProgramImageCyclePhaseOutputClaim<F>,
    r_addr_rw: &[F],
    input_claim: F,
) -> Result<CyclePhaseVerified<F>, VerifierError> {
    let stage = JoltRelationId::ProgramImageClaimReductionCyclePhase;
    let point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    let opening_point = layout.cycle_phase_opening_point(point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        }
    })?;
    let relation = ProgramImageReductionCyclePhase::new(layout, r_addr_rw.to_vec());
    let inputs = ProgramImageReductionCyclePhaseInputClaims {
        contribution: OpeningClaim {
            point: Vec::new(),
            value: input_claim,
        },
    };
    let derived = relation.derive_opening_points(point, &inputs)?;
    let outputs = zip_openings(
        &ProgramImageReductionCyclePhaseOutputClaims {
            program_image: output_claim.opening_claim,
        },
        &derived,
    );
    let expected_output_claim = relation.expected_output(&inputs, &outputs)?;

    Ok(CyclePhaseVerified {
        opening_point,
        expected_output_claim,
    })
}

pub(super) fn append_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6OutputClaims<F>,
    bytecode_read_raf_points: &[Vec<F>],
    booleanity_point: &[F],
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    // Full relations delegate to their derived `append_openings`, single-sourcing
    // the per-field Fiat-Shamir order from the `OutputClaims` derive. `booleanity`
    // stays explicit because its `bytecode_ra` openings are conditionally deduped
    // against the bytecode-read-RAF points.
    claims.bytecode_read_raf.append_openings(transcript);
    for opening_claim in &claims.booleanity.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for (index, opening_claim) in claims.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_read_raf_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_point)
        {
            continue;
        }
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &claims.booleanity.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    claims.ram_hamming_booleanity.append_openings(transcript);
    claims.ram_ra_virtualization.append_openings(transcript);
    claims
        .instruction_ra_virtualization
        .append_openings(transcript);
    claims.inc_claim_reduction.append_openings(transcript);
    if let Some(opening_claim) = &claims.advice_cycle_phase.trusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(opening_claim) = &claims.advice_cycle_phase.untrusted {
        transcript.append_labeled(b"opening_claim", &opening_claim.opening_claim);
    }
    if let Some(output_claims) = &claims.bytecode_claim_reduction {
        match output_claims {
            BytecodeCyclePhaseOutputClaims::Intermediate(opening_claim) => {
                transcript.append_labeled(b"opening_claim", opening_claim);
            }
            BytecodeCyclePhaseOutputClaims::Chunks(chunks) => {
                for opening_claim in chunks {
                    transcript.append_labeled(b"opening_claim", opening_claim);
                }
            }
        }
    }
    if let Some(output_claim) = &claims.program_image_claim_reduction {
        transcript.append_labeled(b"opening_claim", &output_claim.opening_claim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    #[derive(Clone, Default)]
    struct RecordingTranscript {
        chunks: Vec<Vec<u8>>,
    }

    impl Transcript for RecordingTranscript {
        type Challenge = Fr;
        fn new(_label: &'static [u8]) -> Self {
            Self::default()
        }
        fn append_bytes(&mut self, bytes: &[u8]) {
            self.chunks.push(bytes.to_vec());
        }
        fn challenge(&mut self) -> Self::Challenge {
            Fr::from_u64(0)
        }
        fn state(&self) -> [u8; 32] {
            [0u8; 32]
        }
    }

    /// Locks the stage-6 cycle-phase Fiat-Shamir append order against silent drift.
    /// The full relations are single-sourced via their `OutputClaims` derive;
    /// `booleanity` (conditional `bytecode_ra` dedup) and the optional reductions
    /// stay explicit. Points are empty so no `bytecode_ra` element is deduped;
    /// `address_phase` (absorbed in the address phase) and the `None` reductions
    /// carry distinct/absent sentinels to prove they are not appended here.
    #[test]
    fn append_opening_claims_follows_canonical_order() {
        let claims: Stage6OutputClaims<Fr> = Stage6OutputClaims {
            address_phase: Stage6AddressPhaseClaims {
                bytecode_read_raf: fr(901),
                booleanity: fr(902),
                bytecode_val_stages: None,
            },
            bytecode_read_raf: BytecodeReadRafOutputClaims {
                bytecode_ra: vec![fr(1), fr(2)],
            },
            booleanity: BooleanityOutputClaims {
                instruction_ra: vec![fr(3)],
                bytecode_ra: vec![fr(4)],
                ram_ra: vec![fr(5)],
            },
            ram_hamming_booleanity: RamHammingBooleanityOutputClaims {
                ram_hamming_weight: fr(6),
            },
            ram_ra_virtualization: RamRaVirtualizationOutputClaims {
                ram_ra: vec![fr(7)],
            },
            instruction_ra_virtualization: InstructionRaVirtualizationOutputClaims {
                committed_instruction_ra: vec![fr(8)],
            },
            inc_claim_reduction: IncClaimReductionOutputClaims {
                ram_inc: fr(9),
                rd_inc: fr(10),
            },
            advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
                trusted: None,
                untrusted: None,
            },
            bytecode_claim_reduction: None,
            program_image_claim_reduction: None,
        };

        let mut got = RecordingTranscript::default();
        append_opening_claims(&mut got, &claims, &[], &[]);

        let mut want = RecordingTranscript::default();
        for value in (1..=10).map(fr) {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }
}
