use jolt_claims::protocols::jolt::{
    geometry::{
        booleanity::{self, BooleanityDimensions},
        bytecode::{self, BytecodeReadRafDimensions},
        claim_reductions::{
            advice,
            bytecode::{self as bytecode_reduction, BytecodeLaneWeightInputs},
            program_image,
        },
        dimensions::{committed_address_chunks, JoltFormulaDimensions, REGISTER_ADDRESS_BITS},
        instruction,
    },
    relations, AdviceClaimReductionLayout, BytecodeClaimReductionChallenge,
    BytecodeClaimReductionLayout, BytecodeReadRafChallenge, JoltAdviceKind, JoltChallengeId,
    JoltDerivedId, JoltRelationId, PrecommittedClaimReduction, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout,
};
use jolt_claims::{NoChallenges, SymbolicSumcheck};
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
        booleanity_address_phase_inputs_from_upstream, BooleanityAddressPhase,
        BooleanityAddressPhaseOutputClaims,
    },
    bytecode_read_raf::{
        bytecode_read_raf_address_phase_inputs_from_upstream, BytecodeReadRafAddressPhase,
        BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafAddressPhaseInputClaims,
        BytecodeReadRafAddressPhaseOutputClaims,
    },
    committed_reduction_cycle_phase::{
        bytecode_reduction_cycle_phase_inputs_from_values, AdviceCyclePhase,
        AdviceCyclePhaseInputClaims, AdviceCyclePhaseOutputClaims, BytecodeReductionCyclePhase,
        BytecodeReductionCyclePhaseChallenges, BytecodeReductionCyclePhaseOutputClaims,
        ProgramImageReductionCyclePhase, ProgramImageReductionCyclePhaseInputClaims,
        ProgramImageReductionCyclePhaseOutputClaims,
    },
    outputs::{
        BooleanityOutputClaims, BytecodeReadRafOutputClaims, BytecodeReductionWeights,
        IncClaimReductionOutputClaims, InstructionRaVirtualizationOutputClaims,
        RamHammingBooleanityOutputClaims, RamRaVirtualizationOutputClaims,
        Stage6AddressPhaseOutputClaims, Stage6Challenges, Stage6ClearOutput,
        Stage6CyclePhaseOutputClaims, Stage6Output, Stage6OutputClaims, Stage6ZkOutput,
    },
};
use crate::{
    preprocessing::JoltVerifierPreprocessing,
    proof::JoltProof,
    stages::{
        relations::{
            check_relation_boolean_hypercube, zip_openings, ConcreteSumcheck, OpeningClaim,
            OutputAppend, OutputClaims,
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

    let bytecode_address_rel =
        relations::bytecode::ReadRafAddressPhase::new(formula_dimensions.bytecode_read_raf);
    // The committed and uncommitted cycle-phase relations are distinct types; only
    // the (shape-identical) sumcheck rounds/degree/domain are needed here.
    let booleanity_dimensions = BooleanityDimensions::new(
        formula_dimensions.ra_layout,
        log_t,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let booleanity_address_rel =
        relations::booleanity::BooleanityAddressPhase::new(booleanity_dimensions);
    let booleanity_rel = relations::booleanity::BooleanityCyclePhase::new(booleanity_dimensions);
    let ram_hamming_rel = relations::ram::HammingBooleanity::new(trace_dimensions);
    let ram_ra_rel =
        relations::ram::RaVirtualization::new(formula_dimensions.ram_ra_virtualization);
    let instruction_ra_rel = relations::instruction::RaVirtualization::new(
        formula_dimensions.instruction_ra_virtualization,
    );
    let inc_rel = relations::claim_reductions::increments::ClaimReduction::new(trace_dimensions);

    // Bytecode cycle-phase committed/uncommitted are distinct types with the same
    // rounds/degree/domain; collapse the active one to its shape values here.
    let (bytecode_rounds, bytecode_degree, bytecode_domain) = if committed_program {
        let r = relations::bytecode::ReadRafCyclePhaseCommitted::new(
            formula_dimensions.bytecode_read_raf,
        );
        (r.rounds(), r.degree(), r.domain())
    } else {
        let r = relations::bytecode::ReadRafCyclePhase::new(formula_dimensions.bytecode_read_raf);
        (r.rounds(), r.degree(), r.domain())
    };

    let trusted_advice_layout = checked.precommitted.trusted_advice.as_ref();
    let untrusted_advice_layout = checked.precommitted.untrusted_advice.as_ref();

    let trusted_advice_claims = trusted_advice_layout.map(|layout| {
        relations::claim_reductions::advice::CyclePhase::new((
            JoltAdviceKind::Trusted,
            layout.dimensions(),
        ))
    });
    let untrusted_advice_claims = untrusted_advice_layout.map(|layout| {
        relations::claim_reductions::advice::CyclePhase::new((
            JoltAdviceKind::Untrusted,
            layout.dimensions(),
        ))
    });

    let bytecode_reduction_claims = bytecode_reduction_layout.map(|layout| {
        relations::claim_reductions::bytecode::CyclePhase::new((
            layout.dimensions(),
            layout.chunk_count(),
        ))
    });
    let program_image_reduction_claims = program_image_reduction_layout.map(|layout| {
        relations::claim_reductions::program_image::CyclePhase::new(layout.dimensions())
    });

    for (relation, domain, degree) in [
        (
            relations::bytecode::ReadRafAddressPhase::id(),
            bytecode_address_rel.domain(),
            bytecode_address_rel.degree(),
        ),
        (
            relations::bytecode::ReadRafCyclePhase::id(),
            bytecode_domain,
            bytecode_degree,
        ),
        (
            relations::booleanity::BooleanityAddressPhase::id(),
            booleanity_address_rel.domain(),
            booleanity_address_rel.degree(),
        ),
        (
            relations::booleanity::BooleanityCyclePhase::id(),
            booleanity_rel.domain(),
            booleanity_rel.degree(),
        ),
        (
            relations::ram::HammingBooleanity::id(),
            ram_hamming_rel.domain(),
            ram_hamming_rel.degree(),
        ),
        (
            relations::ram::RaVirtualization::id(),
            ram_ra_rel.domain(),
            ram_ra_rel.degree(),
        ),
        (
            relations::instruction::RaVirtualization::id(),
            instruction_ra_rel.domain(),
            instruction_ra_rel.degree(),
        ),
        (
            relations::claim_reductions::increments::ClaimReduction::id(),
            inc_rel.domain(),
            inc_rel.degree(),
        ),
    ] {
        check_relation_boolean_hypercube(relation, domain, degree)?;
    }
    for (relation, domain, degree) in [
        (
            relations::claim_reductions::advice::CyclePhase::id(),
            trusted_advice_claims
                .as_ref()
                .map(|r| (r.domain(), r.degree())),
        ),
        (
            relations::claim_reductions::advice::CyclePhase::id(),
            untrusted_advice_claims
                .as_ref()
                .map(|r| (r.domain(), r.degree())),
        ),
        (
            relations::claim_reductions::bytecode::CyclePhase::id(),
            bytecode_reduction_claims
                .as_ref()
                .map(|r| (r.domain(), r.degree())),
        ),
        (
            relations::claim_reductions::program_image::CyclePhase::id(),
            program_image_reduction_claims
                .as_ref()
                .map(|r| (r.domain(), r.degree())),
        ),
    ]
    .into_iter()
    .filter_map(|(relation, opt)| opt.map(|(domain, degree)| (relation, domain, degree)))
    {
        check_relation_boolean_hypercube(relation, domain, degree)?;
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

    let challenges = |instruction_ra_gamma_powers: Vec<PCS::Field>,
                      inc_gamma: PCS::Field,
                      eta: Option<PCS::Field>| Stage6Challenges {
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
            bytecode_address_rel.rounds(),
            bytecode_address_rel.degree(),
            booleanity_address_rel.rounds(),
            booleanity_address_rel.degree(),
        )?;

        let instruction_ra_gamma_powers = transcript.challenge_scalar_powers(
            formula_dimensions
                .instruction_ra_virtualization
                .num_virtual_ra_polys(),
        );
        let inc_gamma = transcript.challenge_scalar();
        let eta = committed_program.then(|| transcript.challenge_scalar());

        let mut statements = vec![
            SumcheckStatement::new(bytecode_rounds, bytecode_degree),
            SumcheckStatement::new(booleanity_rel.rounds(), booleanity_rel.degree()),
            SumcheckStatement::new(ram_hamming_rel.rounds(), ram_hamming_rel.degree()),
            SumcheckStatement::new(ram_ra_rel.rounds(), ram_ra_rel.degree()),
            SumcheckStatement::new(instruction_ra_rel.rounds(), instruction_ra_rel.degree()),
            SumcheckStatement::new(inc_rel.rounds(), inc_rel.degree()),
        ];
        if let Some(claim) = &trusted_advice_claims {
            statements.push(SumcheckStatement::new(claim.rounds(), claim.degree()));
        }
        if let Some(claim) = &untrusted_advice_claims {
            statements.push(SumcheckStatement::new(claim.rounds(), claim.degree()));
        }
        if let Some(claim) = &bytecode_reduction_claims {
            statements.push(SumcheckStatement::new(claim.rounds(), claim.degree()));
        }
        if let Some(claim) = &program_image_reduction_claims {
            statements.push(SumcheckStatement::new(claim.rounds(), claim.degree()));
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
            .try_instance_point(bytecode_rounds)
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
            .try_instance_point(booleanity_rel.rounds())
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
            .try_instance_point(ram_hamming_rel.rounds())
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
            .try_instance_point(ram_ra_rel.rounds())
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
            .try_instance_point(instruction_ra_rel.rounds())
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
            .try_instance_point(inc_rel.rounds())
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
                claim.rounds(),
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
                claim.rounds(),
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
                claim.rounds(),
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
                claim.rounds(),
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
        let ram_ra_output_openings = RamRaVirtualizationOutputClaims::<PCS::Field> {
            ram_ra: vec![
                PCS::Field::zero();
                formula_dimensions
                    .ram_ra_virtualization
                    .num_committed_ra_polys()
            ],
        }
        .canonical_order();
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
            address_phase: Stage6AddressPhaseOutputClaims {
                bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
                    intermediate: stage6a.bytecode_r_address.clone(),
                    val_stages: if committed_program {
                        vec![
                            stage6a.bytecode_r_address.clone();
                            bytecode_reduction::NUM_BYTECODE_VAL_STAGES
                        ]
                    } else {
                        Vec::new()
                    },
                },
                booleanity: BooleanityAddressPhaseOutputClaims {
                    intermediate: stage6a.booleanity_r_address.clone(),
                },
            },
            cycle_phase: Stage6CyclePhaseOutputClaims {
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
                trusted_advice: trusted_advice.map(|opening_point| AdviceCyclePhaseOutputClaims {
                    trusted: Some(opening_point),
                    untrusted: None,
                }),
                untrusted_advice: untrusted_advice.map(|opening_point| {
                    AdviceCyclePhaseOutputClaims {
                        trusted: None,
                        untrusted: Some(opening_point),
                    }
                }),
                bytecode_reduction: match (bytecode_reduction_layout, bytecode_cycle_phase) {
                    (Some(layout), Some(opening_point)) => {
                        Some(if layout.dimensions().has_address_phase() {
                            BytecodeReductionCyclePhaseOutputClaims {
                                intermediate: Some(opening_point),
                                chunks: Vec::new(),
                            }
                        } else {
                            BytecodeReductionCyclePhaseOutputClaims {
                                intermediate: None,
                                chunks: vec![opening_point; layout.chunk_count()],
                            }
                        })
                    }
                    _ => None,
                },
                program_image_reduction: program_image_cycle_phase.map(|program_image| {
                    ProgramImageReductionCyclePhaseOutputClaims { program_image }
                }),
            },
        };

        return Ok(Stage6Output::Zk(Stage6ZkOutput {
            challenges: challenges(instruction_ra_gamma_powers, inc_gamma, eta),
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
    let has_val_stages = !claims.address_phase.bytecode_read_raf.val_stages.is_empty();
    if committed_program != has_val_stages {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode Val-stage claims presence ({has_val_stages}) does not match committed program mode ({committed_program})"
            ),
        });
    }

    // The bytecode address-phase input claim is the gamma-folded bind of every
    // prior clear stage opening; the relation evaluates it through its input `Expr`
    // from these wired openings + the per-stage folding gammas.
    let bytecode_address_inputs = bytecode_read_raf_address_phase_inputs_from_upstream(
        stage1, stage2, stage3, stage4, stage5,
    )?;
    let num_bytecode_val_stages = if committed_program {
        bytecode_reduction::NUM_BYTECODE_VAL_STAGES
    } else {
        0
    };
    let bytecode_address_relation = BytecodeReadRafAddressPhase::new(
        formula_dimensions.bytecode_read_raf,
        num_bytecode_val_stages,
    );
    // The 6-field address-phase challenge set: the bytecode gamma plus the five
    // per-stage folding gammas, each the degree-1 power of its inline
    // `challenge_scalar_powers` draw (value-equal to the default `draw_challenges`).
    let bytecode_address_challenges = BytecodeReadRafAddressPhaseChallenges {
        gamma: bytecode_gamma,
        stage1_gamma: stage1_gammas[1],
        stage2_gamma: stage2_gammas[1],
        stage3_gamma: stage3_gammas[1],
        stage4_gamma: stage4_gammas[1],
        stage5_gamma: stage5_gammas[1],
    };
    let booleanity_address_relation = BooleanityAddressPhase::new(booleanity_dimensions);

    if trusted_advice_claims.is_none() && claims.cycle_phase.trusted_advice.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if untrusted_advice_claims.is_none() && claims.cycle_phase.untrusted_advice.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::cycle_phase_advice_opening(JoltAdviceKind::Untrusted),
        });
    }
    if bytecode_reduction_claims.is_none() && claims.cycle_phase.bytecode_reduction.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        });
    }
    if program_image_reduction_claims.is_none()
        && claims.cycle_phase.program_image_reduction.is_some()
    {
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
        &bytecode_address_challenges,
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
            address_bytecode_read_raf: claims.address_phase.bytecode_read_raf.intermediate,
            address_booleanity: claims.address_phase.booleanity.intermediate,
            address_val_stages: claims.address_phase.bytecode_read_raf.val_stages.clone(),
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

    let bytecode_point = batch.try_instance_point(bytecode_rounds).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: error.to_string(),
        }
    })?;
    let bytecode_r_cycle = bytecode_point.iter().rev().copied().collect::<Vec<_>>();
    let bytecode_output_openings =
        bytecode::read_raf_output_openings(formula_dimensions.bytecode_read_raf);
    if claims.cycle_phase.bytecode_read_raf.bytecode_ra.len()
        != bytecode_output_openings.bytecode_ra.len()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                bytecode_output_openings.bytecode_ra.len(),
                claims.cycle_phase.bytecode_read_raf.bytecode_ra.len()
            ),
        });
    }

    let bytecode_points = relations
        .bytecode_read_raf
        .derive_opening_points(bytecode_point, &relations.bytecode_read_raf_inputs)?;
    let bytecode_outputs = zip_openings(&claims.cycle_phase.bytecode_read_raf, &bytecode_points);
    // Relations that draw no challenges resolve against this empty set; the others
    // use the per-relation challenge structs the bundle built from the drawn gammas.
    let no_challenges = NoChallenges::default();
    let bytecode_output = relations.bytecode_read_raf.expected_output(
        &relations.bytecode_read_raf_inputs,
        &bytecode_outputs,
        relations.bytecode_gamma,
    )?;
    let bytecode_ra_opening_points = proof
        .one_hot_config
        .committed_address_chunks(&bytecode_r_address)
        .into_iter()
        .map(|r_address_chunk| [r_address_chunk.as_slice(), bytecode_r_cycle.as_slice()].concat())
        .collect::<Vec<_>>();

    let booleanity_point = batch
        .try_instance_point(booleanity_rel.rounds())
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
    let booleanity_outputs = zip_openings(&claims.cycle_phase.booleanity, &booleanity_points);
    let booleanity_output = relations.booleanity.expected_output(
        &relations.booleanity_inputs,
        &booleanity_outputs,
        &relations.booleanity_challenges,
    )?;
    let ram_hamming_point =
        batch
            .try_instance_point(ram_hamming_rel.rounds())
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::RamHammingBooleanity,
                reason: error.to_string(),
            })?;
    let ram_hamming_points = relations
        .ram_hamming
        .derive_opening_points(ram_hamming_point, &relations.ram_hamming_inputs)?;
    let ram_hamming_outputs = zip_openings(
        &claims.cycle_phase.ram_hamming_booleanity,
        &ram_hamming_points,
    );
    let ram_hamming_output = relations.ram_hamming.expected_output(
        &relations.ram_hamming_inputs,
        &ram_hamming_outputs,
        &no_challenges,
    )?;

    let ram_ra_point = batch
        .try_instance_point(ram_ra_rel.rounds())
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::RamRaVirtualization,
            reason: error.to_string(),
        })?;
    let ram_ra_points = relations
        .ram_ra
        .derive_opening_points(ram_ra_point, &relations.ram_ra_inputs)?;
    let ram_ra_outputs = zip_openings(&claims.cycle_phase.ram_ra_virtualization, &ram_ra_points);
    let ram_ra_output = relations.ram_ra.expected_output(
        &relations.ram_ra_inputs,
        &ram_ra_outputs,
        &no_challenges,
    )?;

    let instruction_ra_point = batch
        .try_instance_point(instruction_ra_rel.rounds())
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::InstructionRaVirtualization,
            reason: error.to_string(),
        })?;
    let instruction_ra_points = relations
        .instruction_ra
        .derive_opening_points(instruction_ra_point, &relations.instruction_ra_inputs)?;
    let instruction_ra_outputs = zip_openings(
        &claims.cycle_phase.instruction_ra_virtualization,
        &instruction_ra_points,
    );
    let instruction_ra_output = relations.instruction_ra.expected_output(
        &relations.instruction_ra_inputs,
        &instruction_ra_outputs,
        &relations.instruction_ra_challenges,
    )?;

    let inc_point = batch
        .try_instance_point(inc_rel.rounds())
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::IncClaimReduction,
            reason: error.to_string(),
        })?;
    let inc_points = relations
        .inc
        .derive_opening_points(inc_point, &relations.inc_inputs)?;
    let inc_outputs = zip_openings(&claims.cycle_phase.inc_claim_reduction, &inc_points);
    let inc_output = relations.inc.expected_output(
        &relations.inc_inputs,
        &inc_outputs,
        &relations.inc_challenges,
    )?;

    let trusted_advice = if let (Some(layout), Some(claim), Some(opening_claim)) = (
        trusted_advice_layout,
        trusted_advice_claims.as_ref(),
        claims.cycle_phase.trusted_advice.as_ref(),
    ) {
        Some(verify_advice_cycle_phase(
            &batch,
            claim.rounds(),
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
        claims.cycle_phase.untrusted_advice.as_ref(),
    ) {
        Some(verify_advice_cycle_phase(
            &batch,
            claim.rounds(),
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
    let bytecode_cycle_phase = if let (Some(layout), Some(claim)) = (
        bytecode_reduction_layout,
        bytecode_reduction_claims.as_ref(),
    ) {
        let output_claims = claims.cycle_phase.bytecode_reduction.as_ref().ok_or(
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
            claim.rounds(),
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
        let output_claim = claims.cycle_phase.program_image_reduction.as_ref().ok_or(
            VerifierError::MissingOpeningClaim {
                id: program_image::cycle_phase_output_openings(layout.dimensions())[0],
            },
        )?;
        let r_addr_rw = &stage4.output_claims.ram_val_check.ram_ra.point[..log_k];
        let input_claim = match (
            &relations.program_image_reduction,
            &relations.program_image_reduction_inputs,
        ) {
            (Some(relation), Some(inputs)) => relation.input_claim(inputs, &no_challenges)?,
            _ => {
                return Err(VerifierError::MissingOpeningClaim {
                    id: program_image::ram_val_check_contribution_opening(),
                })
            }
        };
        Some(verify_program_image_cycle_phase(
            &batch,
            claim.rounds(),
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
        address_phase: Stage6AddressPhaseOutputClaims {
            bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
                intermediate: bytecode_r_address.clone(),
                val_stages: vec![
                    bytecode_r_address.clone();
                    claims.address_phase.bytecode_read_raf.val_stages.len()
                ],
            },
            booleanity: BooleanityAddressPhaseOutputClaims {
                intermediate: booleanity_r_address.clone(),
            },
        },
        cycle_phase: Stage6CyclePhaseOutputClaims {
            bytecode_read_raf: bytecode_points,
            booleanity: booleanity_points,
            ram_hamming_booleanity: ram_hamming_points,
            ram_ra_virtualization: ram_ra_points,
            instruction_ra_virtualization: instruction_ra_points,
            inc_claim_reduction: inc_points,
            trusted_advice: trusted_advice
                .as_ref()
                .map(|verified| AdviceCyclePhaseOutputClaims {
                    trusted: Some(verified.opening_point.clone()),
                    untrusted: None,
                }),
            untrusted_advice: untrusted_advice.as_ref().map(|verified| {
                AdviceCyclePhaseOutputClaims {
                    trusted: None,
                    untrusted: Some(verified.opening_point.clone()),
                }
            }),
            bytecode_reduction: bytecode_cycle_phase.as_ref().map(|verified| {
                let is_chunks = claims
                    .cycle_phase
                    .bytecode_reduction
                    .as_ref()
                    .is_some_and(|reduction| reduction.intermediate.is_none());
                if is_chunks {
                    let chunk_count = claims
                        .cycle_phase
                        .bytecode_reduction
                        .as_ref()
                        .map_or(0, |reduction| reduction.chunks.len());
                    BytecodeReductionCyclePhaseOutputClaims {
                        intermediate: None,
                        chunks: vec![verified.opening_point.clone(); chunk_count],
                    }
                } else {
                    BytecodeReductionCyclePhaseOutputClaims {
                        intermediate: Some(verified.opening_point.clone()),
                        chunks: Vec::new(),
                    }
                }
            }),
            program_image_reduction: program_image_cycle_phase.as_ref().map(|verified| {
                ProgramImageReductionCyclePhaseOutputClaims {
                    program_image: verified.opening_point.clone(),
                }
            }),
        },
    };

    Ok(Stage6Output::Clear(Stage6ClearOutput {
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
    let relation = relations::bytecode::ReadRaf::new(inputs.dimensions);
    relation.output_expression::<F>().try_evaluate(
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
            JoltDerivedId::BytecodeReadRaf(public_id) => inputs
                .public_values
                .value(*public_id)
                .ok_or(VerifierError::MissingStageClaimDerived { id: *id }),
            _ => Err(VerifierError::MissingStageClaimDerived { id: *id }),
        },
    )
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
    bytecode_address_rounds: usize,
    bytecode_address_degree: usize,
    booleanity_address_rounds: usize,
    booleanity_address_degree: usize,
) -> Result<Stage6AZkOutput<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let address_statements = vec![
        SumcheckStatement::new(bytecode_address_rounds, bytecode_address_degree),
        SumcheckStatement::new(booleanity_address_rounds, booleanity_address_degree),
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
        jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES
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
        .try_instance_point(bytecode_address_rounds)
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
        .try_instance_point(booleanity_address_rounds)
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
    claims: &Stage6OutputClaims<PCS::Field, PCS::Field>,
    bytecode_relation: &BytecodeReadRafAddressPhase<PCS::Field>,
    bytecode_inputs: &BytecodeReadRafAddressPhaseInputClaims<OpeningClaim<PCS::Field>>,
    bytecode_challenges: &BytecodeReadRafAddressPhaseChallenges<PCS::Field>,
    booleanity_relation: &BooleanityAddressPhase<PCS::Field>,
) -> Result<Stage6AClearOutput<PCS::Field>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let no_challenges = NoChallenges::default();
    let booleanity_inputs = booleanity_address_phase_inputs_from_upstream();
    let bytecode_read_raf_input =
        bytecode_relation.input_claim(bytecode_inputs, bytecode_challenges)?;
    let booleanity_input = booleanity_relation.input_claim(&booleanity_inputs, &no_challenges)?;
    let address_sumcheck_claims = vec![
        SumcheckClaim::new(
            bytecode_relation.rounds(),
            bytecode_relation.degree(),
            bytecode_read_raf_input,
        ),
        SumcheckClaim::new(
            booleanity_relation.rounds(),
            booleanity_relation.degree(),
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
        .try_instance_point(bytecode_relation.rounds())
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
        .try_instance_point(booleanity_relation.rounds())
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
    let bytecode_outputs = zip_openings(&claims.address_phase.bytecode_read_raf, &bytecode_points);
    let booleanity_points =
        booleanity_relation.derive_opening_points(&booleanity_address_point, &booleanity_inputs)?;
    let booleanity_outputs = zip_openings(&claims.address_phase.booleanity, &booleanity_points);

    let address_expected_outputs = [
        bytecode_relation.expected_output(
            bytecode_inputs,
            &bytecode_outputs,
            bytecode_challenges,
        )?,
        booleanity_relation.expected_output(
            &booleanity_inputs,
            &booleanity_outputs,
            &no_challenges,
        )?,
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
    claims: &Stage6OutputClaims<F, F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    // The address-phase order (bytecode `intermediate`, each `val_stages`, then
    // booleanity `intermediate`) is single-sourced from the generated
    // `Stage6AddressPhaseOutputClaims::append_to_transcript` (member declaration
    // order = canonical Fiat-Shamir order; no alias dedup in the address phase).
    claims.address_phase.append_to_transcript(transcript);
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
    rounds: usize,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    opening_claim: &AdviceCyclePhaseOutputClaims<F>,
    stage4: &Stage4ClearOutput<F>,
) -> Result<CyclePhaseVerified<F>, VerifierError> {
    let opening_value = match kind {
        JoltAdviceKind::Trusted => opening_claim.trusted,
        JoltAdviceKind::Untrusted => opening_claim.untrusted,
    }
    .ok_or(VerifierError::MissingOpeningClaim {
        id: advice::cycle_phase_advice_opening(kind),
    })?;
    let advice_point = batch.try_instance_point_at(0, rounds).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        }
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
        trusted: (kind == JoltAdviceKind::Trusted).then_some(opening_value),
        untrusted: (kind == JoltAdviceKind::Untrusted).then_some(opening_value),
    };
    let outputs = zip_openings(&values, &derived);
    let expected_output_claim =
        relation.expected_output(&inputs, &outputs, &NoChallenges::default())?;

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
    rounds: usize,
    layout: &AdviceClaimReductionLayout,
) -> Result<Vec<F>, VerifierError> {
    let advice_point = batch.try_instance_point_at(0, rounds).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReductionCyclePhase,
            reason: error.to_string(),
        }
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
    rounds: usize,
    layout: &BytecodeClaimReductionLayout,
    output_claims: &BytecodeReductionCyclePhaseOutputClaims<F>,
    weights: BytecodeReductionWeights<F>,
    eta: F,
) -> Result<CyclePhaseVerified<F>, VerifierError> {
    let stage = JoltRelationId::BytecodeClaimReductionCyclePhase;
    let point = batch.try_instance_point_at(0, rounds).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        }
    })?;
    let opening_point = layout.cycle_phase_opening_point(point).map_err(|error| {
        VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        }
    })?;
    let has_address_phase = layout.dimensions().has_address_phase();
    // The wire shape must match the reduction mode: an `intermediate` (no chunks)
    // when an address phase follows, else exactly `chunk_count` chunks (no
    // intermediate).
    let shape_ok = match (
        &output_claims.intermediate,
        output_claims.chunks.is_empty(),
        has_address_phase,
    ) {
        (Some(_), true, true) => true,
        (None, false, false) => output_claims.chunks.len() == layout.chunk_count(),
        _ => false,
    };
    if !shape_ok {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: format!(
                "bytecode reduction cycle output shape mismatch (address phase: {has_address_phase})"
            ),
        });
    }
    let values = output_claims.clone();
    let relation = BytecodeReductionCyclePhase::new(layout, weights.clone());
    let challenges = BytecodeReductionCyclePhaseChallenges { eta };
    let inputs = bytecode_reduction_cycle_phase_inputs_from_values(Vec::new());
    let derived = relation.derive_opening_points(point, &inputs)?;
    let outputs = zip_openings(&values, &derived);
    let expected_output_claim = relation.expected_output(&inputs, &outputs, &challenges)?;

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
    rounds: usize,
    precommitted: &PrecommittedClaimReduction,
    stage: JoltRelationId,
) -> Result<Vec<F>, VerifierError> {
    let point = batch.try_instance_point_at(0, rounds).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        }
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
    rounds: usize,
    layout: &ProgramImageClaimReductionLayout,
    output_claim: &ProgramImageReductionCyclePhaseOutputClaims<F>,
    r_addr_rw: &[F],
    input_claim: F,
) -> Result<CyclePhaseVerified<F>, VerifierError> {
    let stage = JoltRelationId::ProgramImageClaimReductionCyclePhase;
    let point = batch.try_instance_point_at(0, rounds).map_err(|error| {
        VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        }
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
            program_image: output_claim.program_image,
        },
        &derived,
    );
    let expected_output_claim =
        relation.expected_output(&inputs, &outputs, &NoChallenges::default())?;

    Ok(CyclePhaseVerified {
        opening_point,
        expected_output_claim,
    })
}

pub(super) fn append_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage6OutputClaims<F, F>,
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
    let cycle = &claims.cycle_phase;
    cycle.bytecode_read_raf.append_openings(transcript);
    for opening_claim in &cycle.booleanity.instruction_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for (index, opening_claim) in cycle.booleanity.bytecode_ra.iter().enumerate() {
        if bytecode_read_raf_points
            .get(index)
            .is_some_and(|point| point.as_slice() == booleanity_point)
        {
            continue;
        }
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    for opening_claim in &cycle.booleanity.ram_ra {
        transcript.append_labeled(b"opening_claim", opening_claim);
    }
    cycle.ram_hamming_booleanity.append_openings(transcript);
    cycle.ram_ra_virtualization.append_openings(transcript);
    cycle
        .instruction_ra_virtualization
        .append_openings(transcript);
    cycle.inc_claim_reduction.append_openings(transcript);
    if let Some(advice) = &cycle.trusted_advice {
        if let Some(opening_claim) = &advice.trusted {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
    }
    if let Some(advice) = &cycle.untrusted_advice {
        if let Some(opening_claim) = &advice.untrusted {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
    }
    if let Some(reduction) = &cycle.bytecode_reduction {
        if let Some(opening_claim) = &reduction.intermediate {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
        for opening_claim in &reduction.chunks {
            transcript.append_labeled(b"opening_claim", opening_claim);
        }
    }
    if let Some(reduction) = &cycle.program_image_reduction {
        transcript.append_labeled(b"opening_claim", &reduction.program_image);
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

    fn sample_claims() -> Stage6OutputClaims<Fr, Fr> {
        Stage6OutputClaims {
            address_phase: Stage6AddressPhaseOutputClaims {
                bytecode_read_raf: BytecodeReadRafAddressPhaseOutputClaims {
                    intermediate: fr(901),
                    val_stages: Vec::new(),
                },
                booleanity: BooleanityAddressPhaseOutputClaims {
                    intermediate: fr(902),
                },
            },
            cycle_phase: Stage6CyclePhaseOutputClaims {
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
                trusted_advice: None,
                untrusted_advice: None,
                bytecode_reduction: None,
                program_image_reduction: None,
            },
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
        let claims = sample_claims();

        let mut got = RecordingTranscript::default();
        append_opening_claims(&mut got, &claims, &[], &[]);

        let mut want = RecordingTranscript::default();
        for value in (1..=10).map(fr) {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }

    /// Locks the stage-6a address-phase Fiat-Shamir append order against silent
    /// drift: bytecode read-RAF `intermediate`, each `val_stages` entry, then
    /// booleanity `intermediate`. Single-sourced from the generated
    /// `Stage6AddressPhaseOutputClaims::append_to_transcript`.
    #[test]
    fn append_address_phase_opening_claims_follows_canonical_order() {
        let mut claims = sample_claims();
        claims.address_phase.bytecode_read_raf.val_stages = vec![fr(903), fr(904)];

        let mut got = RecordingTranscript::default();
        append_address_phase_opening_claims(&mut got, &claims);

        let mut want = RecordingTranscript::default();
        for value in [fr(901), fr(903), fr(904), fr(902)] {
            want.append_labeled(b"opening_claim", &value);
        }

        assert_eq!(got.chunks, want.chunks);
    }
}
