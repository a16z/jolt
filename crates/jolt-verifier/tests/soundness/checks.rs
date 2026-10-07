//! Backend-generic soundness checks over a fixture's argument string and
//! statement. Each backend module runs them over its own fixtures.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "tamper checks fail loudly when a fixture breaks an assumption"
)]

use std::sync::Arc;

use common::jolt_device::JoltDevice;
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_openings::CommitmentScheme;
use jolt_transcript::TranscriptError;
use jolt_verifier::{
    JoltProof, JoltVerifierPreprocessing, ProgramPreprocessing, ProofHeader, VerifierError,
};

use crate::support::narg::{
    assert_coverage_closure, assert_truncations_reject, decode_header, message_in, sweep,
    with_header, CommitmentOf, Region, TracedCase,
};
use jolt_crypto::VectorCommitment;
use jolt_transcript::VerifierTranscript;
use jolt_verifier::CommittedProgramPreprocessing;
use jolt_verifier::JoltProtocolConfig;
use jolt_verifier::JoltSponge;
use jolt_verifier::JOLT_SESSION;

/// Coverage closure, then the single-byte message sweep with deadlines.
/// `budget` caps the number of tampers (`None`: every tamper).
pub fn message_sweep<C: TracedCase>(case: &C, budget: Option<usize>) {
    let honest = case.honest_trace();
    assert_coverage_closure(&honest, case.proof().narg.len());
    let report = sweep(case, budget);
    assert!(report.tampers > 0, "the sweep exercised no tamper");
}

/// Truncation at message boundaries, trailing bytes, an empty argument
/// string, and a mismatched protocol field.
pub fn structural_tampers<C: TracedCase>(case: &C, truncation_budget: Option<usize>) {
    assert_truncations_reject(case, truncation_budget);

    let mut trailing = case.proof().narg.clone();
    trailing.push(0);
    assert!(
        matches!(
            case.verify_proof(&case.with_narg(trailing)),
            Err(VerifierError::Transcript(TranscriptError::TrailingBytes))
        ),
        "a trailing byte must fail the exact-consumption check"
    );

    assert!(
        matches!(
            case.verify_proof(&case.with_narg(Vec::new())),
            Err(VerifierError::Transcript(TranscriptError::Truncated))
        ),
        "an empty argument string must fail at the proof header"
    );

    for protocol in wrong_protocols(case.proof()) {
        let proof = JoltProof {
            protocol,
            narg: case.proof().narg.clone(),
        };
        assert!(
            matches!(
                case.verify_proof(&proof),
                Err(VerifierError::ProtocolConfigMismatch { .. })
            ),
            "a proof declaring {protocol:?} must fail the protocol-config check"
        );
    }
}

fn wrong_protocols(proof: &JoltProof) -> Vec<JoltProtocolConfig> {
    use jolt_verifier::config::CommitmentConfig;
    use jolt_verifier::ZkConfig;

    let mut zk = proof.protocol;
    zk.zk = match zk.zk {
        ZkConfig::Transparent => ZkConfig::BlindFold,
        ZkConfig::BlindFold => ZkConfig::Transparent,
    };
    let mut commitment = proof.protocol;
    commitment.commitment = match commitment.commitment {
        CommitmentConfig::Homomorphic => CommitmentConfig::Packed,
        CommitmentConfig::Packed => CommitmentConfig::Homomorphic,
    };
    vec![zk, commitment]
}

/// Runs `narg` against a mutated statement and requires rejection no later
/// than the region whose public absorptions the mutation changes. Returns
/// the error for callers that pin a typed rejection.
#[must_use = "the typed rejection is the caller's to pin or drop"]
fn statement_rejection<C: TracedCase>(
    label: &str,
    preprocessing: &JoltVerifierPreprocessing<C::Pcs, C::Vc>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&CommitmentOf<C>>,
    proof: &JoltProof,
    deadline: Region,
) -> VerifierError {
    let error = C::verify_statement(preprocessing, public_io, proof, trusted_advice_commitment)
        .expect_err(label);
    let trace = C::trace_statement(
        preprocessing,
        public_io,
        trusted_advice_commitment,
        &proof.narg,
    );
    assert!(trace.result.is_err(), "{label}: traced replay accepted");
    assert!(
        trace.stop_region() <= deadline,
        "{label}: rejected in {:?}, after its {deadline:?} deadline: {:?}",
        trace.stop_region(),
        trace.result
    );
    error
}

fn assert_statement_rejects<C: TracedCase>(
    label: &str,
    preprocessing: &JoltVerifierPreprocessing<C::Pcs, C::Vc>,
    public_io: &JoltDevice,
    trusted_advice_commitment: Option<&CommitmentOf<C>>,
    proof: &JoltProof,
    deadline: Region,
) {
    let _rejection = statement_rejection::<C>(
        label,
        preprocessing,
        public_io,
        trusted_advice_commitment,
        proof,
        deadline,
    );
}

/// Public-statement tampers: the verifier absorbs the statement (the
/// preprocessing digest, memory layout bounds, public I/O, and entry
/// address) in the preamble, so each mutation must reject by the preamble's
/// deadline, and a statement outside the preprocessing's bounds must reject
/// with its typed error.
pub fn statement_tampers<C: TracedCase>(case: &C) {
    let preprocessing = case.preprocessing();
    let public_io = case.public_io();
    let trusted = case.trusted_advice_commitment();
    let proof = case.proof();
    let deadline = Region::statement_deadline();
    let check = |label: &str,
                 preprocessing: &JoltVerifierPreprocessing<C::Pcs, C::Vc>,
                 public_io: &JoltDevice| {
        assert_statement_rejects::<C>(label, preprocessing, public_io, trusted, proof, deadline);
    };
    let typed = |label: &str,
                 preprocessing: &JoltVerifierPreprocessing<C::Pcs, C::Vc>,
                 public_io: &JoltDevice| {
        statement_rejection::<C>(label, preprocessing, public_io, trusted, proof, deadline)
    };

    let mut io = public_io.clone();
    io.inputs
        .first_mut()
        .map(|byte| *byte ^= 1)
        .expect("fixture has public inputs");
    check("flipped public input byte", preprocessing, &io);

    let mut io = public_io.clone();
    io.inputs.push(1);
    check("appended public input byte", preprocessing, &io);

    let mut io = public_io.clone();
    io.outputs
        .first_mut()
        .map(|byte| *byte ^= 1)
        .expect("fixture has public outputs");
    check("flipped public output byte", preprocessing, &io);

    let mut io = public_io.clone();
    io.panic = !io.panic;
    check("flipped panic bit", preprocessing, &io);

    let layout = preprocessing.program.memory_layout();
    let mut io = public_io.clone();
    io.inputs = vec![0; usize::try_from(layout.max_input_size).expect("fits") + 1];
    assert!(matches!(
        typed("oversized public input", preprocessing, &io),
        VerifierError::InputTooLarge { .. }
    ));

    let mut io = public_io.clone();
    io.outputs = vec![1; usize::try_from(layout.max_output_size).expect("fits") + 1];
    assert!(matches!(
        typed("oversized public output", preprocessing, &io),
        VerifierError::OutputTooLarge { .. }
    ));

    let mut io = public_io.clone();
    io.memory_layout.heap_size += 1;
    assert!(matches!(
        typed("public memory layout mismatch", preprocessing, &io),
        VerifierError::MemoryLayoutMismatch
    ));

    let mut tampered = preprocessing.clone();
    tampered.preprocessing_digest[0] ^= 1;
    check("flipped preprocessing digest", &tampered, public_io);

    if let ProgramPreprocessing::Full(full) = &preprocessing.program {
        let mut tampered = preprocessing.clone();
        let mut program = Arc::clone(full);
        Arc::make_mut(&mut program).bytecode.entry_address += 4;
        tampered.program = ProgramPreprocessing::Full(program);
        check("shifted entry address", &tampered, public_io);
    }
}

/// Trusted-advice tampers: dropping or adding the commitment changes the
/// opening schedule, and replacing it with another valid commitment changes
/// the absorbed public commitments.
pub fn trusted_advice_tampers<C: TracedCase>(case: &C, replacement: &CommitmentOf<C>) {
    let deadline = Region::statement_deadline();
    let run = |label: &str, trusted| {
        assert_statement_rejects::<C>(
            label,
            case.preprocessing(),
            case.public_io(),
            trusted,
            case.proof(),
            deadline,
        );
    };
    match case.trusted_advice_commitment() {
        Some(_) => {
            run("dropped trusted advice commitment", None);
            run("replaced trusted advice commitment", Some(replacement));
        }
        None => run("added trusted advice commitment", Some(replacement)),
    }
}

/// Decodes the `index`-th polynomial commitment the proof sends, reading the
/// commitment region from its start (one commitment may span several
/// messages).
pub fn sent_commitment<C: TracedCase>(case: &C, index: usize) -> CommitmentOf<C> {
    let trace = case.honest_trace();
    let start = message_in(&trace, Region::Commitments, 0).start;
    let mut transcript = VerifierTranscript::<JoltSponge>::new(
        &jolt_verifier::jolt_protocol_id::<JoltSponge>(),
        JOLT_SESSION,
        &case.proof().narg[start..],
    );
    let mut receive = || {
        C::Pcs::receive_commitment(&case.preprocessing().pcs_setup, &mut transcript)
            .expect("a sent commitment decodes")
    };
    for _ in 0..index {
        let _skipped = receive();
    }
    receive()
}

/// Well-formed but wrong header values, re-encoded with the production
/// encoder: each decodes, so rejection must come from validation or from the
/// re-randomized transcript, by the preamble's deadline.
pub fn header_equivocations<C: TracedCase>(case: &C) {
    let narg = &case.proof().narg;
    let honest = decode_header(narg);
    let max_trace = case.preprocessing().program.max_padded_trace_length();
    let mut variants: Vec<(&str, ProofHeader)> = vec![
        (
            "trace length 3",
            ProofHeader {
                trace_length: 3,
                ..honest
            },
        ),
        (
            "trace length above the preprocessing bound",
            ProofHeader {
                trace_length: max_trace * 2,
                ..honest
            },
        ),
        ("ram_K 3", ProofHeader { ram_K: 3, ..honest }),
        ("ram_K 0", ProofHeader { ram_K: 0, ..honest }),
        (
            "doubled ram_K",
            ProofHeader {
                ram_K: honest.ram_K * 2,
                ..honest
            },
        ),
        (
            "flipped trace polynomial order",
            ProofHeader {
                trace_polynomial_order: match honest.trace_polynomial_order {
                    TracePolynomialOrder::CycleMajor => TracePolynomialOrder::AddressMajor,
                    TracePolynomialOrder::AddressMajor => TracePolynomialOrder::CycleMajor,
                },
                ..honest
            },
        ),
        (
            "toggled untrusted advice",
            ProofHeader {
                untrusted_advice: !honest.untrusted_advice,
                ..honest
            },
        ),
    ];
    let mut one_hot = honest.one_hot_config;
    one_hot.log_k_chunk += 1;
    variants.push((
        "log_k_chunk + 1",
        ProofHeader {
            one_hot_config: one_hot,
            ..honest
        },
    ));
    let mut rw = honest.rw_config;
    rw.ram_rw_phase1_num_rounds ^= 1;
    variants.push((
        "ram phase-1 rounds toggled",
        ProofHeader {
            rw_config: rw,
            ..honest
        },
    ));
    if honest.trace_length > 1 {
        variants.push((
            "halved trace length",
            ProofHeader {
                trace_length: honest.trace_length / 2,
                ..honest
            },
        ));
    }

    let deadline = Region::statement_deadline();
    for (label, header) in variants {
        let narg = with_header(narg, &header);
        let proof = case.with_narg(narg);
        let error = statement_rejection::<C>(
            label,
            case.preprocessing(),
            case.public_io(),
            case.trusted_advice_commitment(),
            &proof,
            deadline,
        );
        match label {
            "trace length 3" | "trace length above the preprocessing bound" => assert!(
                matches!(error, VerifierError::InvalidTraceLength { .. }),
                "{label}: {error:?}"
            ),
            "ram_K 3" | "ram_K 0" => assert!(
                matches!(error, VerifierError::InvalidRamK { .. }),
                "{label}: {error:?}"
            ),
            _ => {}
        }
    }
}

/// Substitutes one sent commitment's bytes with another same-width sent
/// commitment's, and swaps the two: valid encodings, wrong commitments.
/// Requires at least two same-width commitments in the proof.
pub fn commitment_substitutions<C: TracedCase>(case: &C) {
    let trace = case.honest_trace();
    let commitments: Vec<_> = trace
        .messages()
        .into_iter()
        .filter(|message| message.region == Region::Commitments)
        .map(|message| message.range)
        .collect();
    let (first, second) = commitments
        .iter()
        .enumerate()
        .find_map(|(index, first)| {
            commitments
                .get(index + 1..)?
                .iter()
                .find(|second| second.len() == first.len())
                .map(|second| (first.clone(), second.clone()))
        })
        .expect("the proof sends two same-width commitments");
    let narg = &case.proof().narg;
    assert_ne!(narg[first.clone()], narg[second.clone()]);

    let deadline = Region::statement_deadline();
    let mut substituted = narg.clone();
    substituted.copy_within(second.clone(), first.start);
    let mut swapped = substituted.clone();
    swapped[second.clone()].copy_from_slice(&narg[first.clone()]);
    for (label, narg) in [("substituted", substituted), ("swapped", swapped)] {
        assert_statement_rejects::<C>(
            &format!("{label} commitments {first:?}/{second:?}"),
            case.preprocessing(),
            case.public_io(),
            case.trusted_advice_commitment(),
            &case.with_narg(narg),
            deadline,
        );
    }
}

pub type PreprocessingMutation<C> =
    fn(&mut JoltVerifierPreprocessing<<C as TracedCase>::Pcs, <C as TracedCase>::Vc>);

/// Program-commitment tampers on a committed-program fixture: each listed
/// preprocessing mutation changes the absorbed public commitments.
pub fn committed_program_tampers<C: TracedCase>(
    case: &C,
    mutations: &[(&str, PreprocessingMutation<C>)],
) {
    assert!(
        case.preprocessing().program.committed().is_some(),
        "fixture must carry committed-program preprocessing"
    );
    for (label, mutate) in mutations {
        let mut preprocessing = case.preprocessing().clone();
        mutate(&mut preprocessing);
        assert_statement_rejects::<C>(
            label,
            &preprocessing,
            case.public_io(),
            case.trusted_advice_commitment(),
            case.proof(),
            Region::statement_deadline(),
        );
    }
}

/// Mutable committed-program preprocessing.
pub fn committed_mut<PCS: CommitmentScheme, VC>(
    preprocessing: &mut JoltVerifierPreprocessing<PCS, VC>,
) -> &mut CommittedProgramPreprocessing<PCS>
where
    VC: VectorCommitment<Field = PCS::Field>,
{
    let ProgramPreprocessing::Committed(committed) = &mut preprocessing.program else {
        panic!("fixture must carry committed-program preprocessing");
    };
    committed
}

/// The proof against another fixture's preprocessing of the same guest (the
/// full program against its committed form, or the reverse): the
/// preprocessing digest and the public commitments differ, so the proof is
/// bound to a different statement.
pub fn preprocessing_swap<C: TracedCase>(
    case: &C,
    other: &JoltVerifierPreprocessing<C::Pcs, C::Vc>,
) {
    assert_ne!(
        case.preprocessing().preprocessing_digest,
        other.preprocessing_digest,
        "the swap must change the preprocessing"
    );
    assert_statement_rejects::<C>(
        "swapped preprocessing",
        other,
        case.public_io(),
        case.trusted_advice_commitment(),
        case.proof(),
        Region::statement_deadline(),
    );
}
