#![cfg(all(feature = "cuda", not(feature = "akita")))]
#![expect(
    clippy::expect_used,
    reason = "end-to-end fixture failures must fail the test"
)]

#[path = "../../jolt-prover/tests/support/mod.rs"]
#[cfg_attr(
    feature = "field-inline",
    expect(
        dead_code,
        reason = "reuse main's guest builders, with CUDA proving below"
    )
)]
mod support;

use dory::stages::stage0::TrustedAdviceCommitment;
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
#[cfg(feature = "zk")]
use jolt_field::Ring;
#[cfg(not(feature = "field-inline"))]
use jolt_inlines_keccak256 as _;
#[cfg(not(feature = "field-inline"))]
use jolt_inlines_sha2 as _;
use jolt_kernels::cuda::CudaDoryScheme;
use jolt_kernels::JoltBackend;
use jolt_prover::{
    dory, CommittedProgramProverData, JoltProverPreprocessing, JoltSharedPreprocessing,
    ProverConfig,
};
use jolt_transcript::LegacyBlake2bTranscript;
#[cfg(feature = "zk")]
use jolt_verifier::proof::JoltProofClaims;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use support::GuestCase;

fn prove_guest(
    backend: &JoltBackend<Fr, CudaDoryScheme>,
    case: GuestCase,
    address_first: bool,
    committed: bool,
    order: TracePolynomialOrder,
) {
    let prepared = support::prepare(&case);
    let mut config = ProverConfig::derive_from_dimensions::<Fr>(
        prepared.trace.dimensions,
        &prepared.preprocessing.memory_layout,
        prepared.preprocessing.ram.min_bytecode_address,
        prepared.preprocessing.ram.bytecode_words.len(),
        prepared.preprocessing.max_padded_trace_length,
    )
    .expect("derive config");
    config.trace_polynomial_order = order;
    let log_t = config.trace_length.ilog2() as usize;
    let phase1 = if address_first { 0 } else { log_t as u8 };
    config.rw_config.ram_rw_phase1_num_rounds = phase1;
    config.rw_config.registers_rw_phase1_num_rounds = phase1;
    let preprocessing = if committed {
        dory::preprocess_committed_with_order(prepared.preprocessing, config.trace_polynomial_order)
    } else {
        dory::from_shared(
            JoltSharedPreprocessing::new(prepared.preprocessing).expect("shared preprocessing"),
        )
    }
    .expect("preprocessing");
    let cuda_preprocessing = JoltProverPreprocessing::<CudaDoryScheme, Pedersen<Bn254G1>> {
        verifier: CudaDoryScheme::adopt_verifier_preprocessing(preprocessing.verifier.clone())
            .expect("CUDA verifier preprocessing"),
        pcs_setup: preprocessing.pcs_setup.clone(),
        committed_program: preprocessing.committed_program.as_ref().map(|data| {
            CommittedProgramProverData {
                full: data.full.clone(),
                bytecode_hint: data.bytecode_hint.clone(),
                program_image_hint: data.program_image_hint.clone(),
                trace_order: data.trace_order,
            }
        }),
    };
    let program = preprocessing.program_arc().expect("program preprocessing");
    let public_io = prepared.trace.device.clone();
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config)
            .include_trusted_advice(!case.trusted_advice.is_empty())
            .include_untrusted_advice(!case.untrusted_advice.is_empty()),
        JoltVmWitnessInputs::new(&prepared.program, &program, prepared.trace),
    );
    #[cfg(feature = "field-inline")]
    let witness = witness.with_field_inline().expect("field-inline witness");
    let trusted = (!case.trusted_advice.is_empty()).then(|| {
        dory::commit_trusted_advice(&preprocessing, &case.trusted_advice)
            .expect("trusted advice commitment")
    });
    let cuda_trusted = trusted
        .as_ref()
        .map(|advice| TrustedAdviceCommitment::<CudaDoryScheme> {
            commitment: advice.commitment.clone(),
            hint: advice.hint.clone(),
        });
    let cpu = dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
        JoltBackend::optimized().with_witness(&witness),
        &preprocessing,
        &config,
        trusted.as_ref(),
        &public_io,
    )
    .expect("optimized proof");
    let gpu = dory::prove::<Fr, CudaDoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
        backend.with_witness(&witness),
        &cuda_preprocessing,
        &config,
        cuda_trusted.as_ref(),
        &public_io,
    )
    .expect("CUDA proof");
    jolt_verifier::verify::<Fr, CudaDoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
        &cuda_preprocessing.verifier,
        &public_io,
        &gpu,
        trusted.as_ref().map(|advice| &advice.commitment),
    )
    .expect("CUDA proof verifies");
    #[cfg(not(feature = "zk"))]
    assert_eq!(
        bincode::serde::encode_to_vec(&gpu, bincode::config::standard())
            .expect("serialize CUDA proof"),
        bincode::serde::encode_to_vec(&cpu, bincode::config::standard())
            .expect("serialize optimized proof"),
        "CUDA and optimized proofs must be byte-identical",
    );
    #[cfg(feature = "zk")]
    jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
        &preprocessing.verifier,
        &public_io,
        &cpu,
        trusted.as_ref().map(|advice| &advice.commitment),
    )
    .expect("optimized ZK proof verifies");
    #[cfg(feature = "zk")]
    {
        let mut tampered = gpu;
        let JoltProofClaims::Zk { blindfold_proof } = &mut tampered.claims else {
            unreachable!("a ZK build must produce BlindFold claims")
        };
        blindfold_proof.random_u += Fr::from_u64(1);
        assert!(
            jolt_verifier::verify::<
                Fr,
                CudaDoryScheme,
                Pedersen<Bn254G1>,
                LegacyBlake2bTranscript,
            >(
                &cuda_preprocessing.verifier,
                &public_io,
                &tampered,
                trusted.as_ref().map(|advice| &advice.commitment),
            )
            .is_err(),
            "tampered CUDA ZK proof must be rejected",
        );
    }
}

fn muldiv_case() -> GuestCase {
    GuestCase {
        inputs: postcard::to_stdvec(&[9u32, 5, 3]).expect("inputs"),
        expected_output: Some(postcard::to_stdvec(&15u32).expect("output")),
        ..GuestCase::new("muldiv-guest")
    }
}

fn run(case: GuestCase, address_first: bool, committed: bool) {
    run_with_order(
        case,
        address_first,
        committed,
        TracePolynomialOrder::CycleMajor,
    );
}

fn run_with_order(
    case: GuestCase,
    address_first: bool,
    committed: bool,
    order: TracePolynomialOrder,
) {
    #[cfg(feature = "zk")]
    support::with_zk_stack(move || {
        prove_guest(&JoltBackend::cuda(), case, address_first, committed, order);
    });
    #[cfg(not(feature = "zk"))]
    prove_guest(&JoltBackend::cuda(), case, address_first, committed, order);
}

#[test]
fn muldiv_cycle_first() {
    run(muldiv_case(), false, false);
}
#[test]
fn muldiv_address_first() {
    run(muldiv_case(), true, false);
}
#[test]
#[cfg(not(feature = "field-inline"))]
fn muldiv_committed_program() {
    run(muldiv_case(), true, true);
}

#[test]
#[cfg(not(feature = "field-inline"))]
fn muldiv_committed_program_address_major() {
    run_with_order(
        muldiv_case(),
        false,
        true,
        TracePolynomialOrder::AddressMajor,
    );
}

#[test]
fn advice_address_first() {
    run(advice_case(), true, false);
}

fn advice_case() -> GuestCase {
    GuestCase {
        inputs: postcard::to_stdvec(&12u64).expect("input"),
        untrusted_advice: postcard::to_stdvec(&5u64).expect("untrusted advice"),
        trusted_advice: postcard::to_stdvec(&7u64).expect("trusted advice"),
        expected_output: Some(postcard::to_stdvec(&26u64).expect("output")),
        ..GuestCase::new("advice-consumer-guest")
    }
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn advice_committed_program() {
    run(advice_case(), true, true);
}

#[test]
fn trusted_advice_only() {
    run(
        GuestCase {
            trusted_advice: postcard::to_stdvec(&7u64).expect("trusted advice"),
            ..muldiv_case()
        },
        true,
        false,
    );
}

#[test]
fn untrusted_advice_only() {
    run(
        GuestCase {
            untrusted_advice: postcard::to_stdvec(&5u64).expect("untrusted advice"),
            ..muldiv_case()
        },
        false,
        false,
    );
}
