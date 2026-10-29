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

#[cfg(feature = "field-inline")]
#[path = "../../jolt-prover/tests/support/dory_backend.rs"]
mod dory_backend;

use cudarc::driver::CudaContext;
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
use jolt_kernels::cuda::{device_available, device_count, CudaDoryScheme, DEVICE_COUNT_VARIABLE};
use jolt_kernels::JoltBackend;
use jolt_prover::{
    dory, CommittedProgramProverData, JoltProverPreprocessing, JoltSharedPreprocessing,
    ProverConfig,
};
use jolt_transcript::{Keccak, ProverState, VerifierState};
#[cfg(feature = "zk")]
use jolt_verifier::proof::JoltProofClaims;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
#[cfg(not(feature = "field-inline"))]
use serde::Serialize;
use support::GuestCase;

fn prove_guest(
    backend: &JoltBackend<Fr, CudaDoryScheme>,
    case: GuestCase,
    address_first: bool,
    committed: bool,
    order: TracePolynomialOrder,
) {
    if CudaContext::new(0).is_err() {
        return;
    }
    assert!(
        device_available(),
        "CUDA kernels must load when a device is present"
    );
    if let Ok(requested) = std::env::var(DEVICE_COUNT_VARIABLE) {
        assert_eq!(
            device_count(),
            requested
                .parse::<usize>()
                .expect("requested CUDA device count"),
            "every requested CUDA device must be usable",
        );
    }
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
    let cpu = dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
        JoltBackend::optimized().with_witness(&witness),
        &preprocessing,
        &config,
        trusted.as_ref(),
        &public_io,
    )
    .expect("optimized proof");
    let gpu = dory::prove::<Fr, CudaDoryScheme, Pedersen<Bn254G1>, ProverState<Keccak>>(
        backend.with_witness(&witness),
        &cuda_preprocessing,
        &config,
        cuda_trusted.as_ref(),
        &public_io,
    )
    .expect("CUDA proof");
    jolt_verifier::verify::<Fr, CudaDoryScheme, Pedersen<Bn254G1>, VerifierState<'_, Keccak>>(
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
    jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, VerifierState<'_, Keccak>>(
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
                VerifierState<'_, Keccak>,
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

#[cfg(not(feature = "zk"))]
#[test]
fn reused_registry_isolates_witness_sessions() {
    let backend = JoltBackend::cuda();
    prove_guest(
        &backend,
        muldiv_case(),
        false,
        false,
        TracePolynomialOrder::CycleMajor,
    );
    let different_input = GuestCase {
        inputs: postcard::to_stdvec(&[18u32, 2, 3]).expect("inputs"),
        expected_output: Some(postcard::to_stdvec(&12u32).expect("output")),
        ..GuestCase::new("muldiv-guest")
    };
    prove_guest(
        &backend,
        different_input,
        true,
        false,
        TracePolynomialOrder::CycleMajor,
    );
    prove_guest(
        &backend,
        muldiv_case(),
        false,
        false,
        TracePolynomialOrder::CycleMajor,
    );
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

#[cfg(not(feature = "field-inline"))]
fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    postcard::to_stdvec(value).expect("encode guest input/output")
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn fibonacci() {
    run(
        GuestCase {
            inputs: encode(&100u32),
            expected_output: Some(encode(&354_224_848_179_261_915_075u128)),
            ..GuestCase::new("fibonacci-guest")
        },
        true,
        false,
    );
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn memory_ops() {
    run(
        GuestCase {
            func: Some("memory_ops"),
            // Frozen output of the byte-array model in the prover acceptance matrix.
            expected_output: Some(encode(&(
                0x94a2_fdd2_d3f7_e573u64,
                [
                    0xdb7b_da33_648d_38dfu64,
                    0xe571_1528_d851_877a,
                    0xbc92_62b3_4fa5_20bb,
                ],
            ))),
            ..GuestCase::new("memory-ops-guest")
        },
        false,
        false,
    );
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn sha2() {
    use sha2::{Digest, Sha256};
    let message: Vec<u8> = (0..127u8).collect();
    let digest: [u8; 32] = Sha256::digest(&message).into();
    run(
        GuestCase {
            inputs: encode(&message),
            expected_output: Some(encode(&digest)),
            ..GuestCase::new("sha2-guest")
        },
        true,
        false,
    );
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn sha3() {
    use sha3::{Digest, Keccak256};
    let message: Vec<u8> = (0..300).map(|i| i as u8).collect();
    let digest: [u8; 32] = Keccak256::digest(&message).into();
    run(
        GuestCase {
            func: Some("sha3"),
            inputs: encode(&message),
            expected_output: Some(encode(&digest)),
            ..GuestCase::new("sha3-guest")
        },
        false,
        false,
    );
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn sha3_aligned() {
    use sha3::{Digest, Keccak256};
    let mut blocks = [[0u64; 17]; 2];
    for (i, lane) in blocks.iter_mut().flatten().enumerate() {
        *lane = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    let bytes: Vec<_> = blocks
        .iter()
        .flatten()
        .flat_map(|lane| lane.to_le_bytes())
        .collect();
    let digest: [u8; 32] = Keccak256::digest(&bytes).into();
    run(
        GuestCase {
            func: Some("sha3_aligned"),
            inputs: encode(&blocks),
            expected_output: Some(encode(&digest)),
            ..GuestCase::new("sha3-guest")
        },
        true,
        false,
    );
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn btreemap() {
    run(
        GuestCase {
            stack_size: Some(10_000),
            inputs: encode(&50u32),
            expected_output: Some(encode(&935u128)),
            ..GuestCase::new("btreemap-guest")
        },
        true,
        false,
    );
}

#[cfg(not(feature = "field-inline"))]
#[test]
fn stdlib() {
    run(
        GuestCase {
            func: Some("string_concat"),
            std: true,
            stack_size: Some(1 << 20),
            inputs: encode(&16i32),
            expected_output: Some(encode(&(0..16).map(|i| i.to_string()).collect::<String>())),
            ..GuestCase::new("stdlib-guest")
        },
        false,
        false,
    );
}

#[cfg(feature = "field-inline")]
#[test]
fn field_ops() {
    run(support::field_inline::field_ops(), true, false);
}

#[cfg(feature = "field-inline")]
#[test]
fn field_limbs() {
    use jolt_field::Ring;
    use jolt_host::field_inline::canonical_limbs;
    let p_minus_one =
        canonical_limbs::<Fr, 6>(Fr::from_u64(0) - Fr::from_u64(1)).expect("six limbs");
    run(
        GuestCase {
            inputs: postcard::to_stdvec(&(Some(p_minus_one), Some(6u8))).expect("limbs input"),
            expected_output: Some(postcard::to_stdvec(&p_minus_one).expect("limbs output")),
            field_inline_active: true,
            ..GuestCase::new("field-limbs-guest")
        },
        false,
        false,
    );
}
