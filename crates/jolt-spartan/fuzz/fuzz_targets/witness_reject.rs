#![no_main]

//! Fuzzer that builds a correct witness then corrupts it using fuzzer
//! data. Verifies that the prover always rejects unsatisfying witnesses.

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_spartan::{FirstRoundStrategy, SimpleR1CS, SpartanKey, SpartanProver};
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

type MockPCS = MockCommitmentScheme<Fr>;

fuzz_target!(|data: &[u8]| {
    // Need at least 2 bytes: one for tamper index, one for delta
    if data.len() < 33 {
        return;
    }

    let one = Fr::from_u64(1);
    let r1cs = SimpleR1CS::new(
        2,
        4,
        vec![(0, 1, one), (1, 2, one)],
        vec![(0, 1, one), (1, 1, one)],
        vec![(0, 2, one), (1, 3, one)],
    );
    let key = SpartanKey::from_r1cs(&r1cs);

    // Valid witness: 1, 2, 4, 8
    let mut witness = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(4),
        Fr::from_u64(8),
    ];

    // Tamper one entry
    let tamper_idx = (data[0] as usize) % witness.len();
    let delta = Fr::from_bytes(&data[1..33]);

    if delta == Fr::from_u64(0) {
        return;
    }

    witness[tamper_idx] += delta;

    let mut transcript = Blake2bTranscript::new(b"fuzz-witness");
    let result = SpartanProver::prove::<MockPCS, _>(
        &r1cs,
        &key,
        &witness,
        &(),
        &mut transcript,
        FirstRoundStrategy::Standard,
    );

    // Tampering z[0] = 1 (the constant) may still satisfy if the delta
    // happens to produce a valid solution. For non-trivial circuits this
    // is astronomically unlikely, but we don't assert unconditionally.
    // For z[1..], corruption should always be caught.
    if tamper_idx >= 1 {
        assert!(
            result.is_err(),
            "corrupted witness[{tamper_idx}] should be rejected"
        );
    }
});
