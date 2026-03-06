#![no_main]

//! Fuzzer that generates a valid Spartan proof for a fixed circuit,
//! then deterministically tampers with proof fields based on fuzzer
//! input. Verifies that any tampering is detected.

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_spartan::{
    FirstRoundStrategy, SimpleR1CS, SpartanKey, SpartanProof, SpartanProver, SpartanVerifier,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

type MockPCS = MockCommitmentScheme<Fr>;

fn x_squared_circuit() -> (SimpleR1CS<Fr>, Vec<Fr>, SpartanKey<Fr>) {
    let one = Fr::from_u64(1);
    let r1cs = SimpleR1CS::new(
        2,
        4,
        vec![(0, 1, one), (1, 2, one)],
        vec![(0, 1, one), (1, 1, one)],
        vec![(0, 2, one), (1, 3, one)],
    );
    let key = SpartanKey::from_r1cs(&r1cs);
    let witness = vec![
        Fr::from_u64(1),
        Fr::from_u64(2),
        Fr::from_u64(4),
        Fr::from_u64(8),
    ];
    (r1cs, witness, key)
}

fn valid_proof(
    r1cs: &SimpleR1CS<Fr>,
    key: &SpartanKey<Fr>,
    witness: &[Fr],
) -> SpartanProof<Fr, MockPCS> {
    let mut transcript = Blake2bTranscript::new(b"fuzz");
    SpartanProver::prove::<MockPCS, _>(
        r1cs,
        key,
        witness,
        &(),
        &mut transcript,
        FirstRoundStrategy::Standard,
    )
    .expect("valid proof")
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let (r1cs, witness, key) = x_squared_circuit();
    let mut proof = valid_proof(&r1cs, &key, &witness);

    // Use first byte to select which field to tamper
    let tamper_target = data[0] % 4;
    let delta = Fr::from_bytes(&{
        let mut buf = [0u8; 32];
        let copy_len = (data.len() - 1).min(32);
        buf[..copy_len].copy_from_slice(&data[1..1 + copy_len]);
        buf
    });

    // Skip no-op tampering (delta = 0)
    if delta == Fr::from_u64(0) {
        return;
    }

    match tamper_target {
        0 => proof.az_eval += delta,
        1 => proof.bz_eval += delta,
        2 => proof.cz_eval += delta,
        3 => proof.witness_eval += delta,
        _ => unreachable!(),
    }

    let mut vt = Blake2bTranscript::new(b"fuzz");
    let result = SpartanVerifier::verify::<MockPCS, _>(&key, &proof, &(), &mut vt);
    assert!(
        result.is_err(),
        "tampered proof (target={tamper_target}) should be rejected"
    );
});
