//! A grouped opening whose precommitted object is larger than the final trace
//! group, as when a program's advice capacity exceeds its trace: the setup
//! must cover the larger group, in process and after verifier transport.

#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

#[expect(
    dead_code,
    reason = "shared integration-test support is compiled independently per test file"
)]
mod support;

use akita_params::PolynomialGroupLayout;
use std::sync::Arc;

use jolt_akita::{
    AkitaScheduleArtifacts, AkitaScheme, AkitaSetupParams, AkitaVerifierSetup,
    GroupedScheduleParams, TraceOneHotRows, AKITA_ONE_HOT_K16,
};
use jolt_openings::{
    CommitmentGroupRole, CommitmentScheme, GroupOpeningClaim, TaggedGroupOpeningClaim,
    TransparentObjectSetup,
};
use jolt_poly::{MultilinearPoly, OneHotPolynomial};
use jolt_transcript::{Blake2bTranscript, Transcript};
use support::{f, layout, polynomial};

const FINAL_NUM_VARS: usize = 16;
/// Six variables above this one-column trace group.
const ADVICE_NUM_VARS: usize = 22;
const TRUSTED_ADVICE: CommitmentGroupRole =
    CommitmentGroupRole::new(1, b"trusted_advice", "trusted-advice");

struct TraceRows {
    rows: usize,
}

impl TraceOneHotRows for TraceRows {
    fn num_rows(&self) -> usize {
        self.rows
    }

    fn num_columns(&self) -> usize {
        1
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        selected_rows.fill((row * 7 % 16) as u8);
    }

    fn committed_digit_zero_mask(&self, row: usize) -> u64 {
        u64::from(row.is_multiple_of(16))
    }
}

#[test]
fn grouped_opening_proves_advice_larger_than_the_trace_group() {
    let artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
    let (prover_setup, verifier_setup) =
        AkitaScheme::setup(AkitaSetupParams::one_hot_only_grouped(
            FINAL_NUM_VARS,
            1,
            2,
            layout(5),
            AKITA_ONE_HOT_K16,
            Some(GroupedScheduleParams::new(
                None,
                Some(ADVICE_NUM_VARS),
                Vec::new(),
                PolynomialGroupLayout::new(FINAL_NUM_VARS, 1),
            )),
            artifacts.clone(),
        ))
        .expect("grouped setup should build");

    let advice = polynomial(ADVICE_NUM_VARS, 9);
    let (advice_setup, _) =
        AkitaScheme::transparent_object_setup(&artifacts, ADVICE_NUM_VARS, layout(6))
            .expect("advice object setup should build");
    let (advice_commitment, advice_hint) =
        AkitaScheme::commit(&advice, &advice_setup).expect("advice should commit");
    let advice_point: Vec<_> = (0..ADVICE_NUM_VARS).map(|i| f(7 + 5 * i as u64)).collect();
    let advice_claim = TaggedGroupOpeningClaim::new(
        TRUSTED_ADVICE,
        GroupOpeningClaim::new(
            advice_commitment,
            advice_point.clone(),
            vec![advice.evaluate(&advice_point)],
        ),
    );

    let rows = 1 << (FINAL_NUM_VARS - 4);
    let trace = OneHotPolynomial::new(
        AKITA_ONE_HOT_K16,
        (0..rows).map(|row| Some((row * 7 % 16) as u8)).collect(),
    );
    let trace_point: Vec<_> = (0..FINAL_NUM_VARS).map(|i| f(3 + 2 * i as u64)).collect();
    let trace_evaluation = trace.evaluate(&trace_point);
    let (trace_commitment, trace_hint) = AkitaScheme::commit_trace_one_hot(
        &prover_setup,
        layout(5),
        Arc::new(TraceRows { rows }),
        &[&advice_hint],
    )
    .expect("trace group should commit against the larger advice object");
    let main = GroupOpeningClaim::new(trace_commitment, trace_point, vec![trace_evaluation]);

    let mut prover_transcript = Blake2bTranscript::new(b"akita-grouped-capacity");
    let proof = AkitaScheme::prove_batch(
        &prover_setup,
        vec![(advice_claim.clone(), advice_hint)],
        main.clone(),
        trace_hint,
        &mut prover_transcript,
    )
    .expect("grouped opening should prove");

    let transported: AkitaVerifierSetup = serde_json::from_str(
        &serde_json::to_string(&verifier_setup).expect("verifier setup should serialize"),
    )
    .expect("verifier setup should deserialize");
    for setup in [&verifier_setup, &transported] {
        let mut verifier_transcript = Blake2bTranscript::new(b"akita-grouped-capacity");
        AkitaScheme::verify_batch(
            setup,
            std::slice::from_ref(&advice_claim),
            &main,
            &proof,
            &mut verifier_transcript,
        )
        .expect("grouped opening should verify");
        assert_eq!(prover_transcript.state(), verifier_transcript.state());
    }

    let mut other_context = Blake2bTranscript::new(b"akita-grouped-capacity");
    other_context.append_bytes(b"different preceding Jolt messages");
    assert!(AkitaScheme::verify_batch(
        &transported,
        std::slice::from_ref(&advice_claim),
        &main,
        &proof,
        &mut other_context,
    )
    .is_err());
}
