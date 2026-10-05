#![expect(
    clippy::unwrap_used,
    reason = "Metal regression tests assert valid commitments and openings"
)]

use super::*;
use crate::trace_onehot::TracePackedSelectors;
use jolt_field::{One, Ring};
use jolt_poly::eq_index_msb;
use jolt_transcript::Blake2bTranscript;

struct ConstantRows {
    selectors: Vec<u8>,
    active_zero_rows: Vec<u64>,
}

impl TraceOneHotRows for ConstantRows {
    fn num_rows(&self) -> usize {
        self.selectors.len() / 3
    }

    fn num_columns(&self) -> usize {
        3
    }

    fn fill_row(&self, _row: usize, selected_rows: &mut [u8]) {
        selected_rows.copy_from_slice(&[0, 0, 5]);
    }

    fn packed_selectors(&self) -> Option<TracePackedSelectors<'_>> {
        Some(TracePackedSelectors::new(
            &self.selectors,
            &self.active_zero_rows,
            1,
        ))
    }

    fn committed_digit_zero_mask(&self, _row: usize) -> u64 {
        1
    }
}

fn small_trace_roundtrip(backend: TraceCommitmentBackend) {
    let log_t = 21;
    let num_vars = log_t + 10;
    let layout_digest = [4; 32];
    let (setup, verifier_setup) = AkitaScheme::setup(AkitaSetupParams::one_hot_only(
        num_vars,
        1,
        layout_digest,
        AKITA_ONE_HOT_K16,
        AkitaScheduleArtifacts::shared_from_default_directory(),
    ))
    .unwrap();
    let rows: Arc<dyn TraceOneHotRows> = Arc::new(ConstantRows {
        selectors: [0, 0, 5].repeat(1 << log_t),
        active_zero_rows: vec![u64::MAX; (1 << log_t) / 64],
    });
    let (commitment, hint) = AkitaScheme::commit_trace_one_hot(
        &backend,
        &setup,
        layout_digest,
        64,
        Arc::clone(&rows),
        &[],
    )
    .unwrap();
    if let Some(metal) = backend.required_metal() {
        let metrics = metal.backend.last_commit_metrics().unwrap().unwrap();
        assert!(
            metrics.command_buffers > 0,
            "commitment must dispatch to Metal"
        );
        let (cpu_commitment, _) = AkitaScheme::commit_trace_one_hot(
            &TraceCommitmentBackend::cpu(),
            &setup,
            layout_digest,
            64,
            rows,
            &[],
        )
        .unwrap();
        assert_eq!(commitment, cpu_commitment);
    }

    let point = (0..num_vars)
        .map(|i| AkitaField::from_u64(i as u64 + 2))
        .collect::<Vec<_>>();
    let (column_point, rest) = point.split_at(6);
    let (_, address_point) = rest.split_at(log_t);
    // Every cycle has a live zero in column 0, a cold column 1, and a
    // live five in column 2. All remaining columns are padding.
    let value = eq_index_msb(column_point, 0) * eq_index_msb(address_point, 0)
        + eq_index_msb(column_point, 2) * eq_index_msb(address_point, 5);
    let mut claim = GroupOpeningClaim::new(commitment, point, vec![value]);
    let mut prover_transcript = Blake2bTranscript::<AkitaField>::new(b"small-metal-trace");
    let proof = AkitaNativeBatching::prove_trace_batch(
        &setup,
        vec![],
        claim.clone(),
        hint,
        &mut prover_transcript,
    )
    .unwrap();
    if let Some(metal) = backend.required_metal() {
        let metrics = metal.backend.last_opening_metrics().unwrap().unwrap();
        assert!(
            !metrics.command_wall_time.is_zero(),
            "opening must dispatch to Metal"
        );
    }
    let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"small-metal-trace");
    AkitaNativeBatching::verify_trace_batch(
        &verifier_setup,
        &[],
        &claim,
        &proof,
        &mut verifier_transcript,
    )
    .unwrap();
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
    *claim.evaluations.first_mut().unwrap() += AkitaField::one();
    let mut verifier_transcript = Blake2bTranscript::<AkitaField>::new(b"small-metal-trace");
    assert!(AkitaNativeBatching::verify_trace_batch(
        &verifier_setup,
        &[],
        &claim,
        &proof,
        &mut verifier_transcript,
    )
    .is_err());
}

#[test]
fn small_k16_trace_metal_roundtrip() {
    small_trace_roundtrip(TraceCommitmentBackend::metal_required().unwrap());
}

#[test]
fn small_k16_trace_cpu_roundtrip() {
    small_trace_roundtrip(TraceCommitmentBackend::cpu());
}
