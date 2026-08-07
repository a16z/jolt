pub const SOURCE: &str = include_str!("shader.metal");
pub const MATERIALIZE_PIPELINE: &str = "solinas_outer_remainder_deferred_b_materialize";

mod runtime;

pub use runtime::{
    SpartanOuterDeferredBProbe, SpartanOuterDeferredBProbeConfig, SpartanOuterDeferredBProbeStats,
};

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_field::AkitaField;

    use super::{MATERIALIZE_PIPELINE, SOURCE};
    use crate::metal::solinas::{
        OuterBindingPlan, OuterKernelArtifact, SolinasMetal, SpartanOuterUniskipRow,
    };

    fn splitmix(mut value: u64) -> u64 {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn rows(count: usize) -> Vec<SpartanOuterUniskipRow> {
        (0..count)
            .map(|row| {
                let mut words = std::array::from_fn(|column| {
                    splitmix(row as u64 ^ (column as u64).wrapping_mul(0x1000_0001))
                });
                let selector = splitmix(row as u64 ^ 0xa5a5_5a5a);
                words[19] = selector & ((1 << 20) - 1);
                if selector.is_multiple_of(3) {
                    words[19] &= !0b11;
                } else if selector % 3 == 1 {
                    words[19] = (words[19] & !0b11) | 1;
                } else {
                    words[19] = (words[19] & !0b11) | 2;
                }
                SpartanOuterUniskipRow::from_words(words)
            })
            .collect()
    }

    #[test]
    fn compiler_accepts_deferred_b_materializer() {
        let artifact =
            OuterKernelArtifact::new(SOURCE.to_owned(), OuterBindingPlan::BOnlyV1).unwrap();
        let context = SolinasMetal::for_akita_with_outer_artifact(&artifact).unwrap();
        let pipeline = context
            .compile_named_pipeline(MATERIALIZE_PIPELINE)
            .unwrap();
        let limits = SolinasMetal::limits(&pipeline);

        assert_eq!(limits.thread_execution_width, 32);
        assert!(limits.max_total_threads_per_threadgroup >= 256);
        assert_eq!(limits.static_threadgroup_memory_length, 0);
    }

    #[test]
    fn deferred_b_materializer_matches_the_parent_pipeline() {
        let artifact =
            OuterKernelArtifact::new(SOURCE.to_owned(), OuterBindingPlan::BOnlyV1).unwrap();
        let context = SolinasMetal::for_akita_with_outer_artifact(&artifact).unwrap();
        let resident = context
            .prepare_spartan_outer_uniskip_rows(&rows(256))
            .unwrap();
        let lagrange =
            std::array::from_fn(|index| AkitaField::from_u64(splitmix(0x600d_f00d ^ index as u64)));
        let e_in = (0..16)
            .map(|index| AkitaField::from_u64(splitmix(0x1111 ^ index)))
            .collect::<Vec<_>>();
        let e_out = (0..16)
            .map(|index| AkitaField::from_u64(splitmix(0x2222 ^ index)))
            .collect::<Vec<_>>();
        let mut probe = context
            .prepare_spartan_outer_deferred_b_probe(
                resident,
                &lagrange,
                &e_in,
                &e_out,
                Default::default(),
            )
            .unwrap();

        let parent = probe.run_parent().unwrap();
        let parent_state = probe.read_b_state().unwrap();
        let candidate = probe.run_candidate().unwrap();
        let candidate_state = probe.read_b_state().unwrap();

        assert_eq!(candidate.message, parent.message);
        assert_eq!(candidate_state, parent_state);
        assert_eq!(candidate.pipeline_limits.thread_execution_width, 32);
        assert!(candidate.wall >= candidate.gpu_active);
    }
}
