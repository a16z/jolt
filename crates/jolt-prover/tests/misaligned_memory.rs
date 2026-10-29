//! Soundness: a sub-word access misaligned for its width cannot be proven.
//!
//! The tracer aborts on such an access, so these tests trace one with
//! misaligned floor accesses allowed. Each floor-addressed row then records the
//! RAM address R1CS constraint 0 assigns, `ea − AddressOffset(ea)`, which is
//! not a multiple of 8. The witness satisfies every R1CS row, and `RamAddress`
//! is consumed only by RAM RAF evaluation, which admits only multiples of 8, so
//! that sumcheck is what must reject the proof.

#[cfg(all(
    feature = "prover-fixtures",
    not(any(feature = "zk", feature = "akita", feature = "field-inline"))
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    not(any(feature = "zk", feature = "akita", feature = "field-inline"))
))]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]
mod clear {
    use jolt_claims::protocols::composed::r1cs::rv64::{
        const_column, input_column, rv64_spartan_outer_constraints, NUM_VARS_PER_CYCLE,
    };
    use jolt_claims::protocols::jolt::geometry::spartan::SPARTAN_OUTER_R1CS_INPUTS;
    use jolt_claims::protocols::jolt::JoltPolynomialId;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_prover::{dory, JoltBackend, JoltSharedPreprocessing, ProverConfig, ProverError};
    use jolt_sumcheck::SumcheckError;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend};
    use tracer::instruction::floor_access::allow_misaligned_floor_accesses;

    use crate::support::{self, GuestCase};

    /// Every eq-conditional R1CS row holds on every cycle.
    fn assert_r1cs_satisfied(witness: &TraceBackend) {
        let tables = SPARTAN_OUTER_R1CS_INPUTS.map(|input| {
            JoltWitnessOracle::<Fr>::oracle_table(witness, JoltPolynomialId::Virtual(input))
                .expect("R1CS input table")
        });
        let constraints = rv64_spartan_outer_constraints::<Fr>();
        let mut z = vec![Fr::from_u64(0); NUM_VARS_PER_CYCLE];
        z[const_column()] = Fr::from_u64(1);
        for cycle in 0..tables[0].len() {
            for (input, table) in tables.iter().enumerate() {
                z[input_column(input).expect("R1CS input column")] = table[cycle];
            }
            assert_eq!(constraints.check_witness(&z), Ok(()), "cycle {cycle}");
        }
    }

    /// Traces the memory-ops guest's misaligned access of `kind` and asserts
    /// that its witness satisfies R1CS yet cannot be proven.
    fn assert_misaligned_access_rejected(kind: u8) {
        let misaligned = allow_misaligned_floor_accesses();
        let prepared = support::prepare(&GuestCase {
            func: Some("misaligned_access"),
            inputs: postcard::to_stdvec(&kind).expect("serialize access kind"),
            ..GuestCase::new("memory-ops-guest")
        });
        drop(misaligned);
        assert!(
            prepared
                .trace
                .trace
                .rows()
                .iter()
                .any(|row| (row.is_load() || row.is_store()) && row.ram_address() % 8 != 0),
            "kind {kind}: the trace records no misaligned RAM address",
        );

        let config = ProverConfig::derive_from_dimensions::<Fr>(
            prepared.trace.dimensions,
            &prepared.preprocessing.memory_layout,
            prepared.preprocessing.ram.min_bytecode_address,
            prepared.preprocessing.ram.bytecode_words.len(),
            prepared.preprocessing.max_padded_trace_length,
        )
        .expect("derive config");
        let preprocessing = dory::from_shared(
            JoltSharedPreprocessing::new(prepared.preprocessing).expect("shared preprocessing"),
        )
        .expect("Dory preprocessing");
        let program_preprocessing = preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let public_io = prepared.trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&prepared.program, &program_preprocessing, prepared.trace),
        );
        assert_r1cs_satisfied(&witness);

        match dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            JoltBackend::optimized().with_witness(&witness),
            &preprocessing,
            &config,
            None,
            &public_io,
        ) {
            // The batched stage-2 sumcheck opens with the RAF input claim
            // `2^k·RamAddress(r)`, which the prover's `Σ ra·unmap` rounds cannot
            // reach.
            Err(error) => assert!(
                matches!(
                    &error,
                    ProverError::Sumcheck(SumcheckError::RoundCheckFailed { round: 0, .. })
                ),
                "kind {kind}: unexpected prover error {error:?}",
            ),
            Ok(proof) => assert!(
                jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                    &preprocessing.verifier,
                    &public_io,
                    &proof,
                    None,
                )
                .is_err(),
                "kind {kind}: a misaligned access produced an accepted proof",
            ),
        }
    }

    #[test]
    fn misaligned_lh_is_rejected() {
        assert_misaligned_access_rejected(0);
    }

    #[test]
    fn misaligned_lhu_is_rejected() {
        assert_misaligned_access_rejected(1);
    }

    #[test]
    fn misaligned_lw_is_rejected() {
        assert_misaligned_access_rejected(2);
    }

    #[test]
    fn misaligned_lwu_is_rejected() {
        assert_misaligned_access_rejected(3);
    }

    #[test]
    fn misaligned_sh_is_rejected() {
        assert_misaligned_access_rejected(4);
    }

    #[test]
    fn misaligned_sw_is_rejected() {
        assert_misaligned_access_rejected(5);
    }
}
