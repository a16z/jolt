//! Candidate-only source parity and adversarial constraint checks.
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test vectors and adversarial assignments"
)]
mod tests {
    use akita_challenges::{
        sample_sparse_challenges, D64_SELECTIVE_L2_CHALLENGE_CONFIG, SPARSE_CHALLENGE_STREAM_DOMAIN,
    };
    use akita_pcs::AkitaSerialize;
    use akita_transcript::{blake2b_stream::Blake2bStream, Transcript};
    use jolt_akita::r1cs::{AkitaSparseStreamVar, CandidateError, D64CandidateProfile};
    use jolt_field::{Fr, Ring};
    use jolt_r1cs::{bn254_bits::ByteVar, LinearCombination, R1csBuilder, Variable};

    struct FixedRoot;
    impl Transcript<Fr> for FixedRoot {
        fn bind_instance_bytes(&mut self, _: &[u8]) {}
        fn append_bytes(&mut self, _: &[u8], _: &[u8]) {}
        fn append_field(&mut self, _: &[u8], _: &Fr) {}
        fn append_serde<S: AkitaSerialize>(&mut self, _: &[u8], _: &S) {}
        fn challenge_scalar(&mut self, _: &[u8]) -> Fr {
            Fr::from_u64(0)
        }
        fn challenge_bytes(&mut self, _: &[u8], len: usize) -> Vec<u8> {
            vec![0; len]
        }
        fn challenge_block(&mut self, _: &[u8]) -> [u8; 32] {
            [0; 32]
        }
    }

    #[test]
    fn native_blake_candidate_order_signs_and_dense() {
        let profile = D64CandidateProfile::selective_l2(4).unwrap();
        let native = sample_sparse_challenges::<Fr, _>(
            &mut FixedRoot,
            b"candidate",
            64,
            1,
            &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
            0,
        )
        .unwrap()
        .remove(0);
        let mut bytes = vec![0; profile.tape_len()];
        Blake2bStream::new(SPARSE_CHALLENGE_STREAM_DOMAIN, &[0; 40])
            .unwrap()
            .read(&mut bytes)
            .unwrap();
        let mut builder = R1csBuilder::new();
        let root = std::array::from_fn(|_| ByteVar::allocate(&mut builder, Some(0)));
        let mut stream = AkitaSparseStreamVar::new(&builder, &root, 0).unwrap();
        let tape = stream.read(&mut builder, profile.tape_len()).unwrap();
        for (byte, expected) in tape.iter().zip(bytes) {
            builder.assert_equal(
                byte.expression(),
                LinearCombination::constant(Fr::from_u64(expected.into())),
            );
        }
        let candidate = profile.sample(&mut builder, &tape).unwrap();
        // Independently counted from the canonical root-zero Blake vector: 47 draws + 42 signs.
        assert_eq!(
            builder.evaluate(&candidate.consumed_bytes()).unwrap(),
            Fr::from_u64(89)
        );
        let mut dense = [0i8; 64];
        for ((position, coefficient), (&p, &c)) in candidate
            .positions()
            .iter()
            .zip(candidate.coefficients())
            .zip(native.positions.iter().zip(&native.coeffs))
        {
            builder.assert_equal(
                position.expression(),
                LinearCombination::constant(Fr::from_u64(p as u64)),
            );
            let value = Fr::from_u64(c.unsigned_abs().into());
            builder.assert_equal(
                coefficient.variable(),
                LinearCombination::constant(if c < 0 { -value } else { value }),
            );
            dense[p as usize] = c;
        }
        for (coefficient, c) in candidate.dense().iter().zip(dense) {
            let value = Fr::from_u64(c.unsigned_abs().into());
            builder.assert_equal(
                coefficient.variable(),
                LinearCombination::constant(if c < 0 { -value } else { value }),
            );
        }
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn rejected_bytes_first_acceptance_order_and_tampering() {
        let profile = D64CandidateProfile::new(2, 1, 2).unwrap();
        // 63 is rejected at n=63; accepted draws are 1,2,0. Signs follow all draws.
        let bytes = [0xc1, 0xff, 2, 0, 0, 1, 1, 99, 99];
        let mut builder = R1csBuilder::new();
        let tape: Vec<_> = bytes
            .into_iter()
            .map(|v| ByteVar::allocate(&mut builder, Some(v)))
            .collect();
        let candidate = profile.sample(&mut builder, &tape).unwrap();
        for (position, expected) in candidate.positions().iter().zip([1, 3, 2]) {
            assert_eq!(
                builder.evaluate(&position.expression()).unwrap(),
                Fr::from_u64(expected)
            );
        }
        for (coefficient, expected) in candidate.coefficients().iter().zip([1i64, -1, -2]) {
            let value = Fr::from_u64(expected.unsigned_abs());
            assert_eq!(
                builder
                    .evaluate(&LinearCombination::variable(coefficient.variable()))
                    .unwrap(),
                if expected < 0 { -value } else { value }
            );
        }
        assert_eq!(
            builder.evaluate(&candidate.consumed_bytes()).unwrap(),
            Fr::from_u64(7)
        );
        let mut variables = vec![
            candidate.coefficients()[0].variable(),
            candidate.dense()[1].variable(),
        ];
        for expression in [
            tape[1].expression(),
            candidate.positions()[0].expression(),
            candidate.end_cursor()[7].clone(),
        ] {
            variables.push(
                expression
                    .terms
                    .into_iter()
                    .find(|(v, _)| *v != Variable::ONE)
                    .unwrap()
                    .0,
            );
        }
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        for variable in variables {
            let mut tampered = witness.clone();
            tampered[variable.index()] += Fr::from_u64(1);
            assert!(matrices.check_witness(&tampered).is_err());
        }
    }

    #[test]
    fn private_rejection_count_has_identical_unknown_shape_and_capacity_fails() {
        let profile = D64CandidateProfile::new(2, 0, 2).unwrap();
        let emit = |bytes: Option<[u8; 6]>| {
            let mut builder = R1csBuilder::new();
            let tape: Vec<_> = (0..6)
                .map(|i| ByteVar::allocate(&mut builder, bytes.map(|b| b[i])))
                .collect();
            let _ = profile.sample(&mut builder, &tape).unwrap();
            if bytes.is_some() {
                assert!(builder
                    .clone()
                    .into_matrices()
                    .check_witness(&builder.witness().unwrap())
                    .is_ok());
            }
            builder.into_matrices()
        };
        let known = emit(Some([0; 6]));
        for value in [None, Some([0, 63, 0, 1, 0, 0])] {
            let other = emit(value);
            assert_eq!(known.num_vars, other.num_vars);
            assert_eq!(known.a, other.a);
            assert_eq!(known.b, other.b);
            assert_eq!(known.c, other.c);
        }
        let mut builder = R1csBuilder::new();
        let tape: Vec<_> = [0, 63, 0, 0]
            .into_iter()
            .map(|v| ByteVar::allocate(&mut builder, Some(v)))
            .collect();
        assert!(matches!(
            D64CandidateProfile::new(2, 0, 1)
                .unwrap()
                .sample(&mut builder, &tape),
            Err(CandidateError::CapacityExceeded)
        ));
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_err());
        assert!(D64CandidateProfile::new(usize::MAX, 1, 1).is_err());
        assert!(D64CandidateProfile::new(1, 0, usize::MAX).is_err());
    }

    #[test]
    fn final_unit_range_consumes_no_position_byte() {
        let profile = D64CandidateProfile::new(64, 0, 1).unwrap();
        assert_eq!(profile.tape_len(), 127);
        let mut builder = R1csBuilder::new();
        let tape: Vec<_> = (0..127)
            .map(|_| ByteVar::allocate(&mut builder, Some(0)))
            .collect();
        let candidate = profile.sample(&mut builder, &tape).unwrap();
        assert_eq!(
            builder.evaluate(&candidate.consumed_bytes()).unwrap(),
            Fr::from_u64(127)
        );
        for (i, position) in candidate.positions().iter().enumerate() {
            assert_eq!(
                builder.evaluate(&position.expression()).unwrap(),
                Fr::from_u64(i as u64)
            );
        }
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }
}
