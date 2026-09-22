//! Native epoch-6 sparse byte-stream parity; no sampler acceptance claim.
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "independent vectors and adversarial witness checks"
)]
mod tests {
    use akita_challenges::SPARSE_CHALLENGE_STREAM_DOMAIN;
    use akita_transcript::blake2b_stream::Blake2bStream;
    use jolt_akita::r1cs::AkitaSparseStreamVar;
    use jolt_field::{Fr, Ring};
    use jolt_r1cs::{bn254_bits::ByteVar, LinearCombination, R1csBuilder, Variable};

    #[test]
    fn native_stream_splits_index_separation_and_tamper() {
        // Python hashlib vectors in the accepted native scripts/blake_only_vectors.py.
        let golden = concat!("f4a89454460818a6568bf8472b1b6d70c3434cb5116211e148f0f14287ba1f96e0df4954023036e3380b270debb294b10764236f32106d9a8b830a8210e3c312", "06ff8479262b8c4536aec3a2e9cab61487fb7fcd746769cbc422fcd583bd1031661b3d11d78e93671828c1fc14aa388abd8c5cdebf789a663f5858ca54e84e93");
        let golden: Vec<_> = golden
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        let mut split_shape = None;
        for split in [0, 63, 64, 65, 127, 128, 129] {
            let mut builder = R1csBuilder::new();
            let root = std::array::from_fn(|_| ByteVar::allocate(&mut builder, Some(0)));
            let mut circuit = AkitaSparseStreamVar::new(&builder, &root, 0).unwrap();
            let mut native = Blake2bStream::new(SPARSE_CHALLENGE_STREAM_DOMAIN, &[0; 40]).unwrap();
            let mut expected = vec![0; 129];
            native.read(&mut expected).unwrap();
            assert_eq!(&expected[..128], golden);
            let mut actual = circuit.read(&mut builder, split).unwrap();
            let before = builder.num_vars();
            assert!(circuit.read(&mut builder, 0).unwrap().is_empty());
            assert_eq!(builder.num_vars(), before);
            actual.extend(circuit.read(&mut builder, 129 - split).unwrap());
            for (byte, value) in actual.iter().zip(expected) {
                builder.assert_equal(
                    byte.expression(),
                    LinearCombination::constant(Fr::from_u64(value.into())),
                );
            }
            let witness = builder.witness().unwrap();
            let matrices = builder.into_matrices();
            let shape = (matrices.num_vars, matrices.num_constraints);
            assert_eq!(*split_shape.get_or_insert(shape), shape);
            assert!(matrices.check_witness(&witness).is_ok());
            if split == 65 {
                for byte in [&root[0], &actual[64], &actual[128]] {
                    let var = byte
                        .expression()
                        .terms
                        .into_iter()
                        .find(|(v, _)| *v != Variable::ONE)
                        .unwrap()
                        .0;
                    let mut changed = witness.clone();
                    changed[var.index()] = Fr::from_u64(1) - changed[var.index()];
                    assert!(matrices.check_witness(&changed).is_err());
                }
            }
        }
        let mut builder = R1csBuilder::new();
        let root = std::array::from_fn(|_| ByteVar::allocate(&mut builder, Some(0)));
        let coordinate = 0x0102_0304_0506_0708;
        let mut stream = AkitaSparseStreamVar::new(&builder, &root, coordinate).unwrap();
        let mut context = vec![0; 32];
        context.extend(coordinate.to_le_bytes());
        let mut native = Blake2bStream::new(SPARSE_CHALLENGE_STREAM_DOMAIN, &context).unwrap();
        let mut expected = [0; 64];
        native.read(&mut expected).unwrap();
        assert_ne!(&expected[..], &golden[..64]);
        for (byte, value) in stream.read(&mut builder, 64).unwrap().iter().zip(expected) {
            builder.assert_equal(
                byte.expression(),
                LinearCombination::constant(Fr::from_u64(value.into())),
            );
        }
        let witness = builder.witness().unwrap();
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn unknown_root_has_identical_matrix_shape() {
        let emit = |value| {
            let mut builder = R1csBuilder::new();
            let root = std::array::from_fn(|_| ByteVar::allocate(&mut builder, value));
            let mut stream = AkitaSparseStreamVar::new(&builder, &root, 7).unwrap();
            let _ = stream.read(&mut builder, 65).unwrap();
            builder.into_matrices()
        };
        let known = emit(Some(0));
        for value in [None, Some(255)] {
            let other = emit(value);
            assert_eq!(known.num_vars, other.num_vars);
            assert_eq!(known.a, other.a);
            assert_eq!(known.b, other.b);
            assert_eq!(known.c, other.c);
        }
    }
}
