#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "production differential vectors and witness mutations"
)]
mod tests {
    use akita_pcs::{AkitaTranscript, Transcript};
    use jolt_akita::r1cs::AkitaTranscriptVar;
    use jolt_akita::AkitaField;
    use jolt_field::Fr;
    use jolt_field::Ring;
    use jolt_r1cs::{bn254_bits::ByteVar, R1csBuilder};
    use jolt_r1cs::{LinearCombination, Variable};

    #[test]
    fn actual_akita_framing_chunk_discard_and_reabsorb() {
        let session = b"jolt-akita/bridge\0";
        let instance = b"bound instance descriptor bytes";
        let mut native = AkitaTranscript::<AkitaField>::verifier(session, instance);
        let mut builder = R1csBuilder::new();
        let session_vars: Vec<_> = session
            .iter()
            .map(|&b| ByteVar::allocate(&mut builder, Some(b)))
            .collect();
        let instance_vars: Vec<_> = instance
            .iter()
            .map(|&b| ByteVar::allocate(&mut builder, Some(b)))
            .collect();
        let mut circuit =
            AkitaTranscriptVar::new(&mut builder, &session_vars, &instance_vars).unwrap();
        // 1 then 33 consumes 32 then 64 bytes; 0 must not terminate absorption.
        for (payload, len) in [
            (&b""[..], 0),
            (&b"abc"[..], 1),
            (&b""[..], 33),
            (&b"tail"[..], 32),
        ] {
            native.append_bytes(b"labels ignored upstream", payload);
            let vars: Vec<_> = payload.iter().copied().map(ByteVar::constant).collect();
            circuit.append_bytes(&mut builder, &vars).unwrap();
            let expected = native.challenge_bytes(b"any label", len);
            let actual = circuit.challenge_bytes(&mut builder, len).unwrap();
            for (byte, value) in actual.iter().zip(expected) {
                builder.assert_equal(
                    byte.expression(),
                    LinearCombination::constant(Fr::from_u64(u64::from(value))),
                );
            }
        }
        // Consecutive partial requests must each discard their own final suffix.
        for len in [1, 1, 31] {
            let expected = native.challenge_bytes(b"", len);
            let actual = circuit.challenge_bytes(&mut builder, len).unwrap();
            for (byte, value) in actual.iter().zip(expected) {
                builder.assert_equal(
                    byte.expression(),
                    LinearCombination::constant(Fr::from_u64(u64::from(value))),
                );
            }
        }
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        for byte in [&session_vars[0], &instance_vars[0]] {
            let var = byte
                .expression()
                .terms
                .iter()
                .find(|(v, _)| *v != Variable::ONE)
                .unwrap()
                .0;
            let mut changed = witness.clone();
            changed[var.index()] = Fr::from_u64(1) - changed[var.index()];
            assert!(matrices.check_witness(&changed).is_err());
        }
    }
}
