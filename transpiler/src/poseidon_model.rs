//! Concrete reference model of jolt-transcript's field-aligned `PoseidonSponge`
//! (`U = Fr`, spec §4), used to VERIFY the unit schedule the in-circuit
//! `FieldAlignedLayout` must reproduce (spec §4.5 / T-O5).
//!
//! The model re-implements the compression-chain sponge of
//! `crates/jolt-transcript/src/poseidon.rs` over concrete `Fr` (one `Fr`
//! state; `permute(a, b)` = width-4 Circom Poseidon over `[state, a, b]`;
//! absorb feeds unit pairs zero-padding an odd tail; one squeezed unit = one
//! permute, challenge = the new state). The message-level unit encodings are
//! IMPORTED from `jolt_transcript` (`push_byte_rule_units`,
//! `push_field_frame_units`, `push_commitments_frame_header`,
//! `poseidon_domain_separator_msgs`) so the model cannot drift from the
//! typed codec it mirrors.
//!
//! Two-level verification (spec §10.1, same shape as the T6 gate it
//! replaces): the differential test below proves the model derives the SAME
//! challenges as a real spongefish `ProverState<PoseidonSponge>` driven
//! through the typed `ProverFs`/`FsAbsorb` vocabulary; the symbolic-layout
//! test in `symbolic_traits::verifier_fs` then proves `FieldAlignedLayout`'s
//! challenge ASTs evaluate to the model's challenges.

use ark_bn254::Fr;
use ark_ff::Zero;
use jolt_transcript::{
    poseidon_domain_separator_msgs, push_byte_rule_units, push_commitments_frame_header,
    push_field_frame_units,
};
use light_poseidon::{Poseidon, PoseidonHasher};

/// Concrete reimplementation of the `U = Fr` `PoseidonSponge` compression
/// chain.
pub struct ConcreteFieldSponge {
    hasher: Poseidon<Fr>,
    state: Fr,
}

impl ConcreteFieldSponge {
    #[expect(clippy::expect_used)]
    pub fn new() -> Self {
        Self {
            hasher: Poseidon::<Fr>::new_circom(3).expect("width-4 init"),
            state: Fr::zero(),
        }
    }

    #[expect(clippy::expect_used)]
    fn permute(&mut self, a: Fr, b: Fr) {
        self.state = self
            .hasher
            .hash(&[self.state, a, b])
            .expect("poseidon hash");
    }

    /// One `DuplexSpongeInterface::absorb(units)`: unit pairs fed through the
    /// permutation, zero-padding an odd tail. Every call starts a fresh pair
    /// (the sponge has no buffering), exactly like the real sponge.
    pub fn absorb_units(&mut self, units: &[Fr]) {
        for pair in units.chunks(2) {
            let a = pair[0];
            let b = pair.get(1).copied().unwrap_or_else(Fr::zero);
            self.permute(a, b);
        }
    }

    /// One squeezed native unit = exactly one permute; the challenge is the
    /// new state (`NativeChallenge` identity decode).
    pub fn squeeze_unit(&mut self) -> Fr {
        self.permute(Fr::zero(), Fr::zero());
        self.state
    }
}

impl Default for ConcreteFieldSponge {
    fn default() -> Self {
        Self::new()
    }
}

/// One high-level transcript op, as the verifier issues it through the typed
/// `FsAbsorb`/`ProverFs`/`VerifierFs` vocabulary. Each op's unit encoding is
/// the codec's (spec §4.2):
#[derive(Clone, Debug)]
pub enum HighLevelOp {
    /// `absorb` / `absorb_bytes` / `absorb_commitment` / `absorb_commitment_bytes`
    /// / `read_slice`: the byte rule `[Fr(2L), ceil(L/31) 31-byte-LE chunks]`
    /// over the value's bytes ([`jolt_transcript::RawBytesMsg`]).
    AbsorbBytes(Vec<u8>),
    /// `absorb_scalar(s)` / `write_scalars`+`read_scalars`: the count-led
    /// field frame `[Fr(2k+1), e₁, …, e_k]` ([`jolt_transcript::FieldFrameMsg`]).
    AbsorbScalars(Vec<Fr>),
    /// `write_commitments`+`read_commitments`: frame count unit `Fr(2k+1)`
    /// (zero-padded to a pair) then one byte-rule group per commitment's
    /// canonical compressed bytes ([`jolt_transcript::CommitmentsMsg`]).
    AbsorbCommitments(Vec<Vec<u8>>),
    /// One full-field `Fr` challenge: one squeezed native unit.
    ChallengeFr,
}

impl HighLevelOp {
    /// The complete unit stream (leading tag + payload + even padding) this
    /// op absorbs, built with the codec's exported unit builders.
    pub fn units(&self) -> Vec<Fr> {
        let mut units = Vec::new();
        match self {
            Self::AbsorbBytes(bytes) => push_byte_rule_units(&mut units, bytes),
            Self::AbsorbScalars(elems) => push_field_frame_units(&mut units, elems),
            Self::AbsorbCommitments(groups) => {
                push_commitments_frame_header(&mut units, groups.len());
                for bytes in groups {
                    push_byte_rule_units(&mut units, bytes);
                }
            }
            Self::ChallengeFr => unreachable!("ChallengeFr is a squeeze, not an absorb"),
        }
        units
    }
}

/// Replay a sequence of high-level ops through the concrete model, returning
/// the `Fr` challenges. Construction seeds the sponge with the SAME three
/// domain-separator byte strings the native factories absorb
/// ([`poseidon_domain_separator_msgs`]), each under the byte rule.
pub fn model_challenges(session: &[u8], instance: &[u8; 32], ops: &[HighLevelOp]) -> Vec<Fr> {
    let mut sponge = ConcreteFieldSponge::new();
    for msg in poseidon_domain_separator_msgs(session, *instance) {
        sponge.absorb_units(HighLevelOp::AbsorbBytes(msg.0).units().as_slice());
    }
    let mut out = Vec::new();
    for op in ops {
        match op {
            HighLevelOp::ChallengeFr => out.push(sponge.squeeze_unit()),
            absorb => sponge.absorb_units(&absorb.units()),
        }
    }
    out
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use jolt_core::transcript_msgs::{FsAbsorb, FsChallenge, ProverFs};
    use jolt_transcript::{prover_transcript, PoseidonSponge};

    /// THE LEVEL-1 ORACLE GATE (spec §10.1): the concrete model reproduces a
    /// real `ProverState<PoseidonSponge>`'s `Fr` challenges across a mixed
    /// schedule driven through the typed `ProverFs`/`FsAbsorb` vocabulary —
    /// single scalars, scalar frames, commitment frames (incl. the empty
    /// advice-presence frame), a lone commitment absorb, raw byte messages,
    /// and back-to-back challenges. Proves the field-aligned unit schedule
    /// (domain sep + tagged messages + 1-permute squeeze) is understood
    /// exactly; the in-circuit `FieldAlignedLayout` targets this oracle.
    #[test]
    fn model_matches_real_field_aligned_poseidon_transcript() {
        let mut rng = ark_std::test_rng();
        let session = b"Jolt";
        let instance = [0x5Cu8; 32];

        let scalars: Vec<Fr> = (0..3).map(|_| Fr::rand(&mut rng)).collect();
        let frame: Vec<Fr> = (0..5).map(|_| Fr::rand(&mut rng)).collect();
        // Dory commitments are GT (Fq12) elements: 384 canonical bytes each.
        let commitments: Vec<ark_bn254::Fq12> =
            (0..2).map(|_| ark_bn254::Fq12::rand(&mut rng)).collect();
        let ser = |c: &ark_bn254::Fq12| {
            let mut b = Vec::new();
            ark_serialize::CanonicalSerialize::serialize_compressed(c, &mut b).unwrap();
            assert_eq!(b.len(), 384, "Dory GT must serialize to 384 bytes");
            b
        };

        let mut real = prover_transcript(session, instance, PoseidonSponge::default());
        let mut ops: Vec<HighLevelOp> = Vec::new();
        let mut real_challenges: Vec<Fr> = Vec::new();
        let challenge = |real: &mut _, ops: &mut Vec<HighLevelOp>| {
            let c: Fr = FsChallenge::<Fr>::challenge_field(real);
            ops.push(HighLevelOp::ChallengeFr);
            c
        };

        // Commitments frame (write_commitments → read_commitments).
        ProverFs::<Fr>::write_commitments(&mut real, &commitments);
        ops.push(HighLevelOp::AbsorbCommitments(
            commitments.iter().map(ser).collect(),
        ));
        // Empty commitments frame (the absent untrusted-advice presence frame).
        ProverFs::<Fr>::write_commitments::<ark_bn254::Fq12>(&mut real, &[]);
        ops.push(HighLevelOp::AbsorbCommitments(Vec::new()));
        real_challenges.push(challenge(&mut real, &mut ops));

        // Single-scalar absorbs (sumcheck input claims / flushed opening claims).
        for s in &scalars {
            FsAbsorb::absorb_scalar(&mut real, s);
            ops.push(HighLevelOp::AbsorbScalars(vec![*s]));
        }
        real_challenges.push(challenge(&mut real, &mut ops));

        // A scalar frame (write_scalars → read_scalars) then two back-to-back
        // challenges.
        ProverFs::<Fr>::write_scalars(&mut real, &frame);
        ops.push(HighLevelOp::AbsorbScalars(frame.clone()));
        real_challenges.push(challenge(&mut real, &mut ops));
        real_challenges.push(challenge(&mut real, &mut ops));

        // A multi-scalar absorb (absorb_scalars: one count-led frame).
        FsAbsorb::absorb_scalars(&mut real, &scalars);
        ops.push(HighLevelOp::AbsorbScalars(scalars.clone()));
        real_challenges.push(challenge(&mut real, &mut ops));

        // A lone commitment absorb (trusted commitments: byte rule, NO frame
        // count) and a raw byte message.
        FsAbsorb::absorb_commitment(&mut real, &commitments[0]);
        ops.push(HighLevelOp::AbsorbBytes(ser(&commitments[0])));
        FsAbsorb::absorb_bytes(&mut real, b"jolt-model-test");
        ops.push(HighLevelOp::AbsorbBytes(b"jolt-model-test".to_vec()));
        real_challenges.push(challenge(&mut real, &mut ops));

        // An empty byte message ([Fr(0)] — distinct from the empty
        // commitments frame's [Fr(1)] absorbed above).
        FsAbsorb::absorb_bytes(&mut real, &[]);
        ops.push(HighLevelOp::AbsorbBytes(Vec::new()));
        real_challenges.push(challenge(&mut real, &mut ops));

        let modeled = model_challenges(session, &instance, &ops);
        assert_eq!(
            modeled, real_challenges,
            "field-aligned model diverged from the real PoseidonSponge transcript"
        );
    }

    /// Negative control: a wrongly-regrouped schedule (scalar frame absorbed
    /// without its count unit) must NOT match — the tagged encoding is
    /// load-bearing, not coincidental.
    #[test]
    fn model_negative_control_untagged_frame_diverges() {
        let mut rng = ark_std::test_rng();
        let session = b"Jolt";
        let instance = [0x5Cu8; 32];
        let frame: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();

        let mut real = prover_transcript(session, instance, PoseidonSponge::default());
        ProverFs::<Fr>::write_scalars(&mut real, &frame);
        let real_c: Fr = FsChallenge::<Fr>::challenge_field(&mut real);

        // Untagged variant: absorb the raw elements with no leading count unit.
        let mut sponge = ConcreteFieldSponge::new();
        for msg in poseidon_domain_separator_msgs(session, instance) {
            sponge.absorb_units(HighLevelOp::AbsorbBytes(msg.0).units().as_slice());
        }
        sponge.absorb_units(&frame);
        let wrong_c = sponge.squeeze_unit();

        assert_ne!(
            wrong_c, real_c,
            "untagged absorb also matched — the count tag would be vacuous"
        );
    }
}
