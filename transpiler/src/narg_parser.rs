//! Offline splitter for the spongefish NARG byte-string (non-ZK proofs).
//!
//! In non-ZK mode the NARG is a pure concatenation of self-delimiting frames
//! (8-byte LE length prefix + body): jolt-core writes NARG frames via THREE
//! operations (`jolt-core/src/transcript_msgs.rs` is the authoritative
//! inventory) — `write_scalars` for sumcheck/uni-skip round polys,
//! `write_commitments` for the witness + advice-presence frames, and
//! `write_slice` for the ZK/BlindFold data (out of scope here, non-ZK only).
//! All three produce `BytesMsg`-identical 8-byte-LE length framing, which is
//! why uniform frame-splitting is sound; absorbs and challenge squeezes append
//! no bytes. Frame *meaning* is deliberately not assigned here — it comes from
//! the symbolic replay's read order, which is the single source of protocol
//! truth (`notes/transpiler-deviations-from-spec.md` TDEV-1).

use std::fmt;

use jolt_transcript::read_length_prefixed_body;

/// Width of the `BytesMsg` little-endian length prefix
/// (`crates/jolt-transcript/src/codec.rs`).
pub const FRAME_LEN_PREFIX_BYTES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NargParseError {
    /// Guardrail 4 (spec §16): a ZK NARG carries extra frames (Pedersen round
    /// commitments, per-round degrees, output-claim commitments, BlindFold data).
    /// It would *split* fine — which is exactly why it must be refused explicitly:
    /// the non-ZK replay would silently mis-assign its frames.
    ZkProofUnsupported,
    /// Fewer than 8 bytes remained where a frame length prefix was expected.
    TruncatedLengthPrefix { offset: usize, remaining: usize },
    /// A frame body extends past the end of the NARG.
    TruncatedFrameBody {
        offset: usize,
        expected: u64,
        remaining: usize,
    },
}

impl fmt::Display for NargParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZkProofUnsupported => write!(
                f,
                "ZK proofs are not supported by the transpiler (non-ZK only; spec §16 guardrail 4)"
            ),
            Self::TruncatedLengthPrefix { offset, remaining } => write!(
                f,
                "truncated NARG: {remaining} byte(s) at offset {offset}, expected an 8-byte frame length prefix"
            ),
            Self::TruncatedFrameBody {
                offset,
                expected,
                remaining,
            } => write!(
                f,
                "truncated NARG: frame at offset {offset} declares {expected} byte(s) but only {remaining} remain"
            ),
        }
    }
}

impl std::error::Error for NargParseError {}

/// The NARG split into its ordered frames. Holds raw frame bodies only;
/// interpretation happens at replay time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedNarg {
    frames: Vec<Vec<u8>>,
}

impl ParsedNarg {
    pub fn frames(&self) -> &[Vec<u8>] {
        &self.frames
    }

    pub fn into_frames(self) -> Vec<Vec<u8>> {
        self.frames
    }

    /// Reassemble the exact original NARG bytes (T1 acceptance: byte-exact
    /// round-trip).
    pub fn reserialize(&self) -> Vec<u8> {
        let total: usize = self
            .frames
            .iter()
            .map(|f| FRAME_LEN_PREFIX_BYTES + f.len())
            .sum();
        let mut out = Vec::with_capacity(total);
        for frame in &self.frames {
            out.extend_from_slice(&(frame.len() as u64).to_le_bytes());
            out.extend_from_slice(frame);
        }
        out
    }
}

/// Split a non-ZK NARG into its frames. `zk_mode` must come from the proof's
/// `zk_mode` field; `true` is refused (see [`NargParseError::ZkProofUnsupported`]).
/// Every byte must belong to a complete frame — trailing or truncated bytes are
/// errors (the offline analogue of the verifier's `check_eof`).
pub fn parse_narg(narg: &[u8], zk_mode: bool) -> Result<ParsedNarg, NargParseError> {
    if zk_mode {
        return Err(NargParseError::ZkProofUnsupported);
    }
    let mut frames = Vec::new();
    let mut cursor = narg;
    while !cursor.is_empty() {
        let offset = narg.len() - cursor.len();
        let remaining = cursor.len();
        if remaining < FRAME_LEN_PREFIX_BYTES {
            return Err(NargParseError::TruncatedLengthPrefix { offset, remaining });
        }
        #[expect(clippy::unwrap_used)] // 8-byte slice into [u8; 8] is infallible
        let declared_len = u64::from_le_bytes(cursor[..FRAME_LEN_PREFIX_BYTES].try_into().unwrap());
        let frame = read_length_prefixed_body(&mut cursor).map_err(|_| {
            NargParseError::TruncatedFrameBody {
                offset,
                expected: declared_len,
                remaining: remaining - FRAME_LEN_PREFIX_BYTES,
            }
        })?;
        frames.push(frame.to_vec());
    }
    Ok(ParsedNarg { frames })
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use jolt_core::field::JoltField;
    use jolt_core::transcript_msgs::ProverFs;
    use jolt_transcript::{prover_transcript, Blake2b512};

    /// Frames produced by the REAL jolt-transcript `write_slice` path split and
    /// round-trip byte-exactly.
    #[test]
    fn real_write_slice_frames_round_trip() {
        let mut rng = test_rng();
        let frame_a: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let frame_b: Vec<Fr> = vec![Fr::random(&mut rng)];
        let frame_c: Vec<Fr> = (0..7).map(|_| Fr::random(&mut rng)).collect();

        let mut p = prover_transcript(b"narg-parser-test", [3u8; 32], Blake2b512::default());
        ProverFs::<Fr>::write_slice(&mut p, &frame_a);
        ProverFs::<Fr>::write_slice(&mut p, &frame_b);
        ProverFs::<Fr>::write_slice(&mut p, &frame_c);
        // An empty frame (e.g. the untrusted-advice "absent" presence frame).
        ProverFs::<Fr>::write_slice::<Fr>(&mut p, &[]);
        let narg = p.narg_string().to_vec();

        let parsed = parse_narg(&narg, false).unwrap();
        assert_eq!(parsed.frames().len(), 4);
        assert_eq!(parsed.frames()[0].len(), 3 * 32);
        assert_eq!(parsed.frames()[1].len(), 32);
        assert_eq!(parsed.frames()[2].len(), 7 * 32);
        assert!(parsed.frames()[3].is_empty());
        assert_eq!(parsed.reserialize(), narg, "round-trip must be byte-exact");
    }

    #[test]
    fn empty_narg_parses_to_zero_frames() {
        let parsed = parse_narg(&[], false).unwrap();
        assert_eq!(parsed.frames().len(), 0);
        assert!(parsed.reserialize().is_empty());
    }

    #[test]
    fn zk_mode_is_refused() {
        assert_eq!(
            parse_narg(&[], true),
            Err(NargParseError::ZkProofUnsupported)
        );
    }

    #[test]
    fn truncated_length_prefix_is_rejected() {
        let narg = [1u8, 2, 3]; // < 8 bytes of prefix
        assert_eq!(
            parse_narg(&narg, false),
            Err(NargParseError::TruncatedLengthPrefix {
                offset: 0,
                remaining: 3
            })
        );
    }

    #[test]
    fn truncated_frame_body_is_rejected() {
        let mut narg = 5u64.to_le_bytes().to_vec();
        narg.extend_from_slice(&[0u8; 2]); // body declares 5, only 2 present
        assert_eq!(
            parse_narg(&narg, false),
            Err(NargParseError::TruncatedFrameBody {
                offset: 0,
                expected: 5,
                remaining: 2
            })
        );
    }

    /// A huge declared length must error cleanly, not allocate.
    #[test]
    fn adversarial_length_does_not_allocate() {
        let narg = u64::MAX.to_le_bytes().to_vec();
        assert!(matches!(
            parse_narg(&narg, false),
            Err(NargParseError::TruncatedFrameBody { .. })
        ));
    }
}
