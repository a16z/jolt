use jolt_field::Fr;
use jolt_transcript::{AppendToTranscript, Keccak256Transcript, Transcript};

use super::{AssemblyStatement, Commitment, StreamError, STREAM_LABEL};

pub(crate) struct CountingKeccakTranscript {
    inner: Keccak256Transcript<Fr>,
    pub(crate) hashes: usize,
}

impl Default for CountingKeccakTranscript {
    fn default() -> Self {
        Self::new(b"")
    }
}

impl Transcript for CountingKeccakTranscript {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: Keccak256Transcript::new(label),
            hashes: 1,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.hashes += 1;
        self.inner.append_bytes(bytes);
    }

    fn challenge(&mut self) -> Fr {
        self.hashes += 1;
        self.inner.challenge()
    }

    fn challenge_scalar(&mut self) -> Fr {
        self.hashes += 1;
        self.inner.challenge_scalar()
    }

    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
}

pub fn commitment_prefix_challenges(
    key_digest: &[u8; 32],
    public_statement: &[Fr],
    phases: &[(&[Commitment], usize)],
) -> Vec<Fr> {
    let mut transcript = Keccak256Transcript::<Fr>::new(STREAM_LABEL);
    transcript.append_bytes(key_digest);
    for value in public_statement {
        transcript.append(value);
    }
    let mut challenges = Vec::new();
    for &(commitments, challenge_count) in phases {
        absorb_commitments(commitments, &mut transcript);
        challenges.extend((0..challenge_count).map(|_| transcript.challenge()));
    }
    challenges
}

pub(crate) fn assembly_transcript<T: Transcript<Challenge = Fr>>(
    key_digest: &[u8; 32],
    public_statement: &[Fr],
    commitments: &[Commitment],
    statement: &AssemblyStatement,
) -> Result<(T, Vec<Fr>), StreamError> {
    let mut transcript = T::new(STREAM_LABEL);
    transcript.append_bytes(key_digest);
    for value in public_statement {
        transcript.append(value);
    }
    let mut challenges = Vec::new();
    let mut start = 0usize;
    for phase in &statement.commitment_phases {
        let end = start
            .checked_add(phase.group_count)
            .ok_or(StreamError::StageCount)?;
        let phase_commitments = commitments.get(start..end).ok_or(StreamError::StageCount)?;
        absorb_commitments(phase_commitments, &mut transcript);
        challenges.extend((0..phase.challenge_count).map(|_| transcript.challenge()));
        start = end;
    }
    if start != commitments.len() {
        return Err(StreamError::StageCount);
    }
    Ok((transcript, challenges))
}

fn absorb_commitments<T: Transcript<Challenge = Fr>>(
    commitments: &[Commitment],
    transcript: &mut T,
) {
    for commitment in commitments {
        commitment.append_to_transcript(transcript);
    }
}
