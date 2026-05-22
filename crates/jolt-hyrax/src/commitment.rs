use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "C: Serialize", deserialize = "C: for<'a> Deserialize<'a>"))]
pub struct HyraxCommitment<C> {
    pub rows: Vec<C>,
}

impl<C: AppendToTranscript> AppendToTranscript for HyraxCommitment<C> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&LabelWithCount(
            b"hyrax_commitment_rows",
            self.rows.len() as u64,
        ));
        for row in &self.rows {
            row.append_to_transcript(transcript);
        }
    }
}

impl<F, C> HomomorphicCommitment<F> for HyraxCommitment<C>
where
    F: Field,
    C: HomomorphicCommitment<F>,
{
    fn add(c1: &Self, c2: &Self) -> Self {
        match (c1.rows.is_empty(), c2.rows.is_empty()) {
            (true, true) => Self::default(),
            (true, false) => c2.clone(),
            (false, true) => c1.clone(),
            (false, false) => {
                assert_eq!(
                    c1.rows.len(),
                    c2.rows.len(),
                    "Hyrax commitment row count mismatch",
                );
                let rows = c1
                    .rows
                    .iter()
                    .zip(c2.rows.iter())
                    .map(|(left, right)| C::add(left, right))
                    .collect();
                Self { rows }
            }
        }
    }

    fn linear_combine(c1: &Self, c2: &Self, scalar: &F) -> Self {
        if c2.rows.is_empty() {
            return c1.clone();
        }

        if c1.rows.is_empty() {
            let rows = c2
                .rows
                .iter()
                .map(|right| C::linear_combine(&C::default(), right, scalar))
                .collect();
            return Self { rows };
        }

        assert_eq!(
            c1.rows.len(),
            c2.rows.len(),
            "Hyrax commitment row count mismatch",
        );
        let rows = c1
            .rows
            .iter()
            .zip(c2.rows.iter())
            .map(|(left, right)| C::linear_combine(left, right, scalar))
            .collect();
        Self { rows }
    }
}
