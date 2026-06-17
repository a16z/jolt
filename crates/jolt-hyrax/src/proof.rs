use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct HyraxOpeningProof<F> {
    pub combined_row: Vec<F>,
    pub combined_row_opening_scalar: F,
}

impl<F: AppendToTranscript> AppendToTranscript for HyraxOpeningProof<F> {
    fn append_to_transcript<T: Transcript>(&self, transcript: &mut T) {
        transcript.append(&LabelWithCount(
            b"hyrax_opening_row",
            self.combined_row.len() as u64,
        ));
        for row_coordinate in &self.combined_row {
            transcript.append(row_coordinate);
        }
        transcript.append(&Label(b"hyrax_opening_scalar"));
        transcript.append(&self.combined_row_opening_scalar);
    }
}
