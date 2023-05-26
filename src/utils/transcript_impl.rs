use merlin::Transcript;

trait TranscriptImpl {
  fn append_message(&mut self, label: &'static [u8], message: &[u8]);
  fn append_u64(&mut self, label: &'static [u8], x: u64);
  fn challenge_bytes(&mut self, label: &'static [u8], dest: &mut [u8]);
}

struct MerlinTranscriptImpl {
  pub merlin_transcript: Transcript,
}

impl MerlinTranscriptImpl {
  pub fn new(label: &'static [u8]) -> MerlinTranscriptImpl {
    Self {
      merlin_transcript: Transcript::new(label),
    }
  }
}

impl TranscriptImpl for MerlinTranscriptImpl {
  fn append_message(&mut self, label: &'static [u8], message: &[u8]) {}

  fn append_u64(&mut self, label: &'static [u8], x: u64) {}

  fn challenge_bytes(&mut self, label: &'static [u8], dest: &mut [u8]) {}
}

struct TestTranscriptImpl {}

impl TestTranscriptImpl {
    pub fn new()
}

impl TranscriptImpl for TestTranscriptImpl {
  fn append_message(&mut self, label: &'static [u8], message: &[u8]) {}

  fn append_u64(&mut self, label: &'static [u8], x: u64) {}

  fn challenge_bytes(&mut self, label: &'static [u8], dest: &mut [u8]) {}
}
