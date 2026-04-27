use jolt_compiler::module::Module;
use jolt_compiler::VerifierSchedule;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_r1cs::R1csKey;
use serde::{Deserialize, Serialize};

/// Contains the PCS verifier setup, compiled verifier schedule, R1CS key,
/// and any preprocessed (program-derived, public) data the verifier needs
/// to evaluate preprocessed-polynomial MLEs.
///
/// Constructed once during preprocessing and reused across multiple proof
/// verifications for the same circuit. The schedule is the compiler's output
/// describing the verifier's computation — the verifier is a generic
/// interpreter over it.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct JoltVerifyingKey<F: Field, PCS: CommitmentScheme<Field = F>> {
    /// PCS verifier-side structured reference string.
    pub pcs_setup: PCS::VerifierSetup,
    /// Compiled verifier schedule: stages, claim formulas, sumcheck params.
    pub schedule: VerifierSchedule,
    /// Preprocessed R1CS key for evaluating matrix-vector products.
    pub r1cs_key: R1csKey<F>,
    /// Preprocessed program-derived data for verifier-side MLE evaluation.
    /// Public — derivable from the program ELF + memory layout.
    pub preprocessing: Preprocessing,
}

/// Public preprocessed data needed by the verifier to evaluate
/// preprocessed-polynomial MLEs (e.g. `RamInit` at a stage 4 address point).
///
/// All fields are derivable from the program ELF + IO layout; carrying them
/// in the verifying key avoids re-deriving them from the proof's public IO
/// at every verify call.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Preprocessing {
    /// `initial_ram_state[k]` = the initial value at remapped word index `k`,
    /// for `k ∈ [0, ram_K)`. Built by `jolt_host::ram::build_ram_states`
    /// from the program's data segment + IO advice regions.
    pub initial_ram_state: Vec<u64>,
}

impl<F: Field, PCS: CommitmentScheme<Field = F>> JoltVerifyingKey<F, PCS> {
    /// Construct from a compiled module, PCS verifier setup, and R1CS key.
    /// `preprocessing` defaults to empty — callers that need stage-4+
    /// `RamInit` evaluation must populate it via [`with_preprocessing`].
    pub fn new(module: &Module, pcs_setup: PCS::VerifierSetup, r1cs_key: R1csKey<F>) -> Self {
        Self {
            pcs_setup,
            schedule: module.verifier.clone(),
            r1cs_key,
            preprocessing: Preprocessing::default(),
        }
    }

    /// Attach preprocessed program-derived data needed for verifier-side
    /// MLE evaluation of public polynomials (e.g. `RamInit`).
    pub fn with_preprocessing(mut self, preprocessing: Preprocessing) -> Self {
        self.preprocessing = preprocessing;
        self
    }
}
