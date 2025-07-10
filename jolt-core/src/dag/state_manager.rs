use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Index, RangeFull};

use std::sync::{Arc, Mutex};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::r1cs::builder::CombinedUniformBuilder;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::r1cs::key::UniformSpartanKey;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::utils::transcript::Transcript;

pub type Endianness = bool;
pub const BIG_ENDIAN: Endianness = false;
pub const LITTLE_ENDIAN: Endianness = true;

#[derive(Clone, Debug)]
pub struct OpeningPoint<const E: Endianness, F: JoltField> {
    pub r: Vec<F>,
}

impl<const E: Endianness, F: JoltField> Index<usize> for OpeningPoint<E, F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.r[index]
    }
}

impl<const E: Endianness, F: JoltField> Index<RangeFull> for OpeningPoint<E, F> {
    type Output = [F];

    fn index(&self, _index: RangeFull) -> &Self::Output {
        &self.r[..]
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn split_at(&self, mid: usize) -> (&[F], &[F]) {
        self.r.split_at(mid)
    }
}

impl<const E: Endianness, F: JoltField> OpeningPoint<E, F> {
    pub fn new(r: Vec<F>) -> Self {
        Self { r }
    }

    pub fn endianness(&self) -> &'static str {
        if E == BIG_ENDIAN {
            "big"
        } else {
            "little"
        }
    }

    pub fn match_endianness<const SWAPPED_E: Endianness>(&self) -> OpeningPoint<SWAPPED_E, F>
    where
        F: Clone,
    {
        let mut reversed = self.r.clone();
        if E != SWAPPED_E {
            reversed.reverse();
        }
        OpeningPoint::<SWAPPED_E, F>::new(reversed)
    }
}

impl<F: JoltField> From<Vec<F>> for OpeningPoint<LITTLE_ENDIAN, F> {
    fn from(r: Vec<F>) -> Self {
        Self::new(r)
    }
}

impl<F: JoltField> From<Vec<F>> for OpeningPoint<BIG_ENDIAN, F> {
    fn from(r: Vec<F>) -> Self {
        Self::new(r)
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
    OuterSumcheckAz,        // Az claim from outer sumcheck
    OuterSumcheckBz,        // Bz claim from outer sumcheck
    OuterSumcheckCz,        // Cz claim from outer sumcheck
    OuterSumcheckRxVar,     // rx_var from outer sumcheck -- TODO(markosg04)where is this used ?
    PCSumcheckUnexpandedPC, // UnexpandedPC evaluation from PC sumcheck
    PCSumcheckPC,           // PC evaluation from PC sumcheck
}

pub type Openings<F> = HashMap<OpeningsKeys, (OpeningPoint<LITTLE_ENDIAN, F>, F)>;

pub trait OpeningsExt<F: JoltField> {
    fn get_spartan_z(&self, index: JoltR1CSInputs) -> F;
}

impl<F: JoltField> OpeningsExt<F> for Openings<F> {
    fn get_spartan_z(&self, index: JoltR1CSInputs) -> F {
        self.get(&OpeningsKeys::SpartanZ(index))
            .map(|(_, value)| *value)
            .unwrap_or(F::zero())
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum ProofKeys {
    SpartanOuterSumcheck,
    SpartanInnerSumcheck,
    SpartanShiftSumcheck,
}

pub enum ProofData<F: JoltField, ProofTranscript: Transcript> {
    Spartan(UniformSpartanProof<F, ProofTranscript>),
    SpartanSumcheck(SumcheckInstanceProof<F, ProofTranscript>),
}

pub type Proofs<F, ProofTranscript> = HashMap<ProofKeys, ProofData<F, ProofTranscript>>;

pub struct SpartanState<'a, F: JoltField> {
    pub spartan_key: Option<&'a UniformSpartanKey<F>>,
    pub constraint_builder: Option<&'a CombinedUniformBuilder<F>>,
    pub input_polys: Option<Vec<MultilinearPolynomial<F>>>,
}

pub struct StateManager<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
> {
    pub openings: Arc<Mutex<Openings<F>>>,
    pub prover_accumulator: Arc<Mutex<ProverOpeningAccumulator<F, PCS, ProofTranscript>>>,
    pub verifier_accumulator: Arc<Mutex<VerifierOpeningAccumulator<F, PCS, ProofTranscript>>>,
    pub prover_transcript: RefCell<&'a mut ProofTranscript>,
    pub verifier_transcript: RefCell<&'a mut ProofTranscript>,
    pub proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>,
    spartan_state: SpartanState<'a, F>,
}

impl<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    > StateManager<'a, F, ProofTranscript, PCS>
{
    pub fn new(
        openings: Arc<Mutex<Openings<F>>>,
        prover_accumulator: Arc<Mutex<ProverOpeningAccumulator<F, PCS, ProofTranscript>>>,
        verifier_accumulator: Arc<Mutex<VerifierOpeningAccumulator<F, PCS, ProofTranscript>>>,
        prover_transcript: &'a mut ProofTranscript,
        verifier_transcript: &'a mut ProofTranscript,
        proofs: Arc<Mutex<Proofs<F, ProofTranscript>>>,
        spartan_state: SpartanState<'a, F>,
    ) -> Self {
        Self {
            openings,
            prover_accumulator,
            verifier_accumulator,
            prover_transcript: RefCell::new(prover_transcript),
            verifier_transcript: RefCell::new(verifier_transcript),
            proofs,
            spartan_state,
        }
    }

    pub fn z(&self, idx: JoltR1CSInputs) -> F {
        use OpeningsExt;
        self.openings.lock().unwrap().get_spartan_z(idx)
    }

    pub fn openings(&self, idx: OpeningsKeys) -> F {
        self.openings.lock().unwrap().get(&idx).unwrap().1
    }

    pub fn openings_point(&self, idx: OpeningsKeys) -> OpeningPoint<LITTLE_ENDIAN, F> {
        self.openings.lock().unwrap().get(&idx).unwrap().0.clone()
    }

    pub fn set_spartan_data(
        &mut self,
        key: &'a UniformSpartanKey<F>,
        constraint_builder: &'a CombinedUniformBuilder<F>,
        input_polys: Vec<MultilinearPolynomial<F>>,
    ) {
        self.spartan_state.spartan_key = Some(key);
        self.spartan_state.constraint_builder = Some(constraint_builder);
        self.spartan_state.input_polys = Some(input_polys);
    }

    pub fn get_spartan_data(
        &self,
    ) -> (
        &'a UniformSpartanKey<F>,
        &'a CombinedUniformBuilder<F>,
        &Vec<MultilinearPolynomial<F>>,
    ) {
        (
            self.spartan_state.spartan_key.expect("Spartan key not set"),
            self.spartan_state
                .constraint_builder
                .expect("Constraint builder not set"),
            self.spartan_state
                .input_polys
                .as_ref()
                .expect("Input polys not set"),
        )
    }
}
