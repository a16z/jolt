use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Index;
use std::sync::{Arc, Mutex};

use tracer::instruction::RV32IMCycle;

use crate::field::JoltField;
use crate::jolt::vm::rv32i_vm::PCS;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::opening_proof::ProverOpeningAccumulator;
use crate::r1cs::builder::Constraint;
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

pub type Openings<F> = HashMap<OpeningsKeys, (OpeningPoint<LITTLE_ENDIAN, F>, F)>;
// openings accumulator, &mut transcript, Proofs

impl<F: JoltField> Index<JoltR1CSInputs> for Openings<F> {
    type Output = F;

    fn index(&self, index: JoltR1CSInputs) -> &Self::Output {
        &self[&OpeningsKeys::SpartanZ(index)].1
    }
}


pub struct StateManager<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>> {
    pub openings: Arc<Mutex<Openings<F>>>,
    pub prover_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
    pub verifier_accumulator: ProverOpeningAccumulator<F, PCS, ProofTranscript>,
    pub transcript: ProofTranscript,
    pub proofs: Proofs<F, ProofTranscript>
}

/// State for Spartan-related data
pub struct SpartanState<'a, F: JoltField> {
    pub key: Option<&'a UniformSpartanKey<F>>,
    pub uniform_constraints: Option<Vec<Constraint>>,
    pub input_polys: Option<Vec<MultilinearPolynomial<F>>>,
    pub tau: Option<Vec<F>>,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
    OuterSumcheckClaims, // (Az, Bz, Cz)
}

impl<'a, F: JoltField, ProofTranscript: Transcript> StateManager<F, ProofTranscript> {
    pub fn new(
        openings: Arc<Mutex<Openings<F>>>,
        key: Option<&'a UniformSpartanKey<F>>,
        uniform_constraints: Option<Vec<Constraint>>,
        input_polys: Option<Vec<MultilinearPolynomial<F>>>,
        tau: Option<Vec<F>>,
        outer_sumcheck_claims: Option<(F, F, F)>,
    ) -> Self {
        // Initialize openings with outer sumcheck claims if provided
        if let Some((az, bz, cz)) = outer_sumcheck_claims {
            openings.lock().unwrap().insert(
                OpeningsKeys::OuterSumcheckClaims,
                (
                    OpeningPoint::new(vec![az, bz, cz]),
                    az,
                ),
            );
        }
        
        Self {
            T,
            log_T,
            challenges,
            prover_state,
            verifier_state,
            openings,
            spartan_state: SpartanState {
                key,
                uniform_constraints,
                input_polys,
                tau,
            },
            proofs: Arc::new(Mutex::new(HashMap::new())),
            _phantom: PhantomData,
        }
    }

    pub fn prover_state(&self) -> &ProverState<F> {
        self.prover_state.as_ref().unwrap()
    }

    pub fn verifier_state(&self) -> &VerifierState<F> {
        self.verifier_state.as_ref().unwrap()
    }

    pub fn z(&self, idx: JoltR1CSInputs) -> F {
        self.openings(OpeningsKeys::SpartanZ(idx))
    }

    pub fn r_cycle(&self) -> Vec<F> {
        self.openings_point(OpeningsKeys::SpartanZ(JoltR1CSInputs::Imm))
            .r
    }

    pub fn r_address(&self) -> Vec<F> {
        todo!()
    }

    pub fn openings(&self, idx: OpeningsKeys) -> F {
        self.openings.lock().unwrap().get(&idx).unwrap().1
    }

    pub fn openings_point(&self, idx: OpeningsKeys) -> OpeningPoint<LITTLE_ENDIAN, F> {
        self.openings.lock().unwrap().get(&idx).unwrap().0.clone()
    }

    pub fn trace(&self) -> &'a [RV32IMCycle] {
        self.prover_state.as_ref().unwrap().trace
    }

    // Getters for Spartan outer sumcheck
    pub fn spartan_key(&self) -> &UniformSpartanKey<F> {
        self.spartan_state.key.expect("Spartan key not set")
    }

    pub fn uniform_constraints(&self) -> &[Constraint] {
        self.spartan_state.uniform_constraints
            .as_ref()
            .expect("Uniform constraints not set")
    }

    pub fn input_polys(&self) -> &[MultilinearPolynomial<F>] {
        self.spartan_state.input_polys
            .as_ref()
            .expect("Input polynomials not set")
    }

    pub fn tau(&self) -> &[F] {
        self.spartan_state.tau.as_ref().expect("Tau not set")
    }

    pub fn outer_sumcheck_claims(&self) -> (F, F, F) {
        let openings = self.openings.lock().unwrap();
        if let Some((opening_point, _)) = openings.get(&OpeningsKeys::OuterSumcheckClaims) {
            let values = &opening_point.r;
            (values[0], values[1], values[2])
        } else {
            panic!("Outer sumcheck claims not set")
        }
    }
}

/// Enum to identify different proof types in the HashMap
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub enum ProofKeys {
    /// Stage 1: Spartan proof for R1CS constraints
    SpartanOuterSumcheck,
    SpartanInnerSumcheck,
    SpartanShiftSumcheck,
    /// Stage 2: Read-write sumcheck proofs
    RegistersReadWrite(usize), // Index for multiple read-write proofs
    RamReadWrite(usize),
    /// Stage 3: Instruction lookups
    InstructionLookups(usize),
}

/// Type alias for proof storage
pub enum ProofData<F: JoltField, ProofTranscript: Transcript> {
    Spartan(UniformSpartanProof<F, ProofTranscript>),
    Sumcheck(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, [F; 3]),
    // Add more proof types as needed
}

/// Type alias for the proofs HashMap
pub type Proofs<F, ProofTranscript> = HashMap<ProofKeys, ProofData<F, ProofTranscript>>;
