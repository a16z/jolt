use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Index;
use std::sync::{Arc, Mutex};

use tracer::instruction::RV32IMCycle;

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::r1cs::builder::Constraint;
use crate::r1cs::inputs::JoltR1CSInputs;
use crate::r1cs::key::UniformSpartanKey;
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

impl<F: JoltField> Index<JoltR1CSInputs> for Openings<F> {
    type Output = F;

    fn index(&self, index: JoltR1CSInputs) -> &Self::Output {
        &self[&OpeningsKeys::SpartanZ(index)].1
    }
}

pub struct StateManager<'a, F: JoltField, ProofTranscript: Transcript> {
    pub T: usize,
    pub log_T: usize,
    pub challenges: Challenges<F>,
    pub prover_state: Option<ProverState<'a, F>>,
    pub verifier_state: Option<VerifierState<'a, F>>,
    pub openings: Arc<Mutex<Openings<F>>>,

    // Fields for Spartan outer sumcheck
    pub key: Option<&'a UniformSpartanKey<F>>,
    pub uniform_constraints: Option<Vec<Constraint>>,
    pub input_polys: Option<Vec<MultilinearPolynomial<F>>>,
    pub tau: Option<Vec<F>>,
    pub outer_sumcheck_claims: Option<(F, F, F)>, // (Az, Bz, Cz)
    
    // Phantom data to use the ProofTranscript type parameter
    _phantom: std::marker::PhantomData<ProofTranscript>,
}

pub struct Challenges<F: JoltField> {
    pub instruction_booleanity: F,
    pub instruction_hamming: F,
    pub instruction_read_raf: F,
}

pub struct ProverState<'a, F: JoltField> {
    trace: &'a [RV32IMCycle],
    pub eq_r_cycle: Vec<F>,
}

pub struct VerifierState<'a, F: JoltField> {
    trace: &'a [RV32IMCycle],
    pub eq_r_cycle: Vec<F>,
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug)]
pub enum OpeningsKeys {
    SpartanZ(JoltR1CSInputs),
    InstructionTypeFlag(usize),
    InstructionRa(usize),
}

impl<'a, F: JoltField, ProofTranscript: Transcript> StateManager<'a, F, ProofTranscript> {
    pub fn new(
        T: usize,
        log_T: usize,
        challenges: Challenges<F>,
        prover_state: Option<ProverState<'a, F>>,
        verifier_state: Option<VerifierState<'a, F>>,
        openings: Arc<Mutex<Openings<F>>>,
        key: Option<&'a UniformSpartanKey<F>>,
        uniform_constraints: Option<Vec<Constraint>>,
        input_polys: Option<Vec<MultilinearPolynomial<F>>>,
        tau: Option<Vec<F>>,
        outer_sumcheck_claims: Option<(F, F, F)>,
    ) -> Self {
        Self {
            T,
            log_T,
            challenges,
            prover_state,
            verifier_state,
            openings,
            key,
            uniform_constraints,
            input_polys,
            tau,
            outer_sumcheck_claims,
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
        self.key.expect("Spartan key not set")
    }

    pub fn uniform_constraints(&self) -> &[Constraint] {
        self.uniform_constraints
            .as_ref()
            .expect("Uniform constraints not set")
    }

    pub fn input_polys(&self) -> &[MultilinearPolynomial<F>] {
        self.input_polys
            .as_ref()
            .expect("Input polynomials not set")
    }

    pub fn tau(&self) -> &[F] {
        self.tau.as_ref().expect("Tau not set")
    }

    pub fn outer_sumcheck_claims(&self) -> (F, F, F) {
        self.outer_sumcheck_claims
            .expect("Outer sumcheck claims not set")
    }
}
