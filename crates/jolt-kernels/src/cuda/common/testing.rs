#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test scaffolding: device operations and fixture errors fail loudly"
)]

use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltCommittedPolynomial, JoltPolynomialId};
use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::UnivariatePoly;
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing};
use jolt_riscv::{JoltInstructionRow, RV64IMAC_JOLT};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
use jolt_witness::{
    ChunkVisitor, FixedBackend, JoltWitnessOracle, JoltWitnessPlane, OneHotSource, ProgramSource,
    RowSource, Shape, WitnessError,
};
use proptest::prelude::*;

use crate::reference::ReferenceBackend;
use crate::{PrepareKernel, ProofSession, ProverInputs};

pub fn fr(seed: u64) -> Fr {
    Fr::from_u64(seed.wrapping_mul(2_654_435_761) % 1_000_003 + 1)
}

pub fn arb_point(len: usize) -> impl Strategy<Value = Vec<Fr>> {
    proptest::collection::vec(any::<u64>().prop_map(fr), len)
}

pub struct FixedPlane {
    columns: FixedBackend<Fr>,
    program: JoltProgramPreprocessing,
    label: &'static str,
    log_t: Option<usize>,
}

impl FixedPlane {
    pub fn with_log_t(
        columns: FixedBackend<Fr>,
        label: &'static str,
        log_t: Option<usize>,
    ) -> Self {
        Self {
            columns,
            log_t,
            program: JoltProgramPreprocessing {
                bytecode: BytecodePreprocessing::preprocess(
                    vec![JoltInstructionRow::default()],
                    0,
                    RV64IMAC_JOLT,
                )
                .expect("bytecode fixture"),
                ram: RAMPreprocessing::default(),
                memory_layout: MemoryLayout::default(),
                max_padded_trace_length: 1,
            },
            label,
        }
    }
}

impl JoltWitnessOracle<Fr> for FixedPlane {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        self.columns.shape(id)
    }

    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<Fr>, WitnessError> {
        self.columns.oracle_table(id)
    }

    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        self.columns.committed_order()
    }
}

impl RowSource for FixedPlane {
    fn visit_chunks(
        &self,
        _range: std::ops::Range<usize>,
        _chunk_size: usize,
        _visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        Err(WitnessError::InvalidWitnessData {
            label: self.label,
            reason: "this relation's fixture serves oracle columns only, not trace rows".to_owned(),
        })
    }
}

impl ProgramSource for FixedPlane {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        &self.program
    }
}

impl OneHotSource for FixedPlane {
    fn hot_indices(&self, id: JoltPolynomialId) -> Result<Vec<Option<usize>>, WitnessError> {
        let (log_k, cycles) = self.one_hot_dimensions(id)?;
        let grid = JoltWitnessOracle::<Fr>::oracle_table(self, id)?;
        let mut indices = vec![None; cycles];
        for address in 0..(1usize << log_k) {
            for cycle in 0..cycles {
                if grid[address * cycles + cycle] != Fr::from_u64(0) {
                    if indices[cycle].is_some() {
                        return Err(WitnessError::InvalidWitnessData {
                            label: "cuda test plane",
                            reason: format!("cycle {cycle} of {id:?} has two hot addresses"),
                        });
                    }
                    indices[cycle] = Some(address);
                }
            }
        }
        Ok(indices)
    }

    fn hot_address_bits(&self, id: JoltPolynomialId) -> Result<usize, WitnessError> {
        self.one_hot_dimensions(id).map(|(log_k, _)| log_k)
    }
}

impl FixedPlane {
    fn one_hot_dimensions(&self, id: JoltPolynomialId) -> Result<(usize, usize), WitnessError> {
        let log_rows = self.columns.shape(id)?.log_rows;
        let log_t = self.log_t.ok_or(WitnessError::InvalidWitnessData {
            label: "cuda test plane",
            reason: "fixture declared no cycle count for its one-hot columns".to_owned(),
        })?;
        let log_k = log_rows
            .checked_sub(log_t)
            .ok_or(WitnessError::InvalidWitnessData {
                label: "cuda test plane",
                reason: format!("{id:?} has fewer rows than the declared cycle count"),
            })?;
        Ok((log_k, 1usize << log_t))
    }
}

pub struct RowPlane {
    inner: FixedPlane,
    rows: Vec<TraceRow>,
}

impl RowPlane {
    pub fn new(
        columns: FixedBackend<Fr>,
        label: &'static str,
        log_t: usize,
        rows: Vec<TraceRow>,
    ) -> Self {
        Self {
            inner: FixedPlane::with_log_t(columns, label, Some(log_t)),
            rows,
        }
    }
}

impl JoltWitnessOracle<Fr> for RowPlane {
    fn shape(&self, id: JoltPolynomialId) -> Result<Shape, WitnessError> {
        self.inner.shape(id)
    }

    fn oracle_table(&self, id: JoltPolynomialId) -> Result<Vec<Fr>, WitnessError> {
        JoltWitnessOracle::<Fr>::oracle_table(&self.inner, id)
    }

    fn committed_order(&self) -> Result<Vec<JoltCommittedPolynomial>, WitnessError> {
        self.inner.committed_order()
    }
}

impl ProgramSource for RowPlane {
    fn program_preprocessing(&self) -> &JoltProgramPreprocessing {
        self.inner.program_preprocessing()
    }
}

impl OneHotSource for RowPlane {
    fn hot_indices(&self, id: JoltPolynomialId) -> Result<Vec<Option<usize>>, WitnessError> {
        self.inner.hot_indices(id)
    }

    fn hot_address_bits(&self, id: JoltPolynomialId) -> Result<usize, WitnessError> {
        self.inner.hot_address_bits(id)
    }
}

impl RowSource for RowPlane {
    fn visit_chunks(
        &self,
        range: std::ops::Range<usize>,
        chunk_size: usize,
        visitor: &mut ChunkVisitor<'_>,
    ) -> Result<(), WitnessError> {
        if range.end > self.rows.len() {
            return Err(WitnessError::InvalidWitnessData {
                label: "cuda row plane",
                reason: format!(
                    "requested cycles {range:?} exceed the {} fixture rows",
                    self.rows.len()
                ),
            });
        }
        let env = WitnessEnv::new(self.inner.program_preprocessing());
        let mut start = range.start;
        while start < range.end {
            let end = (start + chunk_size).min(range.end);
            visitor(&self.rows[start..end], self.rows.get(end), &env)?;
            start = end;
        }
        Ok(())
    }
}

pub fn ram_read_cycle(address: u64, value: u64) -> tracer::instruction::Cycle {
    tracer::instruction::Cycle::LW(
        tracer::instruction::RISCVCycle::<tracer::instruction::lw::LW> {
            ram_access: tracer::instruction::RAMRead { address, value },
            ..Default::default()
        },
    )
}

pub fn ram_write_cycle(
    address: u64,
    pre_value: u64,
    post_value: u64,
) -> tracer::instruction::Cycle {
    tracer::instruction::Cycle::SW(
        tracer::instruction::RISCVCycle::<tracer::instruction::sw::SW> {
            ram_access: tracer::instruction::RAMWrite {
                address,
                pre_value,
                post_value,
            },
            ..Default::default()
        },
    )
}

pub fn ram_trace(log_t: usize, ram_k: usize) -> Vec<tracer::instruction::Cycle> {
    (0..1usize << log_t)
        .map(|cycle| {
            let word = 1 + (cycle as u64 * 5) % (ram_k as u64 - 1);
            let address = 8 * word;
            match cycle % 4 {
                0 => tracer::instruction::Cycle::NoOp,
                1 => ram_read_cycle(address, 900 + cycle as u64),
                2 => ram_write_cycle(address, 100 + cycle as u64, 700 + cycle as u64),
                _ => ram_write_cycle(address, 400 + cycle as u64, 400 + cycle as u64),
            }
        })
        .collect()
}

pub fn reference_input_claim<'a, R>(
    witness: &dyn JoltWitnessPlane<Fr>,
    make_inputs: impl Fn() -> ProverInputs<'a, Fr, R>,
) -> Fr
where
    R: ConcreteSumcheck<Fr> + 'a,
    ReferenceBackend: PrepareKernel<Fr, R>,
    SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
    SumcheckOutputClaims<Fr, R>: OutputClaims<Fr>,
    ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
{
    let mut probe = ReferenceBackend
        .prepare(&mut ProofSession::default(), witness, make_inputs())
        .expect("reference prepare for the input-claim probe");
    probe.prove_round(None, 0, Fr::from_u64(0)).map_or_else(
        |error| claim_from_round_check(&error),
        |poly| poly.evaluate(Fr::from_u64(0)) + poly.evaluate(Fr::from_u64(1)),
    )
}

fn claim_from_round_check(error: &SumcheckError<Fr>) -> Fr {
    match error {
        SumcheckError::RoundCheckFailed { actual, .. } => *actual,
        other => panic!("reference kernel failed on the fixture: {other:?}"),
    }
}

pub fn drive<K: ProveRounds<Fr> + ?Sized>(
    kernel: &mut K,
    input_claim: Fr,
    challenges: &[Fr],
) -> Vec<UnivariatePoly<Fr>> {
    let mut polys = Vec::new();
    let mut claim = input_claim;
    let mut bind = None;
    for (round, &challenge) in challenges.iter().enumerate() {
        let poly = kernel
            .prove_round(bind, round, claim)
            .expect("prove_round must succeed");
        claim = poly.evaluate(challenge);
        polys.push(poly);
        bind = Some(challenge);
    }
    kernel
        .finish_rounds(challenges[challenges.len() - 1])
        .expect("finish_rounds must succeed");
    polys
}
