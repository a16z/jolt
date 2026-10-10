#![expect(clippy::unwrap_used, clippy::panic, reason = "test crate")]

use std::collections::BTreeMap;

use jolt_claims::protocols::composed::{ComposedOpeningId, ExternalId};
use jolt_claims::{
    derived, opening, Expr, InputClaims, OutputClaims, SumcheckChallenges, SymbolicSumcheck,
};
use jolt_field::{JoltField, One, Ring, Zero, F128};
use jolt_kernels::reference::naive::NaiveSumcheckProver;
use jolt_kernels::{
    KernelError, KernelSlots, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel,
    SumcheckKernelError, WitnessPlane,
};
use jolt_poly::{BindingOrder, CompressedPoly, Polynomial};
use jolt_prover::{impl_stage_prover, Proved, ProverError, StageProver as _};
use jolt_sumcheck::{
    ClearProof, ClearSumcheckRecorder, SequentialRounds, SumcheckError, SumcheckProof,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::stages::ids::{VerifierChallengeId, VerifierDerivedId};
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckBatch};
use jolt_verifier::VerifierError;

use ids::{ChallengeId, DerivedId, Draw, FamilyExpr, OpeningId, RelationId, VirtualPolynomial};

pub mod ids {
    use super::{ComposedOpeningId, Expr, ExternalId, VerifierChallengeId, VerifierDerivedId};

    pub const FAMILY: &str = "external-cubic";

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum RelationId {
        Full,
        Tail,
        Head,
        Alias,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum VirtualPolynomial {
        Sum,
        A,
        B,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum OpeningId {
        Virtual(VirtualPolynomial, RelationId),
    }

    impl OpeningId {
        pub fn virtual_polynomial(polynomial: VirtualPolynomial, relation: RelationId) -> Self {
            Self::Virtual(polynomial, relation)
        }
    }

    impl From<OpeningId> for ComposedOpeningId {
        fn from(id: OpeningId) -> Self {
            let OpeningId::Virtual(polynomial, relation) = id;
            let relation_index = match relation {
                RelationId::Full => 0,
                RelationId::Tail => 1,
                RelationId::Head => 2,
                RelationId::Alias => 3,
            };
            let polynomial_index = match polynomial {
                VirtualPolynomial::Sum => 0,
                VirtualPolynomial::A => 1,
                VirtualPolynomial::B => 2,
            };
            Self::External(ExternalId {
                family: FAMILY,
                index: 3 * relation_index + polynomial_index,
            })
        }
    }

    impl TryFrom<ComposedOpeningId> for OpeningId {
        type Error = ComposedOpeningId;

        fn try_from(id: ComposedOpeningId) -> Result<Self, Self::Error> {
            match id {
                ComposedOpeningId::External(ExternalId {
                    family: FAMILY,
                    index,
                }) if index < 12 => {
                    let relation = match index / 3 {
                        0 => RelationId::Full,
                        1 => RelationId::Tail,
                        2 => RelationId::Head,
                        3 => RelationId::Alias,
                        _ => return Err(id),
                    };
                    let polynomial = match index % 3 {
                        0 => VirtualPolynomial::Sum,
                        1 => VirtualPolynomial::A,
                        2 => VirtualPolynomial::B,
                        _ => return Err(id),
                    };
                    Ok(Self::Virtual(polynomial, relation))
                }
                ComposedOpeningId::Jolt(_)
                | ComposedOpeningId::FieldInline(_)
                | ComposedOpeningId::External(_) => Err(id),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum DerivedId {
        Weight,
    }

    impl From<DerivedId> for VerifierDerivedId {
        fn from(id: DerivedId) -> Self {
            let index = match id {
                DerivedId::Weight => 0,
            };
            Self::External(ExternalId {
                family: FAMILY,
                index,
            })
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub enum ChallengeId {
        Shift,
    }

    pub enum Draw {
        Shift,
    }

    impl From<Draw> for ChallengeId {
        fn from(draw: Draw) -> Self {
            match draw {
                Draw::Shift => Self::Shift,
            }
        }
    }

    impl From<ChallengeId> for VerifierChallengeId {
        fn from(id: ChallengeId) -> Self {
            let index = match id {
                ChallengeId::Shift => 0,
            };
            Self::External(ExternalId {
                family: FAMILY,
                index,
            })
        }
    }

    pub type FamilyExpr<F> = Expr<F, OpeningId, DerivedId, ChallengeId>;
}

#[derive(Clone, Debug, PartialEq, Eq, SumcheckChallenges)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
#[protocol(ids = crate::ids)]
pub struct Challenges<F> {
    #[challenge(Draw::Shift)]
    pub shift: F,
}

macro_rules! declare_claims {
    ($inputs:ident, $outputs:ident, $id:ident) => {
        #[derive(Clone, Debug, Default, PartialEq, Eq, InputClaims)]
        #[protocol(ids = crate::ids)]
        pub struct $inputs<C> {
            #[opening(Sum, from = $id)]
            pub sum: C,
        }

        #[derive(
            Clone, Debug, PartialEq, Eq, OutputClaims, serde::Serialize, serde::Deserialize,
        )]
        #[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
        #[protocol(ids = crate::ids)]
        #[relation($id)]
        pub struct $outputs<C> {
            #[opening(A)]
            pub a: C,
            #[opening(B)]
            pub b: C,
        }
    };
}

declare_claims!(FullInputs, FullOutputs, Full);
declare_claims!(TailInputs, TailOutputs, Tail);
declare_claims!(HeadInputs, HeadOutputs, Head);
declare_claims!(AliasInputs, AliasOutputs, Alias);

macro_rules! declare_relation {
    ($relation:ident, $symbolic:ident, $inputs:ident, $outputs:ident, $id:ident,
     head = $head:literal, aliases = $aliases:expr) => {
        #[derive(Clone)]
        pub struct $symbolic {
            rounds: usize,
        }

        impl SymbolicSumcheck for $symbolic {
            type RelationId = RelationId;
            type OpeningId = OpeningId;
            type DerivedId = DerivedId;
            type ChallengeId = ChallengeId;
            type Shape = usize;
            type Challenges<F> = Challenges<F>;
            type Inputs<C> = $inputs<C>;
            type Outputs<C> = $outputs<C>;

            fn new(rounds: usize) -> Self {
                Self { rounds }
            }
            fn id() -> RelationId {
                RelationId::$id
            }
            fn rounds(&self) -> usize {
                self.rounds
            }
            fn degree(&self) -> usize {
                3
            }
            fn input_expression<F: Ring>(&self) -> FamilyExpr<F> {
                opening(OpeningId::Virtual(VirtualPolynomial::Sum, RelationId::$id))
            }
            fn output_expression<F: Ring>(&self) -> FamilyExpr<F> {
                derived(DerivedId::Weight)
                    * opening(OpeningId::Virtual(VirtualPolynomial::A, RelationId::$id))
                    * opening(OpeningId::Virtual(VirtualPolynomial::B, RelationId::$id))
            }
        }

        #[derive(Clone)]
        pub struct $relation<F: JoltField> {
            symbolic: $symbolic,
            derived_delta: F,
        }

        impl<F: JoltField> $relation<F> {
            fn new(rounds: usize) -> Self {
                Self {
                    symbolic: $symbolic::new(rounds),
                    derived_delta: F::zero(),
                }
            }
        }

        impl<F: JoltField> ConcreteSumcheck<F> for $relation<F> {
            type Symbolic = $symbolic;
            fn symbolic(&self) -> &$symbolic {
                &self.symbolic
            }
            fn instance_point_offset(&self, batch_num_vars: usize) -> Result<usize, VerifierError> {
                if $head {
                    Ok(0)
                } else {
                    batch_num_vars.checked_sub(self.rounds()).ok_or_else(|| {
                        VerifierError::StageClaimSumcheckFailed {
                            stage: format!("{:?}", self.id()),
                            reason: "member exceeds batch dimension".to_owned(),
                        }
                    })
                }
            }
            fn derive_opening_points(
                &self,
                point: &[F],
                _inputs: &$inputs<Vec<F>>,
            ) -> Result<$outputs<Vec<F>>, VerifierError> {
                Ok($outputs {
                    a: point.to_vec(),
                    b: point.to_vec(),
                })
            }
            fn derive_output_term(
                &self,
                id: &DerivedId,
                _inputs: &$inputs<Vec<F>>,
                outputs: &$outputs<Vec<F>>,
                challenges: &Challenges<F>,
            ) -> Result<F, VerifierError> {
                match id {
                    DerivedId::Weight => Ok(challenges.shift
                        + outputs.a[0]
                        + F::from_u64(7) * outputs.a[self.rounds() - 1]
                        + self.derived_delta),
                }
            }
            fn aliased_output_openings() -> Vec<(OpeningId, OpeningId)> {
                $aliases
            }
        }
    };
}

declare_relation!(
    FullRelation,
    FullSymbolic,
    FullInputs,
    FullOutputs,
    Full,
    head = false,
    aliases = vec![]
);
declare_relation!(
    TailRelation,
    TailSymbolic,
    TailInputs,
    TailOutputs,
    Tail,
    head = false,
    aliases = vec![]
);
declare_relation!(
    HeadRelation,
    HeadSymbolic,
    HeadInputs,
    HeadOutputs,
    Head,
    head = true,
    aliases = vec![]
);
declare_relation!(
    AliasRelation,
    AliasSymbolic,
    AliasInputs,
    AliasOutputs,
    Alias,
    head = false,
    aliases = vec![(
        OpeningId::Virtual(VirtualPolynomial::A, RelationId::Alias),
        OpeningId::Virtual(VirtualPolynomial::A, RelationId::Full),
    )]
);

#[derive(Clone, SumcheckBatch)]
pub struct ToySumchecks<F: JoltField> {
    pub full: FullRelation<F>,
    pub tail: TailRelation<F>,
    pub head: HeadRelation<F>,
}

#[derive(SumcheckBatch)]
pub struct AliasSumchecks<F: JoltField> {
    pub full: FullRelation<F>,
    pub alias: AliasRelation<F>,
}

#[derive(Clone)]
struct MemberTables<F: JoltField> {
    a: Vec<F>,
    b: Vec<F>,
    d: Vec<F>,
}

impl<F: JoltField> MemberTables<F> {
    fn cube_sum(&self) -> F {
        self.d
            .iter()
            .zip(&self.a)
            .zip(&self.b)
            .map(|((d, a), b)| *d * *a * *b)
            .sum()
    }
}

pub struct ToyTables<F: JoltField> {
    members: BTreeMap<RelationId, MemberTables<F>>,
}

pub struct ToyPlane;

impl<F: JoltField> WitnessPlane<F> for ToyPlane {
    type Ref<'w>
        = &'w ToyTables<F>
    where
        F: 'w;
}

#[derive(Default)]
struct TablePrepare {
    omit: Option<VirtualPolynomial>,
}

macro_rules! prepare_relation {
    ($relation:ident, $id:ident) => {
        impl<F: JoltField> PrepareKernel<F, $relation<F>, ToyPlane> for TablePrepare {
            fn prepare(
                &self,
                _session: &mut ProofSession,
                witness: &ToyTables<F>,
                inputs: ProverInputs<'_, F, $relation<F>>,
            ) -> Result<Box<dyn SumcheckKernel<F, Relation = $relation<F>>>, KernelError<F>> {
                let tables = &witness.members[&RelationId::$id];
                let openings = [
                    (VirtualPolynomial::A, &tables.a),
                    (VirtualPolynomial::B, &tables.b),
                ]
                .into_iter()
                .filter(|(id, _)| Some(*id) != self.omit)
                .map(|(id, values)| {
                    (
                        OpeningId::Virtual(id, RelationId::$id),
                        Polynomial::new(values.clone()),
                    )
                })
                .collect();
                let derived =
                    BTreeMap::from([(DerivedId::Weight, Polynomial::new(tables.d.clone()))]);
                Ok(Box::new(NaiveSumcheckProver::new(
                    &inputs,
                    openings,
                    derived,
                    BindingOrder::HighToLow,
                )?))
            }
        }
    };
}

prepare_relation!(FullRelation, Full);
prepare_relation!(TailRelation, Tail);
prepare_relation!(HeadRelation, Head);
prepare_relation!(AliasRelation, Alias);

#[derive(KernelSlots)]
struct ToyKernels<F: JoltField> {
    full: Box<dyn PrepareKernel<F, FullRelation<F>, ToyPlane>>,
    tail: Box<dyn PrepareKernel<F, TailRelation<F>, ToyPlane>>,
    head: Box<dyn PrepareKernel<F, HeadRelation<F>, ToyPlane>>,
    alias: Box<dyn PrepareKernel<F, AliasRelation<F>, ToyPlane>>,
}

impl<F: JoltField> Default for ToyKernels<F> {
    fn default() -> Self {
        Self {
            full: Box::<TablePrepare>::default(),
            tail: Box::<TablePrepare>::default(),
            head: Box::<TablePrepare>::default(),
            alias: Box::<TablePrepare>::default(),
        }
    }
}

toy_sumchecks_members!(impl_stage_prover plane = ToyPlane,);

mod proving {
    use super::{
        AliasChallenges, AliasInputClaims, AliasInputPoints, AliasOutputClaims, AliasOutputPoints,
        AliasRelation, FullRelation, ToyPlane,
    };
    use core::ops::Deref;
    use jolt_field::JoltField;
    use jolt_prover::impl_stage_prover;

    pub struct AliasSumchecks<F: JoltField>(pub super::AliasSumchecks<F>);

    impl<F: JoltField> Deref for AliasSumchecks<F> {
        type Target = super::AliasSumchecks<F>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    alias_sumchecks_members!(impl_stage_prover plane = ToyPlane,);
}

type ToyProof = Proved<F128, ToySumchecks<F128>, F128>;
type AliasProof = Proved<F128, proving::AliasSumchecks<F128>, F128>;
type BinaryTranscript = Blake2bTranscript<F128>;

struct Fixture {
    batch: ToySumchecks<F128>,
    aliases: proving::AliasSumchecks<F128>,
    tables: ToyTables<F128>,
    kernels: ToyKernels<F128>,
    inputs: ToyInputClaims<F128>,
    alias_inputs: AliasInputClaims<F128>,
    challenges: ToyChallenges<F128>,
    alias_challenges: AliasChallenges<F128>,
}

impl Fixture {
    fn new() -> Self {
        let batch = ToySumchecks {
            full: FullRelation::new(5),
            tail: TailRelation::new(3),
            head: HeadRelation::new(3),
        };
        let aliases = proving::AliasSumchecks(AliasSumchecks {
            full: FullRelation::new(5),
            alias: AliasRelation::new(5),
        });
        let mut transcript = Blake2bTranscript::new(b"external-cubic-stage");
        let challenges: ToyChallenges<F128> = batch.draw_challenges(&mut transcript).unwrap();
        let mut alias_transcript = Blake2bTranscript::new(b"external-cubic-stage");
        let alias_challenges: AliasChallenges<F128> =
            aliases.draw_challenges(&mut alias_transcript).unwrap();
        let members: BTreeMap<_, _> = [
            (RelationId::Full, 5, challenges.full.shift),
            (RelationId::Tail, 3, challenges.tail.shift),
            (RelationId::Head, 3, challenges.head.shift),
            (RelationId::Alias, 5, alias_challenges.alias.shift),
        ]
        .into_iter()
        .enumerate()
        .map(|(member, (id, rounds, shift))| {
            let a = (0..1usize << rounds)
                .map(|row| {
                    F128::from_raw(
                        0x9876_5432_10ab_cdef_7654_3210_89ab_cdef
                            ^ ((row + 1) as u128 * (row + 17) as u128 * 0x123_4567),
                    )
                })
                .collect();
            let b = (0..1usize << rounds)
                .map(|row| {
                    F128::from_raw(
                        0xfedc_ba98_7654_3210_1234_5678_9abc_def0
                            ^ ((row + 3 + member) as u128 * (row + 29) as u128 * 0x765_4321),
                    )
                })
                .collect();
            let d = (0..1usize << rounds)
                .map(|row| {
                    shift
                        + F128::from_u64(((row >> (rounds - 1)) & 1) as u64)
                        + F128::from_u64(7) * F128::from_u64((row & 1) as u64)
                })
                .collect();
            (id, MemberTables { a, b, d })
        })
        .collect();
        let tables = ToyTables { members };
        let inputs = ToyInputClaims {
            full: FullInputs {
                sum: tables.members[&RelationId::Full].cube_sum(),
            },
            tail: TailInputs {
                sum: tables.members[&RelationId::Tail].cube_sum(),
            },
            head: HeadInputs {
                sum: tables.members[&RelationId::Head].cube_sum(),
            },
        };
        let alias_inputs = AliasInputClaims {
            full: FullInputs {
                sum: tables.members[&RelationId::Full].cube_sum(),
            },
            alias: AliasInputs {
                sum: tables.members[&RelationId::Alias].cube_sum(),
            },
        };
        Self {
            batch,
            aliases,
            tables,
            kernels: ToyKernels::default(),
            inputs,
            alias_inputs,
            challenges,
            alias_challenges,
        }
    }

    fn transcript(&self) -> BinaryTranscript {
        let mut transcript = BinaryTranscript::new(b"external-cubic-stage");
        let challenges = self.batch.draw_challenges(&mut transcript).unwrap();
        assert_eq!(challenges, self.challenges);
        transcript
    }

    fn prove(&self) -> Result<(ToyProof, BinaryTranscript), ProverError<F128>> {
        let mut transcript = self.transcript();
        let proof = self.batch.prove(
            &self.kernels,
            &mut ProofSession::default(),
            &mut SequentialRounds,
            &self.tables,
            &self.inputs,
            &self.batch.empty_input_points(),
            &self.challenges,
            ClearSumcheckRecorder::<F128, F128>::new(),
            &mut transcript,
        )?;
        Ok((proof, transcript))
    }

    fn prove_aliases(&self) -> (AliasProof, BinaryTranscript) {
        let mut transcript = BinaryTranscript::new(b"external-cubic-stage");
        let challenges = self.aliases.draw_challenges(&mut transcript).unwrap();
        assert_eq!(challenges, self.alias_challenges);
        let proof = self
            .aliases
            .prove(
                &self.kernels,
                &mut ProofSession::default(),
                &mut SequentialRounds,
                &self.tables,
                &self.alias_inputs,
                &self.aliases.empty_input_points(),
                &self.alias_challenges,
                ClearSumcheckRecorder::<F128, F128>::new(),
                &mut transcript,
            )
            .unwrap();
        (proof, transcript)
    }

    fn verify(
        &self,
        batch: &ToySumchecks<F128>,
        inputs: &ToyInputClaims<F128>,
        outputs: &ToyOutputClaims<F128>,
        proof: &SumcheckProof<F128, F128>,
    ) -> Result<(ToyOutputPoints<F128>, BinaryTranscript), VerifierError> {
        let mut transcript = self.transcript();
        let points = batch.verify_clear(
            inputs,
            &batch.empty_input_points(),
            &self.challenges,
            outputs,
            proof,
            &mut transcript,
            0,
        )?;
        batch.append_output_claims(&mut transcript, outputs);
        Ok((points, transcript))
    }

    fn verify_aliases(
        &self,
        outputs: &AliasOutputClaims<F128>,
        proof: &SumcheckProof<F128, F128>,
    ) -> Result<(AliasOutputPoints<F128>, BinaryTranscript), VerifierError> {
        let batch = &self.aliases.0;
        let mut transcript = Blake2bTranscript::new(b"external-cubic-stage");
        let challenges = batch.draw_challenges(&mut transcript).unwrap();
        let points = batch.verify_clear(
            &self.alias_inputs,
            &batch.empty_input_points(),
            &challenges,
            outputs,
            proof,
            &mut transcript,
            0,
        )?;
        batch.append_output_claims(&mut transcript, outputs);
        Ok((points, transcript))
    }

    fn coefficients(&self, inputs: &ToyInputClaims<F128>) -> [F128; 3] {
        let mut transcript = self.transcript();
        for sum in [inputs.full.sum, inputs.tail.sum, inputs.head.sum] {
            transcript.append_labeled(b"sumcheck_claim", &sum);
        }
        std::array::from_fn(|_| transcript.challenge_scalar())
    }

    fn assert_nondegenerate(&self, proof: &ToyProof) {
        let c = self.coefficients(&self.inputs);
        let r = &proof.output_points.full.a;
        let scales = [
            F128::one(),
            (F128::one() - r[0]) * (F128::one() - r[1]),
            (F128::one() - r[3]) * (F128::one() - r[4]),
        ];
        for ((id, point), (coefficient, scale)) in [
            (RelationId::Full, &r[..]),
            (RelationId::Tail, &r[2..]),
            (RelationId::Head, &r[..3]),
        ]
        .into_iter()
        .zip(c.into_iter().zip(scales))
        {
            let tables = &self.tables.members[&id];
            let a = Polynomial::new(tables.a.clone()).evaluate(point);
            let b = Polynomial::new(tables.b.clone()).evaluate(point);
            let d = Polynomial::new(tables.d.clone()).evaluate(point);
            for factor in [coefficient, scale, a, b, d] {
                assert!(!factor.is_zero());
            }
        }
    }
}

#[test]
fn external_binary_stage_matches_cube_sums_and_verifier_twin() {
    let fixture = Fixture::new();
    let (proof, prover_transcript) = fixture.prove().unwrap();
    let (points, verifier_transcript) = fixture
        .verify(
            &fixture.batch,
            &fixture.inputs,
            &proof.output_claims,
            &proof.recorded.proof,
        )
        .unwrap();
    assert_eq!(proof.output_points, points);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
    let r = &points.full.a;
    assert_eq!(points.tail.a, r[2..]);
    assert_eq!(points.head.a, r[..3]);
    for (id, sum, a, b, a_point, b_point) in [
        (
            RelationId::Full,
            fixture.inputs.full.sum,
            proof.output_claims.full.a,
            proof.output_claims.full.b,
            &points.full.a,
            &points.full.b,
        ),
        (
            RelationId::Tail,
            fixture.inputs.tail.sum,
            proof.output_claims.tail.a,
            proof.output_claims.tail.b,
            &points.tail.a,
            &points.tail.b,
        ),
        (
            RelationId::Head,
            fixture.inputs.head.sum,
            proof.output_claims.head.a,
            proof.output_claims.head.b,
            &points.head.a,
            &points.head.b,
        ),
    ] {
        let tables = &fixture.tables.members[&id];
        assert_eq!(
            sum,
            tables
                .d
                .iter()
                .zip(&tables.a)
                .zip(&tables.b)
                .map(|((d, a), b)| *d * *a * *b)
                .sum()
        );
        assert_eq!(a, Polynomial::new(tables.a.clone()).evaluate(a_point));
        assert_eq!(b, Polynomial::new(tables.b.clone()).evaluate(b_point));
    }
    assert!(FullRelation::<F128>::aliased_output_openings().is_empty());
    assert!(TailRelation::<F128>::aliased_output_openings().is_empty());
    assert!(HeadRelation::<F128>::aliased_output_openings().is_empty());
    fixture.assert_nondegenerate(&proof);
    let SumcheckProof::Clear(ClearProof::Compressed(rounds)) = &proof.recorded.proof else {
        panic!("clear recorder must produce compressed rounds")
    };
    assert_eq!(rounds.round_polynomials.len(), 5);
    assert!(rounds
        .round_polynomials
        .iter()
        .any(|round| round.coeffs_except_linear_term().len() == 3));
}

#[test]
fn external_verifier_rejects_changed_head_input_claim() {
    let fixture = Fixture::new();
    let (proof, _) = fixture.prove().unwrap();
    fixture.assert_nondegenerate(&proof);
    let mut inputs = fixture.inputs.clone();
    inputs.head.sum += F128::one();
    assert_ne!(inputs.head.sum, fixture.inputs.head.sum);
    assert!(!fixture.coefficients(&inputs)[2].is_zero());
    assert!(fixture
        .verify(
            &fixture.batch,
            &inputs,
            &proof.output_claims,
            &proof.recorded.proof,
        )
        .is_err());
}

#[test]
fn external_verifier_rejects_changed_tail_input_claim() {
    let fixture = Fixture::new();
    let (proof, _) = fixture.prove().unwrap();
    fixture.assert_nondegenerate(&proof);
    let mut inputs = fixture.inputs.clone();
    inputs.tail.sum += F128::one();
    assert_ne!(inputs.tail.sum, fixture.inputs.tail.sum);
    assert!(!fixture.coefficients(&inputs)[1].is_zero());
    assert!(fixture
        .verify(
            &fixture.batch,
            &inputs,
            &proof.output_claims,
            &proof.recorded.proof,
        )
        .is_err());
}

#[test]
fn external_verifier_rejects_changed_round_message() {
    let fixture = Fixture::new();
    let (proof, _) = fixture.prove().unwrap();
    fixture.assert_nondegenerate(&proof);
    let mut changed = proof.recorded.proof.clone();
    let SumcheckProof::Clear(ClearProof::Compressed(rounds)) = &mut changed else {
        panic!("clear recorder must produce compressed rounds")
    };
    let mut coefficients = rounds.round_polynomials[0]
        .coeffs_except_linear_term()
        .to_vec();
    coefficients[0] += F128::one();
    rounds.round_polynomials[0] = CompressedPoly::new(coefficients);
    // In characteristic two, changing c0 leaves the reconstructed c1 fixed,
    // so the perturbation evaluates to one at every round challenge.
    assert_eq!(F128::one() + F128::one(), F128::zero());
    assert!(fixture
        .verify(
            &fixture.batch,
            &fixture.inputs,
            &proof.output_claims,
            &changed
        )
        .is_err());
}

#[test]
fn external_verifier_rejects_changed_output_claim() {
    let fixture = Fixture::new();
    let (proof, _) = fixture.prove().unwrap();
    fixture.assert_nondegenerate(&proof);
    let mut outputs = proof.output_claims.clone();
    outputs.head.a += F128::one();
    let r = &proof.output_points.full.a;
    let scale = (F128::one() - r[3]) * (F128::one() - r[4]);
    let d = Polynomial::new(fixture.tables.members[&RelationId::Head].d.clone()).evaluate(&r[..3]);
    assert!(
        !(fixture.coefficients(&fixture.inputs)[2] * scale * d * proof.output_claims.head.b)
            .is_zero()
    );
    assert!(fixture
        .verify(
            &fixture.batch,
            &fixture.inputs,
            &outputs,
            &proof.recorded.proof,
        )
        .is_err());
}

#[test]
fn external_verifier_rejects_changed_derived_output_term() {
    let fixture = Fixture::new();
    let (proof, _) = fixture.prove().unwrap();
    fixture.assert_nondegenerate(&proof);
    let mut batch = fixture.batch.clone();
    batch.tail.derived_delta = F128::one();
    let r = &proof.output_points.full.a;
    let expected =
        Polynomial::new(fixture.tables.members[&RelationId::Tail].d.clone()).evaluate(&r[2..]);
    let changed = fixture.challenges.tail.shift + r[2] + F128::from_u64(7) * r[4] + F128::one();
    assert_eq!(changed - expected, F128::one());
    let scale = (F128::one() - r[0]) * (F128::one() - r[1]);
    assert!(!(fixture.coefficients(&fixture.inputs)[1]
        * scale
        * proof.output_claims.tail.a
        * proof.output_claims.tail.b)
        .is_zero());
    assert!(fixture
        .verify(
            &batch,
            &fixture.inputs,
            &proof.output_claims,
            &proof.recorded.proof
        )
        .is_err());
}

#[test]
fn external_newtype_stage_is_accepted_by_batch_verifier() {
    let fixture = Fixture::new();
    let (proof, prover_transcript) = fixture.prove_aliases();
    let (points, verifier_transcript) = fixture
        .verify_aliases(&proof.output_claims, &proof.recorded.proof)
        .unwrap();
    assert_eq!(proof.output_points, points);
    assert_eq!(points.full.a, points.alias.a);
    assert_eq!(proof.output_claims.full.a, proof.output_claims.alias.a);
    assert_eq!(prover_transcript.state(), verifier_transcript.state());
    for (id, sum, a, b, point) in [
        (
            RelationId::Full,
            fixture.alias_inputs.full.sum,
            proof.output_claims.full.a,
            proof.output_claims.full.b,
            &points.full.a,
        ),
        (
            RelationId::Alias,
            fixture.alias_inputs.alias.sum,
            proof.output_claims.alias.a,
            proof.output_claims.alias.b,
            &points.alias.a,
        ),
    ] {
        let tables = &fixture.tables.members[&id];
        assert_eq!(sum, tables.cube_sum());
        assert_eq!(a, Polynomial::new(tables.a.clone()).evaluate(point));
        assert_eq!(b, Polynomial::new(tables.b.clone()).evaluate(point));
    }
}

#[test]
fn external_alias_rejection_preserves_both_external_ids() {
    let fixture = Fixture::new();
    let (proof, _) = fixture.prove_aliases();
    let _ = fixture
        .verify_aliases(&proof.output_claims, &proof.recorded.proof)
        .unwrap();
    let mut outputs = proof.output_claims.clone();
    outputs.alias.a += F128::one();
    assert_ne!(outputs.alias.a, outputs.full.a);
    assert!(matches!(
        fixture.verify_aliases(&outputs, &proof.recorded.proof),
        Err(VerifierError::StageClaimOpeningMismatch {
            left: ComposedOpeningId::External(ExternalId {
                family: "external-cubic",
                index: 10
            }),
            right: ComposedOpeningId::External(ExternalId {
                family: "external-cubic",
                index: 1
            }),
            ..
        })
    ));
}

#[test]
fn external_prover_rejects_opening_table_drift_in_first_round() {
    let mut fixture = Fixture::new();
    let tables = fixture.tables.members.get_mut(&RelationId::Full).unwrap();
    assert!(!tables.d[0].is_zero());
    assert!(!tables.b[0].is_zero());
    tables.a[0] += F128::one();
    let actual = tables.cube_sum();
    let expected = fixture.inputs.full.sum;
    assert_ne!(actual, expected);
    assert!(matches!(
        fixture.prove().err().unwrap(),
        ProverError::Sumcheck(SumcheckError::RoundCheckFailed {
            round: 0, expected: e, actual: a,
        }) if e == expected && a == actual
    ));
}

#[test]
fn external_prover_rejects_derived_table_drift_after_binding() {
    let mut fixture = Fixture::new();
    let (honest, _) = fixture.prove().unwrap();
    let point = &honest.output_points.full.a;
    let tables = fixture.tables.members.get_mut(&RelationId::Full).unwrap();
    let original = Polynomial::new(tables.d.clone()).evaluate(point);
    for value in &mut tables.d {
        *value += F128::one();
    }
    assert_eq!(
        Polynomial::new(tables.d.clone()).evaluate(point) - original,
        F128::one()
    );
    fixture.inputs.full.sum = tables.cube_sum();
    assert!(matches!(
        fixture.prove().err().unwrap(),
        ProverError::Kernel(KernelError::SumcheckKernel(
            SumcheckKernelError::DerivedTableDrift {
                id: VerifierDerivedId::External(ExternalId { family: "external-cubic", index: 0 }),
                expected, got,
            }
        )) if got - expected == F128::one()
    ));
}

#[test]
fn external_preparer_reports_missing_external_opening_table() {
    let mut fixture = Fixture::new();
    fixture.kernels.full = Box::new(TablePrepare {
        omit: Some(VirtualPolynomial::B),
    });
    assert!(matches!(
        fixture.prove().err().unwrap(),
        ProverError::Kernel(KernelError::MissingOpeningTable {
            id: ComposedOpeningId::External(ExternalId {
                family: "external-cubic",
                index: 2
            }),
        })
    ));
}
