use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint, PolynomialId,
    ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{
    InputClaimConstraint, OutputClaimConstraint, ProductTerm, ValueSource,
};
use crate::subprotocols::sumcheck_claim::{
    CachedPointRef, ChallengePart, Claim, ClaimExpr, InputOutputClaims, SumcheckFrontend,
    VerifierEvaluablePolynomial,
};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::zkvm::instruction::{CircuitFlags, Flags, JoltTraceCycle};
use crate::zkvm::witness::VirtualPolynomial;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use tracer::instruction::Cycle;

// RAM activation-booleanity sumcheck (packed path)
//
// Digit-zero virtualization derives the RAM activation as
// `M_RAM = Load + Store` (`specs/digit-zero-virtualization.md`). This
// sumcheck binds the two flag columns at the stage-6b cycle point, proving
//   0 = Σ_j eq(r_cycle, j) · (B(j)^2 − B(j)),    B := Load + Store,
// and produces the `OpFlags(Load)`/`OpFlags(Store)` openings the stage-7
// digit-zero baselines consume.
//
// WARNING: the check is deliberately a single booleanity on the *sum*, not a
// γ-batch of per-flag legs. The flag columns are virtual — never committed —
// so a prover chooses them after every challenge is drawn, and a
// γ-combination of independent legs has non-Boolean solutions for any fixed
// γ (`L² − L = c`, `S² − S = −c/γ`). `B² = B` has only Boolean roots
// pointwise regardless of when the columns are chosen; only the sum flows
// into the reconstruction, so the split between the two openings is
// deliberately unconstrained.

/// Degree bound of the sumcheck round polynomials in
/// [`ActivationBooleanitySumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative, Clone)]
pub struct ActivationBooleanitySumcheckParams<F: JoltField> {
    pub r_cycle: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> ActivationBooleanitySumcheckParams<F> {
    pub fn new(opening_accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        Self { r_cycle }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for ActivationBooleanitySumcheckParams<F> {
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self, _: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn normalize_opening_point(
        &self,
        challenges: &[<F as JoltField>::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        InputClaimConstraint::default()
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let load = OpeningId::virt(
            VirtualPolynomial::OpFlags(CircuitFlags::Load),
            SumcheckId::RamActivationBooleanity,
        );
        let store = OpeningId::virt(
            VirtualPolynomial::OpFlags(CircuitFlags::Store),
            SumcheckId::RamActivationBooleanity,
        );

        // eq·(B² − B) with B = load + store, expanded to sum-of-products:
        // eq·(L·L + L·S + S·L + S·S) − eq·(L + S).
        let square = |a, b| {
            ProductTerm::scaled(
                ValueSource::Challenge(0),
                vec![ValueSource::Opening(a), ValueSource::Opening(b)],
            )
        };
        let terms = vec![
            square(load, load),
            square(load, store),
            square(store, load),
            square(store, store),
            ProductTerm::scaled(ValueSource::Challenge(1), vec![ValueSource::Opening(load)]),
            ProductTerm::scaled(ValueSource::Challenge(1), vec![ValueSource::Opening(store)]),
        ];

        Some(OutputClaimConstraint::sum_of_products(terms))
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        let r_cycle_final = self.normalize_opening_point(sumcheck_challenges);

        let eq_eval: F = EqPolynomial::<F>::mle(
            &r_cycle_final.r.iter().cloned().rev().collect::<Vec<_>>(),
            &self
                .r_cycle
                .r
                .iter()
                .cloned()
                .rev()
                .collect::<Vec<F::Challenge>>(),
        );

        vec![eq_eval, -eq_eval]
    }
}

#[derive(Allocative)]
pub struct ActivationBooleanitySumcheckProver<F: JoltField> {
    eq_r_cycle: GruenSplitEqPolynomial<F>,
    load: MultilinearPolynomial<F>,
    store: MultilinearPolynomial<F>,
    pub params: ActivationBooleanitySumcheckParams<F>,
}

impl<F: JoltField> ActivationBooleanitySumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamActivationBooleanitySumcheckProver::initialize")]
    #[expect(
        clippy::expect_used,
        reason = "trace rows are final Jolt instruction rows"
    )]
    pub fn initialize(params: ActivationBooleanitySumcheckParams<F>, trace: &[Cycle]) -> Self {
        let (load, store): (Vec<bool>, Vec<bool>) = trace
            .par_iter()
            .map(|cycle| {
                let flags = JoltTraceCycle::try_new(cycle)
                    .expect("activation columns require final Jolt instruction rows")
                    .circuit_flags();
                (flags[CircuitFlags::Load], flags[CircuitFlags::Store])
            })
            .unzip();
        let load = MultilinearPolynomial::from(load);
        let store = MultilinearPolynomial::from(store);

        let eq_r_cycle = GruenSplitEqPolynomial::new(&params.r_cycle.r, BindingOrder::LowToHigh);

        Self {
            eq_r_cycle,
            load,
            store,
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for ActivationBooleanitySumcheckProver<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    #[tracing::instrument(
        skip_all,
        name = "RamActivationBooleanitySumcheckProver::compute_message"
    )]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let eq = &self.eq_r_cycle;
        let load = &self.load;
        let store = &self.store;

        // Accumulate constant (c0) and quadratic (e) coefficients via generic
        // split-eq fold over the activation sum B = load + store.
        let [c0, e] = eq.par_fold_out_in_unreduced::<2>(&|g| {
            let b0 = load.get_bound_coeff(2 * g) + store.get_bound_coeff(2 * g);
            let b1 = load.get_bound_coeff(2 * g + 1) + store.get_bound_coeff(2 * g + 1);
            let delta = b1 - b0;
            [b0.square() - b0, delta.square()]
        });
        eq.gruen_poly_deg_3(c0, e, previous_claim)
    }

    #[tracing::instrument(
        skip_all,
        name = "RamActivationBooleanitySumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_cycle.bind(r_j);
        self.load.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.store.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            VirtualPolynomial::OpFlags(CircuitFlags::Load),
            SumcheckId::RamActivationBooleanity,
            opening_point.clone(),
            self.load.final_sumcheck_claim(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::OpFlags(CircuitFlags::Store),
            SumcheckId::RamActivationBooleanity,
            opening_point,
            self.store.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct ActivationBooleanitySumcheckVerifier<F: JoltField> {
    params: ActivationBooleanitySumcheckParams<F>,
}

impl<F: JoltField> ActivationBooleanitySumcheckVerifier<F> {
    pub fn new(opening_accumulator: &dyn OpeningAccumulator<F>) -> Self {
        Self {
            params: ActivationBooleanitySumcheckParams::new(opening_accumulator),
        }
    }
}

impl<F: JoltField, T: Transcript, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, T, A> for ActivationBooleanitySumcheckVerifier<F>
{
    fn input_claim(&self, accumulator: &A) -> F {
        let result = self.params.input_claim(accumulator);

        #[cfg(test)]
        {
            let reference_result =
                Self::input_output_claims().input_claim(&[F::one()], accumulator);
            assert_eq!(result, reference_result);
        }

        result
    }

    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let load_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Load),
                SumcheckId::RamActivationBooleanity,
            )
            .1;
        let store_claim = accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::OpFlags(CircuitFlags::Store),
                SumcheckId::RamActivationBooleanity,
            )
            .1;

        let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let eq = EqPolynomial::<F>::mle(
            sumcheck_challenges,
            &r_cycle
                .r
                .iter()
                .cloned()
                .rev()
                .collect::<Vec<F::Challenge>>(),
        );

        let activation = load_claim + store_claim;
        let result = (activation.square() - activation) * eq;

        #[cfg(test)]
        {
            let r = self.params.normalize_opening_point(sumcheck_challenges);
            let reference_result =
                Self::input_output_claims().expected_output_claim(&r, &[F::one()], accumulator);
            assert_eq!(result, reference_result);
        }

        result
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
        accumulator.append_virtual(
            VirtualPolynomial::OpFlags(CircuitFlags::Load),
            SumcheckId::RamActivationBooleanity,
            opening_point.clone(),
        );
        accumulator.append_virtual(
            VirtualPolynomial::OpFlags(CircuitFlags::Store),
            SumcheckId::RamActivationBooleanity,
            opening_point,
        );
    }
}

impl<F: JoltField> SumcheckFrontend<F> for ActivationBooleanitySumcheckVerifier<F> {
    fn input_output_claims() -> InputOutputClaims<F> {
        let load: ClaimExpr<F> = VirtualPolynomial::OpFlags(CircuitFlags::Load).into();
        let store: ClaimExpr<F> = VirtualPolynomial::OpFlags(CircuitFlags::Store).into();
        let activation = load + store;
        let activation_squared = activation.clone() * activation.clone();

        let eq_r_stage1 = VerifierEvaluablePolynomial::Eq(CachedPointRef {
            opening: PolynomialId::Virtual(VirtualPolynomial::LookupOutput),
            sumcheck: SumcheckId::SpartanOuter,
            part: ChallengePart::Cycle,
        });

        InputOutputClaims {
            claims: vec![Claim {
                // NOTE: In this case, the input claim is 0, so this is just the sumcheck to
                // take r_cycle from.
                input_sumcheck_id: SumcheckId::SpartanOuter,
                input_claim_expr: F::zero().into(),
                batching_poly: eq_r_stage1,
                expected_output_claim_expr: activation_squared - activation,
            }],
            output_sumcheck_id: SumcheckId::RamActivationBooleanity,
        }
    }
}
