//! The Spartan shift (stage 3) kernel: a naive member over the cycle domain.
//!
//! The summand is
//! `eq⁺¹(τ_low, j) · (upc + γ·pc + γ²·is_virtual + γ³·is_first_in_sequence)(j)
//!  + γ⁴ · eq⁺¹(r_product, j) · (1 − is_noop(j))`
//! — the shift is carried entirely by the two `eq+1` factors (the value
//! tables are unshifted, defined at every cycle including `T − 1`); each
//! `eq+1` table is one multilinear whose MLE is the verifier's closed-form
//! `EqPlusOnePolynomial::evaluate` (no wraparound — consistent with the
//! `NextIsNoop = 1` missing-successor convention upstream).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::spartan::{
    is_first_in_sequence_shift, is_noop_shift, is_virtual_shift, pc_shift, unexpanded_pc_shift,
};
use jolt_claims::protocols::jolt::relations::spartan::SpartanShiftChallenges;
use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanShiftPublic, TraceDimensions};
use jolt_field::Field;
use jolt_poly::{BindingOrder, EqPlusOnePolynomial, Polynomial};
use jolt_verifier::stages::stage3::outputs::SpartanShift;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::dense_view;
use crate::spartan_shift::SpartanShiftProver;
use crate::{KernelError, NaiveSumcheckProver, ProofSession, ProveSumcheck, ReferenceBackend};

impl<F: Field> SpartanShiftProver<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        trace_dimensions: TraceDimensions,
        product_uniskip_tau_low: &[F],
        product_remainder_point: &[F],
        challenges: &SpartanShiftChallenges<F>,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
    ) -> Result<Box<dyn ProveSumcheck<F, Relation = SpartanShift<F>>>, KernelError<F>> {
        let ids = [
            unexpanded_pc_shift(),
            pc_shift(),
            is_virtual_shift(),
            is_first_in_sequence_shift(),
            is_noop_shift(),
        ];
        let opening_tables = ids
            .into_iter()
            .map(|id| Ok((id, Polynomial::new(dense_view(witness, id)?))))
            .collect::<Result<BTreeMap<_, _>, KernelError<F>>>()?;
        let derived_tables = BTreeMap::from([
            (
                JoltDerivedId::from(SpartanShiftPublic::EqPlusOneOuter),
                Polynomial::new(EqPlusOnePolynomial::evals(product_uniskip_tau_low, None).1),
            ),
            (
                JoltDerivedId::from(SpartanShiftPublic::EqPlusOneProduct),
                Polynomial::new(EqPlusOnePolynomial::evals(product_remainder_point, None).1),
            ),
        ]);

        let relation = SpartanShift::new(
            trace_dimensions,
            product_uniskip_tau_low.to_vec(),
            product_remainder_point.to_vec(),
        );
        Ok(Box::new(NaiveSumcheckProver::new(
            relation,
            challenges,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )?))
    }
}
