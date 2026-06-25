use jolt_field::Field;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

pub fn gruen<F: Field>(point: &[F], binding_order: BindingOrder) -> GruenSplitEqPolynomial<F> {
    GruenSplitEqPolynomial::new(point, binding_order)
}

pub fn e_out_in_for_window<F: Field>(
    split_eq: &GruenSplitEqPolynomial<F>,
    window_size: usize,
) -> (&[F], &[F]) {
    split_eq.e_out_in_for_window(window_size)
}

pub fn e_active_for_window<F: Field>(
    split_eq: &GruenSplitEqPolynomial<F>,
    window_size: usize,
) -> Vec<F> {
    split_eq.e_active_for_window(window_size)
}
