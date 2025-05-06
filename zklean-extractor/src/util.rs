use jolt_core::field::JoltField;

const TAB: &str = "  ";

pub fn indent(level: usize) -> String {
    std::iter::repeat(String::from(TAB))
        .take(level)
        .collect::<Vec<_>>()
        .concat()
}

/// A [`JoltField`] that can be used to write a ZKLean representation of a computation.
pub trait ZkLeanReprField: JoltField + Sized {
    fn register(name: char, size: usize) -> Vec<Self>;

    fn as_computation(&self) -> String;
}

/// A [`JoltField`] that can be evaluated over another [`JoltField`] (e.g., [`crate::mle_ast::MleAst`]).
#[cfg(test)]
pub trait Evaluatable: ZkLeanReprField + Sized {
    fn evaluate<F: JoltField>(&self, vars: &[F]) -> F;
}

#[cfg(test)]
pub fn test_evaluate_fn<F: JoltField, G: Evaluatable>(
    values_u64: &[u64],
    subtable_over_ref_field: impl jolt_core::jolt::subtable::LassoSubtable<F>,
    subtable_over_ast_field: impl jolt_core::jolt::subtable::LassoSubtable<G>,
) -> (F, F, G) {
    let reg_size = values_u64.len();
    let values = values_u64.into_iter().map(|&n| F::from_u64(n)).collect::<Vec<_>>();
    let expected_result = subtable_over_ref_field.evaluate_mle(&values);
    let register = G::register('x', reg_size);
    let to_evaluate = subtable_over_ast_field.evaluate_mle(&register);
    let actual_result = to_evaluate.evaluate(&values);
    (actual_result, expected_result, to_evaluate)
}
