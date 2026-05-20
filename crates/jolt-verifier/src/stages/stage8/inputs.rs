use jolt_field::Field;

use crate::stages::{stage6::Stage6ClearOutput, stage7::Stage7Output};

#[derive(Clone, Copy)]
pub struct Deps<'a, F: Field> {
    pub stage6: &'a Stage6ClearOutput<F>,
    pub stage7: &'a Stage7Output<F>,
}

pub fn deps<'a, F: Field>(
    stage6: &'a Stage6ClearOutput<F>,
    stage7: &'a Stage7Output<F>,
) -> Deps<'a, F> {
    Deps { stage6, stage7 }
}
