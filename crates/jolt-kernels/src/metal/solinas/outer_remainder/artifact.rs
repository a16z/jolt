#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OuterBindingPlan {
    #[default]
    BOnlyV1,
    BOnlyPadded56V1,
}
