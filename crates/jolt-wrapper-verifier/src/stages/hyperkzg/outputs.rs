#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HyperKzgOutput;

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperKzgZkOutput<C> {
    pub hiding_evaluation_commitment: C,
}
