use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_interleaved_poly::DenseInterleavedPolynomial;
use crate::subprotocols::grand_product::{BatchedDenseGrandProduct, BatchedGrandProductProver, BatchedGrandProductLayer};
use crate::utils::math::Math;
use crate::utils::transcript::Transcript;
use rayon::prelude::*;

impl<F, PCS, ProofTranscript> BatchedGrandProductProver<F, PCS, ProofTranscript>
    for BatchedDenseGrandProduct<F>
where
    F: JoltField,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
    ProofTranscript: Transcript,
{
    // (leaf values, batch size)
    type Leaves = (Vec<F>, usize);
    type Config = ();

    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProduct::construct")]
    fn construct(leaves: Self::Leaves) -> Self {
        let (leaves, batch_size) = leaves;
        assert_eq!(leaves.len() % batch_size, 0);
        assert!((leaves.len() / batch_size).is_power_of_two());

        let num_layers = (leaves.len() / batch_size).log_2();
        let mut layers: Vec<DenseInterleavedPolynomial<F>> = Vec::with_capacity(num_layers);
        layers.push(DenseInterleavedPolynomial::new(leaves));

        for i in 0..num_layers - 1 {
            let previous_layer = &layers[i];
            let new_layer = previous_layer.layer_output();
            layers.push(new_layer);
        }

        Self { layers }
    }
    #[tracing::instrument(skip_all, name = "BatchedDenseGrandProduct::construct_with_config")]
    fn construct_with_config(leaves: Self::Leaves, _config: Self::Config) -> Self {
        <Self as BatchedGrandProductProver<F, PCS, ProofTranscript>>::construct(leaves)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn claimed_outputs(&self) -> Vec<F> {
        let last_layer = &self.layers[self.layers.len() - 1];
        last_layer
            .par_chunks(2)
            .map(|chunk| chunk[0] * chunk[1])
            .collect()
    }

    fn layers(
        &'_ mut self,
    ) -> impl Iterator<Item = &'_ mut dyn BatchedGrandProductLayer<F, ProofTranscript>> {
        self.layers
            .iter_mut()
            .map(|layer| layer as &mut dyn BatchedGrandProductLayer<F, ProofTranscript>)
            .rev()
    }
}
