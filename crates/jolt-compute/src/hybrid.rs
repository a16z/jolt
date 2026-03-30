//! Hybrid compute backend that wraps a primary (GPU) and fallback (CPU) backend.
//!
//! Buffers start on the primary backend and automatically downgrade to the
//! fallback when they shrink below a configurable threshold during
//! `interpolate_pairs_inplace`. This captures the key observation that GPU
//! kernel launch overhead dominates for small buffers, while GPU parallelism
//! dominates for large ones.
//!
//! The transition is **one-way**: once a buffer migrates to the fallback, it
//! stays there for the rest of the sumcheck. All buffers in a stage bind in
//! lockstep (same scalar, same round), so they transition together.

use jolt_compiler::CompositionFormula;
use jolt_field::Field;

use crate::{BackendWitness, BindingOrder, ComputeBackend, EqInput, RoundCoeffs, Scalar};

/// Hybrid backend that delegates to a primary or fallback backend based on
/// buffer size.
///
/// `P` is the primary backend (typically GPU — Metal, CUDA, WebGPU).
/// `F` is the fallback backend (typically CPU).
///
/// All uploads go to the primary. When `interpolate_pairs_inplace` shrinks a
/// buffer below `threshold`, its data is downloaded from the primary and
/// re-uploaded to the fallback. Subsequent operations on that buffer use
/// the fallback backend.
pub struct HybridBackend<P: ComputeBackend, Fb: ComputeBackend> {
    primary: P,
    fallback: Fb,
    /// Buffers with active element count at or below this value migrate to
    /// the fallback on the next `interpolate_pairs_inplace`.
    threshold: usize,
}

impl<P: ComputeBackend, Fb: ComputeBackend> HybridBackend<P, Fb> {
    pub fn new(primary: P, fallback: Fb, threshold: usize) -> Self {
        Self {
            primary,
            fallback,
            threshold,
        }
    }

    pub fn primary(&self) -> &P {
        &self.primary
    }

    pub fn fallback(&self) -> &Fb {
        &self.fallback
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }
}

/// Buffer that lives on either the primary or fallback backend.
///
/// Starts as `Primary` after `upload`. Transitions to `Fallback` when the
/// buffer shrinks below the hybrid backend's threshold during binding.
/// This transition is one-way — a fallback buffer never moves back to the
/// primary.
pub enum HybridBuffer<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> {
    Primary(P::Buffer<T>),
    Fallback(Fb::Buffer<T>),
}

// SAFETY: `HybridBuffer` delegates to `P::Buffer` or `Fb::Buffer`, both of
// which are `Send + Sync` as required by `ComputeBackend`. The enum adds no
// interior mutability or thread-unsafe state.
unsafe impl<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> Send for HybridBuffer<T, P, Fb>
where
    P::Buffer<T>: Send,
    Fb::Buffer<T>: Send,
{
}
// SAFETY: See `Send` impl above.
unsafe impl<T: Scalar, P: ComputeBackend, Fb: ComputeBackend> Sync for HybridBuffer<T, P, Fb>
where
    P::Buffer<T>: Sync,
    Fb::Buffer<T>: Sync,
{
}

/// Compiled kernel holding both backend's compiled forms.
///
/// Always compiles for both backends at kernel compile time so that we can
/// dispatch to either without recompilation when buffers transition.
pub struct HybridKernel<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> {
    primary: P::CompiledKernel<Fld>,
    fallback: Fb::CompiledKernel<Fld>,
}

// SAFETY: `HybridKernel` wraps `P::CompiledKernel` and `Fb::CompiledKernel`,
// both `Send + Sync` as required by `ComputeBackend`. No additional state.
unsafe impl<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> Send for HybridKernel<Fld, P, Fb>
where
    P::CompiledKernel<Fld>: Send,
    Fb::CompiledKernel<Fld>: Send,
{
}
// SAFETY: See `Send` impl above.
unsafe impl<Fld: Field, P: ComputeBackend, Fb: ComputeBackend> Sync for HybridKernel<Fld, P, Fb>
where
    P::CompiledKernel<Fld>: Sync,
    Fb::CompiledKernel<Fld>: Sync,
{
}

/// Sparse buffer on either the primary or fallback backend.
pub enum HybridSparseBuffer<F: Field, P: ComputeBackend, Fb: ComputeBackend> {
    Primary(P::SparseBuffer<F>),
    Fallback(Fb::SparseBuffer<F>),
}

// SAFETY: Delegates to inner types which are Send + Sync per ComputeBackend.
unsafe impl<F: Field, P: ComputeBackend, Fb: ComputeBackend> Send for HybridSparseBuffer<F, P, Fb>
where
    P::SparseBuffer<F>: Send,
    Fb::SparseBuffer<F>: Send,
{
}
// SAFETY: Delegates to inner type which is Sync per ComputeBackend.
unsafe impl<F: Field, P: ComputeBackend, Fb: ComputeBackend> Sync for HybridSparseBuffer<F, P, Fb>
where
    P::SparseBuffer<F>: Sync,
    Fb::SparseBuffer<F>: Sync,
{
}

/// Sumcheck witness on either the primary or fallback backend.
pub enum HybridSumcheckWitness<F: Field, P: ComputeBackend, Fb: ComputeBackend> {
    Primary(P::SumcheckWitness<F>),
    Fallback(Fb::SumcheckWitness<F>),
}

// SAFETY: Delegates to inner types which are Send per BackendWitness.
unsafe impl<F: Field, P: ComputeBackend, Fb: ComputeBackend> Send
    for HybridSumcheckWitness<F, P, Fb>
where
    P::SumcheckWitness<F>: Send,
    Fb::SumcheckWitness<F>: Send,
{
}
// SAFETY: Delegates to inner types which are Sync per BackendWitness.
unsafe impl<F: Field, P: ComputeBackend, Fb: ComputeBackend> Sync
    for HybridSumcheckWitness<F, P, Fb>
where
    P::SumcheckWitness<F>: Sync,
    Fb::SumcheckWitness<F>: Sync,
{
}

impl<F: Field, P: ComputeBackend, Fb: ComputeBackend> BackendWitness<F>
    for HybridSumcheckWitness<F, P, Fb>
{
    fn round_polynomial(&self) -> RoundCoeffs<F> {
        match self {
            Self::Primary(w) => w.round_polynomial(),
            Self::Fallback(w) => w.round_polynomial(),
        }
    }

    fn bind(&mut self, challenge: F) {
        match self {
            Self::Primary(w) => w.bind(challenge),
            Self::Fallback(w) => w.bind(challenge),
        }
    }

    fn set_claim(&mut self, claim: F) {
        match self {
            Self::Primary(w) => w.set_claim(claim),
            Self::Fallback(w) => w.set_claim(claim),
        }
    }

    fn first_round_polynomial(&self) -> Option<RoundCoeffs<F>> {
        match self {
            Self::Primary(w) => w.first_round_polynomial(),
            Self::Fallback(w) => w.first_round_polynomial(),
        }
    }

    fn produced_evaluations(&self) -> Vec<(usize, F)> {
        match self {
            Self::Primary(w) => w.produced_evaluations(),
            Self::Fallback(w) => w.produced_evaluations(),
        }
    }
}

impl<P: ComputeBackend, Fb: ComputeBackend> HybridBackend<P, Fb> {
    fn should_migrate(&self, len: usize) -> bool {
        len <= self.threshold
    }
}

fn convert_eq<'a, Fld: Field, P: ComputeBackend, Fb: ComputeBackend, T: ComputeBackend>(
    eq: EqInput<'a, HybridBackend<P, Fb>, Fld>,
    extract: impl Fn(&'a HybridBuffer<Fld, P, Fb>) -> &'a T::Buffer<Fld>,
) -> EqInput<'a, T, Fld> {
    match eq {
        EqInput::Unit => EqInput::Unit,
        EqInput::Weighted(w) => EqInput::Weighted(extract(w)),
        EqInput::Tensor { outer, inner } => EqInput::Tensor {
            outer: extract(outer),
            inner: extract(inner),
        },
    }
}

impl<P: ComputeBackend, Fb: ComputeBackend> ComputeBackend for HybridBackend<P, Fb> {
    type Buffer<T: Scalar> = HybridBuffer<T, P, Fb>;
    type CompiledKernel<Fld: Field> = HybridKernel<Fld, P, Fb>;
    type SumcheckWitness<Fld: Field> = HybridSumcheckWitness<Fld, P, Fb>;
    type SparseBuffer<Fld: Field> = HybridSparseBuffer<Fld, P, Fb>;

    fn compile_kernel_with_challenges<Fld: Field>(
        &self,
        formula: &CompositionFormula,
        challenges: &[Fld],
    ) -> Self::CompiledKernel<Fld> {
        HybridKernel {
            primary: self
                .primary
                .compile_kernel_with_challenges(formula, challenges),
            fallback: self
                .fallback
                .compile_kernel_with_challenges(formula, challenges),
        }
    }

    fn upload<T: Scalar>(&self, data: &[T]) -> Self::Buffer<T> {
        // Small data goes directly to fallback — no point in a GPU round-trip.
        if data.len() <= self.threshold {
            HybridBuffer::Fallback(self.fallback.upload(data))
        } else {
            HybridBuffer::Primary(self.primary.upload(data))
        }
    }

    fn download<T: Scalar>(&self, buf: &Self::Buffer<T>) -> Vec<T> {
        match buf {
            HybridBuffer::Primary(b) => self.primary.download(b),
            HybridBuffer::Fallback(b) => self.fallback.download(b),
        }
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Self::Buffer<T> {
        if len <= self.threshold {
            HybridBuffer::Fallback(self.fallback.alloc(len))
        } else {
            HybridBuffer::Primary(self.primary.alloc(len))
        }
    }

    fn len<T: Scalar>(&self, buf: &Self::Buffer<T>) -> usize {
        match buf {
            HybridBuffer::Primary(b) => self.primary.len(b),
            HybridBuffer::Fallback(b) => self.fallback.len(b),
        }
    }

    fn interpolate_pairs<T, Fld>(&self, buf: Self::Buffer<T>, scalar: Fld) -> Self::Buffer<Fld>
    where
        T: Scalar,
        Fld: Field + From<T>,
    {
        match buf {
            HybridBuffer::Primary(b) => {
                let result_len = self.primary.len(&b) / 2;
                if result_len <= self.threshold {
                    // Migrate: download, bind on fallback.
                    let data = self.primary.download(&b);
                    let fb_buf = self.fallback.upload(&data);
                    HybridBuffer::Fallback(self.fallback.interpolate_pairs(fb_buf, scalar))
                } else {
                    HybridBuffer::Primary(self.primary.interpolate_pairs(b, scalar))
                }
            }
            HybridBuffer::Fallback(b) => {
                HybridBuffer::Fallback(self.fallback.interpolate_pairs(b, scalar))
            }
        }
    }

    fn interpolate_pairs_inplace<Fld: Field>(
        &self,
        buf: &mut Self::Buffer<Fld>,
        scalar: Fld,
        order: BindingOrder,
    ) {
        // Determine if we need to migrate after this bind.
        let current_len = self.len(buf);
        let result_len = current_len / 2;

        match buf {
            HybridBuffer::Primary(b) => {
                if self.should_migrate(result_len) {
                    // Bind on primary first (it holds the data), then migrate.
                    self.primary.interpolate_pairs_inplace(b, scalar, order);
                    let data = self.primary.download(b);
                    *buf = HybridBuffer::Fallback(self.fallback.upload(&data));
                } else {
                    self.primary.interpolate_pairs_inplace(b, scalar, order);
                }
            }
            HybridBuffer::Fallback(b) => {
                self.fallback.interpolate_pairs_inplace(b, scalar, order);
            }
        }
    }

    fn interpolate_pairs_batch<Fld: Field>(
        &self,
        bufs: Vec<Self::Buffer<Fld>>,
        scalar: Fld,
    ) -> Vec<Self::Buffer<Fld>> {
        // Partition into primary and fallback, batch each, reassemble.
        // Track original indices to preserve order.
        let mut primary_bufs = Vec::new();
        let mut fallback_bufs = Vec::new();
        let mut indices: Vec<(bool, usize)> = Vec::with_capacity(bufs.len()); // (is_primary, idx_in_partition)

        for buf in bufs {
            match buf {
                HybridBuffer::Primary(b) => {
                    indices.push((true, primary_bufs.len()));
                    primary_bufs.push(b);
                }
                HybridBuffer::Fallback(b) => {
                    indices.push((false, fallback_bufs.len()));
                    fallback_bufs.push(b);
                }
            }
        }

        let mut primary_results = self.primary.interpolate_pairs_batch(primary_bufs, scalar);
        let mut fallback_results = self.fallback.interpolate_pairs_batch(fallback_bufs, scalar);

        // Reassemble in original order, migrating small primary results.
        let mut results = Vec::with_capacity(indices.len());
        for &(is_primary, idx) in &indices {
            if is_primary {
                // We need to take from primary_results — use a sentinel swap.
                let b = std::mem::replace(&mut primary_results[idx], self.primary.alloc(0));
                let len = self.primary.len(&b);
                if len <= self.threshold {
                    let data = self.primary.download(&b);
                    results.push(HybridBuffer::Fallback(self.fallback.upload(&data)));
                } else {
                    results.push(HybridBuffer::Primary(b));
                }
            } else {
                let b = std::mem::replace(&mut fallback_results[idx], self.fallback.alloc(0));
                results.push(HybridBuffer::Fallback(b));
            }
        }
        results
    }

    fn interpolate_pairs_batch_inplace<Fld: Field>(
        &self,
        bufs: &mut [Self::Buffer<Fld>],
        scalar: Fld,
        order: BindingOrder,
    ) {
        for buf in bufs.iter_mut() {
            self.interpolate_pairs_inplace(buf, scalar, order);
        }
    }

    fn pairwise_reduce<Fld: Field>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        eq: EqInput<'_, Self, Fld>,
        kernel: &Self::CompiledKernel<Fld>,
        num_evals: usize,
        order: BindingOrder,
    ) -> Vec<Fld> {
        // All buffers in a stage bind in lockstep, so they should all be
        // on the same backend. Check the first input to determine which.
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in pairwise_reduce")
                            }
                        })
                        .collect();
                    let p_eq = convert_eq::<Fld, P, Fb, P>(eq, |buf| match buf {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("eq buffer backend mismatch in pairwise_reduce")
                        }
                    });
                    self.primary
                        .pairwise_reduce(&p_inputs, p_eq, &kernel.primary, num_evals, order)
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in pairwise_reduce")
                            }
                        })
                        .collect();
                    let f_eq = convert_eq::<Fld, P, Fb, Fb>(eq, |buf| match buf {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("eq buffer backend mismatch in pairwise_reduce")
                        }
                    });
                    self.fallback.pairwise_reduce(
                        &f_inputs,
                        f_eq,
                        &kernel.fallback,
                        num_evals,
                        order,
                    )
                }
            }
        } else {
            vec![Fld::zero(); num_evals]
        }
    }

    fn pairwise_reduce_fixed<Fld: Field, const D: usize>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        eq: EqInput<'_, Self, Fld>,
        kernel: &Self::CompiledKernel<Fld>,
        order: BindingOrder,
    ) -> [Fld; D] {
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_fixed")
                            }
                        })
                        .collect();
                    let p_eq = convert_eq::<Fld, P, Fb, P>(eq, |buf| match buf {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("eq buffer backend mismatch in pairwise_reduce_fixed")
                        }
                    });
                    self.primary.pairwise_reduce_fixed::<Fld, D>(
                        &p_inputs,
                        p_eq,
                        &kernel.primary,
                        order,
                    )
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_fixed")
                            }
                        })
                        .collect();
                    let f_eq = convert_eq::<Fld, P, Fb, Fb>(eq, |buf| match buf {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("eq buffer backend mismatch in pairwise_reduce_fixed")
                        }
                    });
                    self.fallback.pairwise_reduce_fixed::<Fld, D>(
                        &f_inputs,
                        f_eq,
                        &kernel.fallback,
                        order,
                    )
                }
            }
        } else {
            [Fld::zero(); D]
        }
    }

    fn pairwise_reduce_multi<Fld: Field>(
        &self,
        inputs: &[&Self::Buffer<Fld>],
        eq: EqInput<'_, Self, Fld>,
        kernels: &[(&Self::CompiledKernel<Fld>, usize)],
        order: BindingOrder,
    ) -> Vec<Vec<Fld>> {
        if let Some(first) = inputs.first() {
            match first {
                HybridBuffer::Primary(_) => {
                    let p_inputs: Vec<&P::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Primary(inner) => inner,
                            HybridBuffer::Fallback(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_multi")
                            }
                        })
                        .collect();
                    let p_eq = convert_eq::<Fld, P, Fb, P>(eq, |buf| match buf {
                        HybridBuffer::Primary(w) => w,
                        HybridBuffer::Fallback(_) => {
                            panic!("eq buffer backend mismatch in pairwise_reduce_multi")
                        }
                    });
                    let p_kernels: Vec<(&P::CompiledKernel<Fld>, usize)> =
                        kernels.iter().map(|(k, n)| (&k.primary, *n)).collect();
                    self.primary
                        .pairwise_reduce_multi(&p_inputs, p_eq, &p_kernels, order)
                }
                HybridBuffer::Fallback(_) => {
                    let f_inputs: Vec<&Fb::Buffer<Fld>> = inputs
                        .iter()
                        .map(|b| match b {
                            HybridBuffer::Fallback(inner) => inner,
                            HybridBuffer::Primary(_) => {
                                panic!("mixed buffer backends in pairwise_reduce_multi")
                            }
                        })
                        .collect();
                    let f_eq = convert_eq::<Fld, P, Fb, Fb>(eq, |buf| match buf {
                        HybridBuffer::Fallback(w) => w,
                        HybridBuffer::Primary(_) => {
                            panic!("eq buffer backend mismatch in pairwise_reduce_multi")
                        }
                    });
                    let f_kernels: Vec<(&Fb::CompiledKernel<Fld>, usize)> =
                        kernels.iter().map(|(k, n)| (&k.fallback, *n)).collect();
                    self.fallback
                        .pairwise_reduce_multi(&f_inputs, f_eq, &f_kernels, order)
                }
            }
        } else {
            kernels.iter().map(|(_, n)| vec![Fld::zero(); *n]).collect()
        }
    }

    fn product_table<Fld: Field>(&self, point: &[Fld]) -> Self::Buffer<Fld> {
        let len = 1usize << point.len();
        if len <= self.threshold {
            HybridBuffer::Fallback(self.fallback.product_table(point))
        } else {
            HybridBuffer::Primary(self.primary.product_table(point))
        }
    }

    fn sum<Fld: Field>(&self, buf: &Self::Buffer<Fld>) -> Fld {
        match buf {
            HybridBuffer::Primary(b) => self.primary.sum(b),
            HybridBuffer::Fallback(b) => self.fallback.sum(b),
        }
    }

    fn dot_product<Fld: Field>(&self, a: &Self::Buffer<Fld>, b: &Self::Buffer<Fld>) -> Fld {
        match (a, b) {
            (HybridBuffer::Primary(a), HybridBuffer::Primary(b)) => self.primary.dot_product(a, b),
            (HybridBuffer::Fallback(a), HybridBuffer::Fallback(b)) => {
                self.fallback.dot_product(a, b)
            }
            _ => panic!("mixed buffer backends in dot_product"),
        }
    }

    fn scale<Fld: Field>(&self, buf: &mut Self::Buffer<Fld>, scalar: Fld) {
        match buf {
            HybridBuffer::Primary(b) => self.primary.scale(b, scalar),
            HybridBuffer::Fallback(b) => self.fallback.scale(b, scalar),
        }
    }

    fn add<Fld: Field>(&self, a: &Self::Buffer<Fld>, b: &Self::Buffer<Fld>) -> Self::Buffer<Fld> {
        match (a, b) {
            (HybridBuffer::Primary(a), HybridBuffer::Primary(b)) => {
                HybridBuffer::Primary(self.primary.add(a, b))
            }
            (HybridBuffer::Fallback(a), HybridBuffer::Fallback(b)) => {
                HybridBuffer::Fallback(self.fallback.add(a, b))
            }
            _ => panic!("mixed buffer backends in add"),
        }
    }

    fn sub<Fld: Field>(&self, a: &Self::Buffer<Fld>, b: &Self::Buffer<Fld>) -> Self::Buffer<Fld> {
        match (a, b) {
            (HybridBuffer::Primary(a), HybridBuffer::Primary(b)) => {
                HybridBuffer::Primary(self.primary.sub(a, b))
            }
            (HybridBuffer::Fallback(a), HybridBuffer::Fallback(b)) => {
                HybridBuffer::Fallback(self.fallback.sub(a, b))
            }
            _ => panic!("mixed buffer backends in sub"),
        }
    }

    fn accumulate<Fld: Field>(
        &self,
        buf: &mut Self::Buffer<Fld>,
        scalar: Fld,
        other: &Self::Buffer<Fld>,
    ) {
        match (buf, other) {
            (HybridBuffer::Primary(b), HybridBuffer::Primary(o)) => {
                self.primary.accumulate(b, scalar, o);
            }
            (HybridBuffer::Fallback(b), HybridBuffer::Fallback(o)) => {
                self.fallback.accumulate(b, scalar, o);
            }
            _ => panic!("mixed buffer backends in accumulate"),
        }
    }

    fn scale_batch<Fld: Field>(&self, bufs: &mut [Self::Buffer<Fld>], scalar: Fld) {
        for buf in bufs.iter_mut() {
            self.scale(buf, scalar);
        }
    }

    fn fused_interpolate_reduce<Fld: Field>(
        &self,
        inputs: &mut [Self::Buffer<Fld>],
        weights: &mut Self::Buffer<Fld>,
        interpolation_scalar: Fld,
        kernel: &Self::CompiledKernel<Fld>,
        num_evals: usize,
    ) -> Vec<Fld> {
        if inputs.is_empty() {
            return vec![Fld::zero(); num_evals];
        }

        let is_primary = matches!(inputs[0], HybridBuffer::Primary(_));

        if is_primary {
            // Temporarily move inner buffers out to form a contiguous slice.
            // alloc(0) creates a zero-sized placeholder — no real allocation.
            let mut p_bufs: Vec<P::Buffer<Fld>> = inputs
                .iter_mut()
                .map(|b| match b {
                    HybridBuffer::Primary(inner) => std::mem::replace(inner, self.primary.alloc(0)),
                    HybridBuffer::Fallback(_) => {
                        panic!("mixed buffer backends in fused_interpolate_reduce")
                    }
                })
                .collect();
            let mut p_weights = match weights {
                HybridBuffer::Primary(w) => std::mem::replace(w, self.primary.alloc(0)),
                HybridBuffer::Fallback(_) => {
                    panic!("weight buffer backend mismatch in fused_interpolate_reduce")
                }
            };

            let result = self.primary.fused_interpolate_reduce(
                &mut p_bufs,
                &mut p_weights,
                interpolation_scalar,
                &kernel.primary,
                num_evals,
            );

            // Check migration before putting buffers back
            let new_len = self.primary.len(&p_bufs[0]);
            if self.should_migrate(new_len) {
                for (slot, p_buf) in inputs.iter_mut().zip(p_bufs) {
                    let data = self.primary.download(&p_buf);
                    *slot = HybridBuffer::Fallback(self.fallback.upload(&data));
                }
                let w_data = self.primary.download(&p_weights);
                *weights = HybridBuffer::Fallback(self.fallback.upload(&w_data));
            } else {
                for (slot, p_buf) in inputs.iter_mut().zip(p_bufs) {
                    *slot = HybridBuffer::Primary(p_buf);
                }
                *weights = HybridBuffer::Primary(p_weights);
            }

            result
        } else {
            let mut f_bufs: Vec<Fb::Buffer<Fld>> = inputs
                .iter_mut()
                .map(|b| match b {
                    HybridBuffer::Fallback(inner) => {
                        std::mem::replace(inner, self.fallback.alloc(0))
                    }
                    HybridBuffer::Primary(_) => {
                        panic!("mixed buffer backends in fused_interpolate_reduce")
                    }
                })
                .collect();
            let mut f_weights = match weights {
                HybridBuffer::Fallback(w) => std::mem::replace(w, self.fallback.alloc(0)),
                HybridBuffer::Primary(_) => {
                    panic!("weight buffer backend mismatch in fused_interpolate_reduce")
                }
            };

            let result = self.fallback.fused_interpolate_reduce(
                &mut f_bufs,
                &mut f_weights,
                interpolation_scalar,
                &kernel.fallback,
                num_evals,
            );

            for (slot, f_buf) in inputs.iter_mut().zip(f_bufs) {
                *slot = HybridBuffer::Fallback(f_buf);
            }
            *weights = HybridBuffer::Fallback(f_weights);

            result
        }
    }

    fn make_witness<Fld: Field>(
        &self,
        kernel: &Self::CompiledKernel<Fld>,
        inputs: Vec<Self::Buffer<Fld>>,
        challenges: &[Fld],
    ) -> Self::SumcheckWitness<Fld> {
        // Determine backend from first input buffer. All inputs must be on
        // the same backend (enforced by lockstep binding).
        let is_primary = inputs
            .first()
            .is_some_and(|b| matches!(b, HybridBuffer::Primary(_)));

        if is_primary {
            let p_inputs: Vec<P::Buffer<Fld>> = inputs
                .into_iter()
                .map(|b| match b {
                    HybridBuffer::Primary(inner) => inner,
                    HybridBuffer::Fallback(_) => {
                        panic!("mixed buffer backends in make_witness")
                    }
                })
                .collect();
            HybridSumcheckWitness::Primary(self.primary.make_witness(
                &kernel.primary,
                p_inputs,
                challenges,
            ))
        } else {
            let f_inputs: Vec<Fb::Buffer<Fld>> = inputs
                .into_iter()
                .map(|b| match b {
                    HybridBuffer::Fallback(inner) => inner,
                    HybridBuffer::Primary(_) => {
                        panic!("mixed buffer backends in make_witness")
                    }
                })
                .collect();
            HybridSumcheckWitness::Fallback(self.fallback.make_witness(
                &kernel.fallback,
                f_inputs,
                challenges,
            ))
        }
    }

    fn upload_sparse<Fld: Field>(&self, entries: &[(usize, Vec<Fld>)]) -> Self::SparseBuffer<Fld> {
        // Sparse data is typically small; default to fallback.
        HybridSparseBuffer::Fallback(self.fallback.upload_sparse(entries))
    }

    fn sparse_reduce<Fld: Field>(
        &self,
        entries: &Self::SparseBuffer<Fld>,
        eq: &Self::Buffer<Fld>,
        kernel: &Self::CompiledKernel<Fld>,
        challenges: &[Fld],
        num_evals: usize,
    ) -> Vec<Fld> {
        match (entries, eq) {
            (HybridSparseBuffer::Primary(e), HybridBuffer::Primary(eq_buf)) => self
                .primary
                .sparse_reduce(e, eq_buf, &kernel.primary, challenges, num_evals),
            (HybridSparseBuffer::Fallback(e), HybridBuffer::Fallback(eq_buf)) => self
                .fallback
                .sparse_reduce(e, eq_buf, &kernel.fallback, challenges, num_evals),
            _ => panic!("mixed buffer backends in sparse_reduce"),
        }
    }

    fn sparse_bind<Fld: Field>(
        &self,
        entries: &mut Self::SparseBuffer<Fld>,
        challenge: Fld,
        order: BindingOrder,
    ) {
        match entries {
            HybridSparseBuffer::Primary(e) => self.primary.sparse_bind(e, challenge, order),
            HybridSparseBuffer::Fallback(e) => self.fallback.sparse_bind(e, challenge, order),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use jolt_field::{Field, Fr};
    use num_traits::Zero;

    /// Minimal "mock" backend for testing without pulling in the full CpuBackend.
    /// Uses Vec<T> buffers and trivial implementations.
    ///
    /// We use this to avoid a dev-dependency on jolt-cpu in jolt-compute's own
    /// tests while still exercising the HybridBackend logic.
    struct MockBackend {
        name: &'static str,
    }

    impl MockBackend {
        fn new(name: &'static str) -> Self {
            Self { name }
        }
    }

    struct MockKernel<F: Field> {
        _marker: std::marker::PhantomData<F>,
    }

    struct MockSumcheckWitness<F: Field> {
        _marker: std::marker::PhantomData<F>,
    }

    impl<F: Field> BackendWitness<F> for MockSumcheckWitness<F> {
        fn round_polynomial(&self) -> RoundCoeffs<F> {
            panic!("mock witness")
        }
        fn bind(&mut self, _challenge: F) {
            panic!("mock witness")
        }
    }

    impl ComputeBackend for MockBackend {
        type Buffer<T: Scalar> = Vec<T>;
        type CompiledKernel<F: Field> = MockKernel<F>;
        type SumcheckWitness<F: Field> = MockSumcheckWitness<F>;
        type SparseBuffer<F: Field> = Vec<(usize, Vec<F>)>;

        fn compile_kernel_with_challenges<F: Field>(
            &self,
            _formula: &CompositionFormula,
            _challenges: &[F],
        ) -> MockKernel<F> {
            MockKernel {
                _marker: std::marker::PhantomData,
            }
        }

        fn upload<T: Scalar>(&self, data: &[T]) -> Vec<T> {
            data.to_vec()
        }

        fn download<T: Scalar>(&self, buf: &Vec<T>) -> Vec<T> {
            buf.clone()
        }

        fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
            // SAFETY: All `Scalar` types (integers, bool, field elements) have
            // all-zeros as a valid bit pattern.
            vec![unsafe { std::mem::zeroed() }; len]
        }

        fn len<T: Scalar>(&self, buf: &Vec<T>) -> usize {
            buf.len()
        }

        fn interpolate_pairs<T, Fld>(&self, buf: Vec<T>, scalar: Fld) -> Vec<Fld>
        where
            T: Scalar,
            Fld: Field + From<T>,
        {
            let n = buf.len() / 2;
            let mut result = Vec::with_capacity(n);
            for i in 0..n {
                let lo = Fld::from(buf[2 * i]);
                let hi = Fld::from(buf[2 * i + 1]);
                result.push(lo + scalar * (hi - lo));
            }
            result
        }

        fn interpolate_pairs_inplace<Fld: Field>(
            &self,
            buf: &mut Vec<Fld>,
            scalar: Fld,
            order: BindingOrder,
        ) {
            let n = buf.len();
            let half = n / 2;
            match order {
                BindingOrder::LowToHigh => {
                    for i in 0..half {
                        let lo = buf[2 * i];
                        let hi = buf[2 * i + 1];
                        buf[i] = lo + scalar * (hi - lo);
                    }
                }
                BindingOrder::HighToLow => {
                    for i in 0..half {
                        let lo = buf[i];
                        let hi = buf[i + half];
                        buf[i] = lo + scalar * (hi - lo);
                    }
                }
            }
            buf.truncate(half);
        }

        fn pairwise_reduce<Fld: Field>(
            &self,
            _inputs: &[&Vec<Fld>],
            _eq: EqInput<'_, Self, Fld>,
            _kernel: &MockKernel<Fld>,
            num_evals: usize,
            _order: BindingOrder,
        ) -> Vec<Fld> {
            // Return identifiable values based on backend name for testing dispatch.
            if self.name == "primary" {
                vec![Fld::from_u64(1); num_evals]
            } else {
                vec![Fld::from_u64(2); num_evals]
            }
        }

        fn product_table<Fld: Field>(&self, point: &[Fld]) -> Vec<Fld> {
            let len = 1usize << point.len();
            vec![Fld::one(); len]
        }

        fn sum<Fld: Field>(&self, buf: &Vec<Fld>) -> Fld {
            buf.iter().fold(Fld::zero(), |acc, x| acc + *x)
        }

        fn dot_product<Fld: Field>(&self, a: &Vec<Fld>, b: &Vec<Fld>) -> Fld {
            a.iter()
                .zip(b.iter())
                .fold(Fld::zero(), |acc, (x, y)| acc + *x * *y)
        }

        fn scale<Fld: Field>(&self, buf: &mut Vec<Fld>, scalar: Fld) {
            for x in buf.iter_mut() {
                *x *= scalar;
            }
        }

        fn add<Fld: Field>(&self, a: &Vec<Fld>, b: &Vec<Fld>) -> Vec<Fld> {
            a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
        }

        fn sub<Fld: Field>(&self, a: &Vec<Fld>, b: &Vec<Fld>) -> Vec<Fld> {
            a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect()
        }

        fn accumulate<Fld: Field>(&self, buf: &mut Vec<Fld>, scalar: Fld, other: &Vec<Fld>) {
            for (x, y) in buf.iter_mut().zip(other.iter()) {
                *x += scalar * *y;
            }
        }

        fn make_witness<Fld: Field>(
            &self,
            _kernel: &MockKernel<Fld>,
            _inputs: Vec<Vec<Fld>>,
            _challenges: &[Fld],
        ) -> MockSumcheckWitness<Fld> {
            MockSumcheckWitness {
                _marker: std::marker::PhantomData,
            }
        }

        fn upload_sparse<Fld: Field>(
            &self,
            entries: &[(usize, Vec<Fld>)],
        ) -> Vec<(usize, Vec<Fld>)> {
            entries.to_vec()
        }

        fn sparse_reduce<Fld: Field>(
            &self,
            _entries: &Vec<(usize, Vec<Fld>)>,
            _eq: &Vec<Fld>,
            _kernel: &MockKernel<Fld>,
            _challenges: &[Fld],
            num_evals: usize,
        ) -> Vec<Fld> {
            vec![Fld::zero(); num_evals]
        }

        fn sparse_bind<Fld: Field>(
            &self,
            _entries: &mut Vec<(usize, Vec<Fld>)>,
            _challenge: Fld,
            _order: BindingOrder,
        ) {
        }
    }

    fn make_hybrid(threshold: usize) -> HybridBackend<MockBackend, MockBackend> {
        HybridBackend::new(
            MockBackend::new("primary"),
            MockBackend::new("fallback"),
            threshold,
        )
    }

    fn is_primary<T: Scalar>(buf: &HybridBuffer<T, MockBackend, MockBackend>) -> bool {
        matches!(buf, HybridBuffer::Primary(_))
    }

    #[test]
    fn upload_large_goes_to_primary() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..16).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 16);
    }

    #[test]
    fn upload_small_goes_to_fallback() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 4);
    }

    #[test]
    fn interpolate_pairs_inplace_migrates_at_threshold() {
        let hybrid = make_hybrid(4);
        // Start with 8 elements on primary.
        let data: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(is_primary(&buf));

        // After bind: 4 elements — at threshold, should migrate.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf), "should have migrated to fallback");
        assert_eq!(hybrid.len(&buf), 4);
    }

    #[test]
    fn interpolate_pairs_inplace_stays_primary_above_threshold() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..16).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(is_primary(&buf));

        // After bind: 8 elements — above threshold, stays primary.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(is_primary(&buf), "should stay on primary");
        assert_eq!(hybrid.len(&buf), 8);
    }

    #[test]
    fn interpolate_pairs_fallback_stays_fallback() {
        let hybrid = make_hybrid(4);
        // Upload small — goes to fallback.
        let data: Vec<Fr> = (0..4).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(!is_primary(&buf));

        // Bind on fallback — stays fallback.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(2), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 2);
    }

    #[test]
    fn pairwise_reduce_dispatches_to_correct_backend() {
        use jolt_compiler::{CompositionFormula, Factor, ProductTerm};

        let hybrid = make_hybrid(4);
        let formula = CompositionFormula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![
                Factor::Input(0),
                Factor::Input(1),
                Factor::Input(2),
                Factor::Input(3),
            ],
        }]);
        let kernel = hybrid.compile_kernel::<Fr>(&formula);

        // Primary inputs with Weighted eq → dispatches to primary → returns all 1s.
        let large: Vec<Fr> = vec![Fr::from_u64(1); 16];
        let weights_large: Vec<Fr> = vec![Fr::from_u64(1); 8];
        let buf_a = hybrid.upload(&large);
        let buf_w = hybrid.upload(&weights_large);
        assert!(is_primary(&buf_a));

        let result = hybrid.pairwise_reduce(
            &[&buf_a],
            EqInput::Weighted(&buf_w),
            &kernel,
            4,
            BindingOrder::LowToHigh,
        );
        assert_eq!(result, vec![Fr::from_u64(1); 4]);

        // Fallback inputs with Weighted eq → dispatches to fallback → returns all 2s.
        let small: Vec<Fr> = vec![Fr::from_u64(1); 4];
        let weights_small: Vec<Fr> = vec![Fr::from_u64(1); 2];
        let buf_b = hybrid.upload(&small);
        let buf_ws = hybrid.upload(&weights_small);
        assert!(!is_primary(&buf_b));

        let result = hybrid.pairwise_reduce(
            &[&buf_b],
            EqInput::Weighted(&buf_ws),
            &kernel,
            4,
            BindingOrder::LowToHigh,
        );
        assert_eq!(result, vec![Fr::from_u64(2); 4]);

        // Unit eq also dispatches correctly.
        let result = hybrid.pairwise_reduce(
            &[&buf_a],
            EqInput::Unit,
            &kernel,
            4,
            BindingOrder::LowToHigh,
        );
        assert_eq!(result, vec![Fr::from_u64(1); 4]);
    }

    #[test]
    fn interpolate_pairs_consuming_migrates() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let buf = hybrid.upload(&data);
        assert!(is_primary(&buf));

        // Result has 4 elements — at threshold, should migrate.
        let result = hybrid.interpolate_pairs(buf, Fr::from_u64(2));
        assert!(!is_primary(&result));
        assert_eq!(hybrid.len(&result), 4);
    }

    #[test]
    fn product_table_respects_threshold() {
        let hybrid = make_hybrid(8);

        // 2^2 = 4 elements — below threshold → fallback.
        let small_table = hybrid.product_table::<Fr>(&[Fr::from_u64(1), Fr::from_u64(2)]);
        assert!(!is_primary(&small_table));

        // 2^4 = 16 elements — above threshold → primary.
        let large_table = hybrid.product_table::<Fr>(&[
            Fr::from_u64(1),
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(4),
        ]);
        assert!(is_primary(&large_table));
    }

    #[test]
    fn download_works_from_both_backends() {
        let hybrid = make_hybrid(4);
        let large: Vec<Fr> = (0..8).map(Fr::from_u64).collect();
        let small: Vec<Fr> = (0..4).map(Fr::from_u64).collect();

        let buf_p = hybrid.upload(&large);
        let buf_f = hybrid.upload(&small);

        assert_eq!(hybrid.download(&buf_p), large);
        assert_eq!(hybrid.download(&buf_f), small);
    }

    #[test]
    fn sum_dispatches_correctly() {
        let hybrid = make_hybrid(4);
        let large: Vec<Fr> = vec![Fr::from_u64(3); 8];
        let small: Vec<Fr> = vec![Fr::from_u64(5); 2];

        let buf_p = hybrid.upload(&large);
        let buf_f = hybrid.upload(&small);

        assert_eq!(hybrid.sum(&buf_p), Fr::from_u64(24));
        assert_eq!(hybrid.sum(&buf_f), Fr::from_u64(10));
    }

    #[test]
    fn full_lifecycle_primary_to_fallback() {
        let hybrid = make_hybrid(4);

        // Start large on primary.
        let data: Vec<Fr> = (0..32).map(Fr::from_u64).collect();
        let mut buf = hybrid.upload(&data);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 32);

        // Round 1: 32 → 16, stays primary.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 16);

        // Round 2: 16 → 8, stays primary.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 8);

        // Round 3: 8 → 4, migrates to fallback.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 4);

        // Round 4: 4 → 2, stays on fallback.
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::from_u64(1), BindingOrder::LowToHigh);
        assert!(!is_primary(&buf));
        assert_eq!(hybrid.len(&buf), 2);
    }

    #[test]
    fn bind_values_correct_after_migration() {
        let hybrid = make_hybrid(4);
        let data: Vec<Fr> = vec![
            Fr::from_u64(10),
            Fr::from_u64(20),
            Fr::from_u64(30),
            Fr::from_u64(40),
            Fr::from_u64(50),
            Fr::from_u64(60),
            Fr::from_u64(70),
            Fr::from_u64(80),
        ];
        let mut buf = hybrid.upload(&data);

        // Bind with scalar=0 → takes all "lo" values: [10, 30, 50, 70].
        hybrid.interpolate_pairs_inplace(&mut buf, Fr::zero(), BindingOrder::LowToHigh);
        let result = hybrid.download(&buf);
        assert_eq!(
            result,
            vec![
                Fr::from_u64(10),
                Fr::from_u64(30),
                Fr::from_u64(50),
                Fr::from_u64(70),
            ]
        );
        // Should have migrated (4 ≤ threshold=4).
        assert!(!is_primary(&buf));
    }

    #[test]
    #[should_panic(expected = "mixed buffer backends")]
    fn mixed_backends_panics() {
        use jolt_compiler::{CompositionFormula, Factor, ProductTerm};

        let hybrid = make_hybrid(4);
        let formula = CompositionFormula::from_terms(vec![ProductTerm {
            coefficient: 1,
            factors: vec![Factor::Input(0), Factor::Input(1)],
        }]);
        let kernel = hybrid.compile_kernel::<Fr>(&formula);

        let large: Vec<Fr> = vec![Fr::from_u64(1); 16];
        let small: Vec<Fr> = vec![Fr::from_u64(1); 4];
        let buf_p = hybrid.upload(&large);
        let buf_f = hybrid.upload(&small);

        // Weights on primary, but second input on fallback — should panic.
        let weights: Vec<Fr> = vec![Fr::from_u64(1); 8];
        let w_buf = hybrid.upload(&weights);

        let _ = hybrid.pairwise_reduce(
            &[&buf_p, &buf_f],
            EqInput::Weighted(&w_buf),
            &kernel,
            2,
            BindingOrder::LowToHigh,
        );
    }
}
