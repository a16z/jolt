//! The byte link's prover side; the protocol is specified with its verifier,
//! `jolt_verifier::stages::byte_link`.
//!
//! A prover drives Fiat–Shamir through [`ByteLinkTranscript`]: it hands every
//! message over in protocol order and takes every challenge from it, so labels
//! and wire live with the stage.
//! Points crossing this boundary are canonical MSB-first; inside, round `i` of a
//! sumcheck binds bit `i` of the remaining index, so a layer's child point is
//! `(reverse(s), µ)`.
//!
//! [`reference`] proves on the host; `metal::solinas::byte_link` proves on the
//! GPU with the same messages. A backend's [`ByteLinkKernel`] picks one.

#[cfg(any(test, feature = "test-utils"))]
pub mod fixtures;
pub mod reference;

use jolt_claims::protocols::jolt::lattice::byte_link::{
    ByteLinkBatch, ByteLinkInputs, HistogramGroup,
};
use jolt_field::JoltField;
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::byte_link::{ByteLinkCompression, ByteLinkOpenings};

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::metal::solinas::byte_link::ByteLinkHistograms;
use crate::KernelError;
use reference::{ByteTrace, Histograms};

/// Starts a link proof over a byte trace.
pub trait ByteLinkKernel<F: JoltField>: Send + Sync {
    /// `W` of every pack of `trace` at the stage-6b cycle point.
    fn histograms<'a>(
        &self,
        trace: ByteTrace<'a>,
        inputs: &ByteLinkInputs<F>,
    ) -> Result<Box<dyn ByteLinkRun<F> + 'a>, KernelError<F>>;
}

/// One link proof once `W` exists.
pub trait ByteLinkRun<F: JoltField> {
    /// The tables the stage commits as the two histogram groups.
    fn tables(&self) -> HistogramTables<'_, F>;

    /// Everything after the `W` commitments and the compression challenges.
    fn prove(
        self: Box<Self>,
        inputs: &ByteLinkInputs<F>,
        compression: &ByteLinkCompression<F>,
        transcript: &mut dyn ByteLinkTranscript<F>,
    ) -> Result<ByteLinkOpenings<F>, KernelError<F>>;
}

/// Where a run holds `W`.
pub enum HistogramTables<'a, F> {
    Host(&'a Histograms<F>),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Device(&'a ByteLinkHistograms),
}

/// A prover message, in protocol order. `layer` counts the variables of the
/// layer's parent index, so the top layer of a batch is 0; `round` indexes the
/// rounds of its sumcheck. Messages marked derived carry values the verifier
/// computes itself, so the stage absorbs them without putting them on the wire.
#[derive(Clone, Copy, Debug)]
pub enum ByteLinkMessage<'a, F: JoltField> {
    /// `(P, B)` roots of the seven trace trees, then of the seven table trees.
    Roots(&'a [(F, F)]),
    /// Derived: the `(P, B)` claims every tree of the batch brings into
    /// `layer`, at `point`.
    LayerClaims {
        batch: ByteLinkBatch,
        layer: usize,
        point: &'a [F],
        claims: &'a [(F, F)],
    },
    /// A cubic round polynomial of a layer sumcheck.
    LayerRound {
        batch: ByteLinkBatch,
        layer: usize,
        round: usize,
        poly: &'a UnivariatePoly<F>,
    },
    /// The bound children `[P_0, B_0, P_1, B_1]` of every tree after the
    /// layer's last round.
    Children {
        batch: ByteLinkBatch,
        layer: usize,
        children: &'a [[F; 4]],
    },
    /// Derived: the values a group's query reduction batches, per pack its
    /// marginal query values, then its table leaf `W(y)`.
    QueryValues {
        group: HistogramGroup,
        values: &'a [F],
    },
    /// A quadratic round polynomial of a query reduction.
    QueryRound {
        group: HistogramGroup,
        round: usize,
        poly: &'a UnivariatePoly<F>,
    },
    /// `W` of every pack of the group at the reduction's final point.
    QueryFinals {
        group: HistogramGroup,
        values: &'a [F],
    },
    /// Derived: the trace leaf point `z` and the seven denominator leaf claims
    /// `B_j(z)`.
    SourceClaims {
        point: &'a [F],
        denominators: &'a [F],
    },
    /// A quadratic round polynomial of the `Q` reduction.
    SourceRound {
        round: usize,
        poly: &'a UnivariatePoly<F>,
    },
    /// Every column of `Q` at the reduction's final point, in slot order.
    SourceFinals(&'a [F]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteLinkDraw {
    /// Independent `(λ_P, λ_B)` per tree of the batch, as `2 · trees` scalars.
    LayerWeights {
        batch: ByteLinkBatch,
        layer: usize,
    },
    LayerChallenge {
        batch: ByteLinkBatch,
        layer: usize,
        round: usize,
    },
    /// The child selector `µ` after a layer.
    ChildSelector {
        batch: ByteLinkBatch,
        layer: usize,
    },
    /// One independent weight per query value.
    QueryWeights {
        group: HistogramGroup,
    },
    QueryChallenge {
        group: HistogramGroup,
        round: usize,
    },
    /// The point `θ` of the zero-slot claims `D_30(θ) = D_31(θ) = 0`,
    /// `log_rows` scalars MSB-first.
    ZeroSlotPoint,
    /// Ten scalars: `α_j` of the seven denominators, `α_F`, then the zero
    /// slots' `α_30, α_31`.
    SourceWeights,
    SourceChallenge {
        round: usize,
    },
}

/// The Fiat–Shamir side of the link, implemented by the stage code.
pub trait ByteLinkTranscript<F: JoltField> {
    fn append(&mut self, message: ByteLinkMessage<'_, F>);
    fn challenges(&mut self, draw: ByteLinkDraw, count: usize) -> Vec<F>;
}

impl<F: JoltField, T: ByteLinkTranscript<F> + ?Sized> ByteLinkTranscript<F> for &mut T {
    fn append(&mut self, message: ByteLinkMessage<'_, F>) {
        (**self).append(message);
    }

    fn challenges(&mut self, draw: ByteLinkDraw, count: usize) -> Vec<F> {
        (**self).challenges(draw, count)
    }
}
