//! The lattice-mode final-opening endgame: what happens to each committed
//! polynomial's leaf claim instead of the base stage-8 RLC batch (which needs
//! the commitment homomorphism Akita lacks).

use serde::{Deserialize, Serialize};

use super::super::{JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use super::ids::LatticeColumn;
use super::views::LatticeView;

/// How one committed polynomial's final claim is discharged in lattice mode.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeFinalOpening {
    /// One `PrefixPackedClaim` on the packed witness. `leaf` names the
    /// relation output the claim's value comes from; `None` means the claim
    /// is produced by the view-discharge reduction rather than a relation
    /// (trusted-advice bytes and the precommitted sub-columns).
    Packed {
        column: LatticeColumn,
        leaf: Option<JoltOpeningId>,
    },
    /// The logical value is a weighted sum of packed cells; discharged
    /// through the view's [`DecodeTerm`](super::views::DecodeTerm) list.
    Decoded { view: LatticeView },
    /// Never PCS-opened: the polynomial's consumer claims leave the base PIOP
    /// through lattice relations (the `IncVirtualization` chain).
    Virtualized,
}

/// The discharge for each committed polynomial's final opening. Total over
/// `JoltCommittedPolynomial` so a new committed polynomial cannot land in
/// lattice mode without a discharge decision.
pub fn final_opening(polynomial: JoltCommittedPolynomial) -> LatticeFinalOpening {
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            LatticeFinalOpening::Virtualized
        }
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_)
        | JoltCommittedPolynomial::UnsignedIncChunk(_)
        | JoltCommittedPolynomial::UnsignedIncMsb
        | JoltCommittedPolynomial::TrustedAdviceBytes
        | JoltCommittedPolynomial::UntrustedAdviceBytes => LatticeFinalOpening::Packed {
            column: LatticeColumn::from(polynomial),
            leaf: packed_column_leaf(LatticeColumn::from(polynomial)),
        },
        JoltCommittedPolynomial::TrustedAdvice => LatticeFinalOpening::Decoded {
            view: LatticeView::AdviceWord {
                kind: JoltAdviceKind::Trusted,
            },
        },
        JoltCommittedPolynomial::UntrustedAdvice => LatticeFinalOpening::Decoded {
            view: LatticeView::AdviceWord {
                kind: JoltAdviceKind::Untrusted,
            },
        },
        JoltCommittedPolynomial::BytecodeChunk(chunk) => LatticeFinalOpening::Decoded {
            view: LatticeView::BytecodeChunkLanes { chunk },
        },
        JoltCommittedPolynomial::ProgramImageInit => LatticeFinalOpening::Decoded {
            view: LatticeView::ProgramImageWord,
        },
    }
}

/// The relation output supplying a packed column's claim value, or `None`
/// when the view-discharge reduction produces it (view-only sub-columns and
/// trusted-advice bytes, which have no in-protocol validity relation).
pub fn packed_column_leaf(column: LatticeColumn) -> Option<JoltOpeningId> {
    let LatticeColumn::Committed(polynomial) = column else {
        return None;
    };
    let relation = match polynomial {
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_) => JoltRelationId::HammingWeightClaimReduction,
        JoltCommittedPolynomial::UnsignedIncChunk(_) => {
            JoltRelationId::UnsignedIncChunkReconstruction
        }
        JoltCommittedPolynomial::UnsignedIncMsb => JoltRelationId::Booleanity,
        JoltCommittedPolynomial::UntrustedAdviceBytes => JoltRelationId::AdviceBytesValidity,
        _ => return None,
    };
    Some(JoltOpeningId::committed(polynomial, relation))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::super::super::geometry::ra::JoltRaPolynomialLayout;
    use super::super::packing::{proof_packed_columns, ProofPackingShape};
    use super::*;

    #[test]
    fn incs_are_virtualized_and_never_opened() {
        assert_eq!(
            final_opening(JoltCommittedPolynomial::RdInc),
            LatticeFinalOpening::Virtualized
        );
        assert_eq!(
            final_opening(JoltCommittedPolynomial::RamInc),
            LatticeFinalOpening::Virtualized
        );
    }

    #[test]
    fn decomposed_polynomials_discharge_through_views() {
        assert_eq!(
            final_opening(JoltCommittedPolynomial::BytecodeChunk(3)),
            LatticeFinalOpening::Decoded {
                view: LatticeView::BytecodeChunkLanes { chunk: 3 }
            }
        );
        assert_eq!(
            final_opening(JoltCommittedPolynomial::UntrustedAdvice),
            LatticeFinalOpening::Decoded {
                view: LatticeView::AdviceWord {
                    kind: JoltAdviceKind::Untrusted
                }
            }
        );
        assert_eq!(
            final_opening(JoltCommittedPolynomial::ProgramImageInit),
            LatticeFinalOpening::Decoded {
                view: LatticeView::ProgramImageWord
            }
        );
    }

    #[test]
    fn every_proof_packed_column_has_a_claim_source() {
        let shape = ProofPackingShape {
            ra_layout: JoltRaPolynomialLayout::new(2, 1, 1).unwrap(),
            log_t: 5,
            log_k_chunk: 8,
            untrusted_advice_word_vars: Some(4),
        };
        for (column, _) in proof_packed_columns(&shape).unwrap() {
            let leaf = packed_column_leaf(column).unwrap();
            // The leaf claim opens the column's own polynomial.
            let LatticeColumn::Committed(polynomial) = column else {
                panic!("proof packed columns are committed polynomials");
            };
            assert_eq!(
                leaf,
                JoltOpeningId::committed(
                    polynomial,
                    match leaf {
                        JoltOpeningId::Polynomial { relation, .. } => relation,
                        _ => unreachable!(),
                    }
                )
            );
            assert_eq!(
                final_opening(polynomial),
                LatticeFinalOpening::Packed {
                    column,
                    leaf: Some(leaf)
                }
            );
        }
    }

    #[test]
    fn view_only_columns_have_no_relation_leaf() {
        assert_eq!(packed_column_leaf(LatticeColumn::ProgramImageBytes), None);
        assert_eq!(
            packed_column_leaf(LatticeColumn::Committed(
                JoltCommittedPolynomial::TrustedAdviceBytes
            )),
            None
        );
        assert_eq!(
            final_opening(JoltCommittedPolynomial::TrustedAdviceBytes),
            LatticeFinalOpening::Packed {
                column: LatticeColumn::Committed(JoltCommittedPolynomial::TrustedAdviceBytes),
                leaf: None,
            }
        );
    }
}
