//! Lower the existing relation AST; this module owns no instruction formula.
use bytemuck_derive::{Pod, Zeroable};
use jolt_claims::protocols::jolt::relations::instruction::InputVirtualization;
use jolt_claims::protocols::jolt::{
    InstructionInputPublic, JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial,
    UnbatchedClaimExpr,
};
use jolt_riscv::InstructionFlags;

use super::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct InputTerms {
    pub(crate) left: u32,
    pub(crate) right: u32,
    terms: [[u32; 2]; 5],
}
impl InputTerms {
    pub(crate) fn lower() -> Result<Self, MetalError> {
        let relation = InputVirtualization::unbatched_relation();
        if relation.output_relation != JoltRelationId::InstructionInputVirtualization
            || relation.claims.len() != 2
        {
            return Err(invalid("unsupported instruction-input relation"));
        }
        let mut left = None;
        let mut right = None;
        for claim in relation.claims {
            if claim.offset
                || claim.input_relation != JoltRelationId::SpartanProductVirtualization
                || claim.output_weight != InstructionInputPublic::EqProduct.into()
            {
                return Err(invalid("unsupported instruction-input weight/offset"));
            }
            let mut terms = Vec::new();
            collect_terms(&claim.output, &mut terms)?;
            match claim.input {
                UnbatchedClaimExpr::Polynomial(JoltPolynomialId::Virtual(
                    JoltVirtualPolynomial::LeftInstructionInput,
                )) if left.is_none() => left = Some(terms),
                UnbatchedClaimExpr::Polynomial(JoltPolynomialId::Virtual(
                    JoltVirtualPolynomial::RightInstructionInput,
                )) if right.is_none() => right = Some(terms),
                _ => return Err(invalid("unsupported instruction-input input polynomial")),
            }
        }
        let (Some(left), Some(right)) = (left, right) else {
            return Err(invalid("missing instruction-input relation side"));
        };
        if left.is_empty() || right.is_empty() || left.len() + right.len() > 5 {
            return Err(invalid("instruction-input terms exceed native descriptor"));
        }
        let mut lowered = Self {
            left: left.len() as u32,
            right: right.len() as u32,
            terms: [[0; 2]; 5],
        };
        for (dst, term) in lowered.terms.iter_mut().zip(left.into_iter().chain(right)) {
            *dst = term;
        }
        Ok(lowered)
    }
    pub(crate) fn evaluate(&self, row: &NativeCore) -> [F; 2] {
        let mut result = [F::zero(); 2];
        for (i, &[flag, operand]) in self.terms[..(self.left + self.right) as usize]
            .iter()
            .enumerate()
        {
            if row.flags & (1 << flag) != 0 {
                result[usize::from(i >= self.left as usize)] += row.operand(operand);
            }
        }
        result
    }
}
fn collect_terms(expr: &UnbatchedClaimExpr, terms: &mut Vec<[u32; 2]>) -> Result<(), MetalError> {
    match expr {
        UnbatchedClaimExpr::Add(left, right) => {
            collect_terms(left, terms)?;
            collect_terms(right, terms)
        }
        UnbatchedClaimExpr::Mul(left, right) => {
            let UnbatchedClaimExpr::Polynomial(JoltPolynomialId::Virtual(
                JoltVirtualPolynomial::InstructionFlags(flag),
            )) = &**left
            else {
                return Err(invalid("native input term is not flag times operand"));
            };
            let UnbatchedClaimExpr::Polynomial(JoltPolynomialId::Virtual(operand)) = &**right
            else {
                return Err(invalid("native input operand is not a polynomial"));
            };
            let flag = match flag {
                InstructionFlags::LeftOperandIsRs1Value => 0,
                InstructionFlags::LeftOperandIsPC => 1,
                InstructionFlags::RightOperandIsRs2Value => 2,
                InstructionFlags::RightOperandIsImm => 3,
                InstructionFlags::RightOperandIsRdWriteValue => 4,
                _ => return Err(invalid("unsupported native input flag")),
            };
            let operand = match operand {
                JoltVirtualPolynomial::Rs1Value => 0,
                JoltVirtualPolynomial::UnexpandedPC => 1,
                JoltVirtualPolynomial::Rs2Value => 2,
                JoltVirtualPolynomial::Imm => 3,
                JoltVirtualPolynomial::RdWriteValue => 4,
                _ => return Err(invalid("unsupported native input operand")),
            };
            terms.push([flag, operand]);
            Ok(())
        }
        _ => Err(invalid("unsupported native input expression")),
    }
}
