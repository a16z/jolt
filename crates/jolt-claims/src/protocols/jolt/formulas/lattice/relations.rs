use jolt_field::{FromPrimitiveInt, RingCore};

use crate::protocols::jolt::{
    IncVirtualizationChallenge, IncVirtualizationPublic, JoltChallengeId, JoltExpr, JoltPublicId,
    JoltRelationClaims, JoltRelationId, UnsignedIncChunkReconstructionChallenge,
    UnsignedIncChunkReconstructionPublic,
};
use crate::{challenge, constant, opening, public};

use super::super::dimensions::{JoltSumcheckSpec, TraceDimensions};
use super::openings::{
    inc_virtualization_inc_opening, inc_virtualization_ram_read_write_opening,
    inc_virtualization_ram_val_check_opening, inc_virtualization_rd_read_write_opening,
    inc_virtualization_rd_val_evaluation_opening, inc_virtualization_store_opening,
    unsigned_inc_chunk_opening, unsigned_inc_chunking, unsigned_inc_msb_opening,
    unsigned_inc_opening,
};

pub fn inc_virtualization_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = inc_virtualization_challenge(IncVirtualizationChallenge::Gamma);

    let input = opening(inc_virtualization_ram_read_write_opening())
        + gamma.clone() * opening(inc_virtualization_ram_val_check_opening())
        + gamma.clone().pow(2) * opening(inc_virtualization_rd_read_write_opening())
        + gamma.clone().pow(3) * opening(inc_virtualization_rd_val_evaluation_opening());

    let ram_coeff = inc_virtualization_public(IncVirtualizationPublic::EqRamReadWrite)
        + gamma.clone() * inc_virtualization_public(IncVirtualizationPublic::EqRamValCheck);
    let gamma_2 = gamma.clone().pow(2);
    let rd_coeff = inc_virtualization_public(IncVirtualizationPublic::EqRegistersReadWrite)
        + gamma.clone()
            * inc_virtualization_public(IncVirtualizationPublic::EqRegistersValEvaluation);
    let store = opening(inc_virtualization_store_opening());
    let output = opening(inc_virtualization_inc_opening())
        * (ram_coeff * store.clone() + gamma_2 * rd_coeff * (JoltExpr::one() - store));

    JoltRelationClaims::new(
        JoltRelationId::IncVirtualization,
        dimensions.sumcheck(3),
        input,
        output,
    )
}

pub fn unsigned_inc_claim_reduction_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let input = opening(inc_virtualization_inc_opening()) + constant(F::from_u128(1u128 << 64));
    let output = opening(unsigned_inc_opening());

    JoltRelationClaims::new(
        JoltRelationId::UnsignedIncClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

pub fn unsigned_inc_msb_booleanity_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let msb = opening(unsigned_inc_msb_opening());
    JoltRelationClaims::new(
        JoltRelationId::Booleanity,
        dimensions.sumcheck(2),
        JoltExpr::zero(),
        msb.clone() * msb.clone() - msb,
    )
}

pub fn unsigned_inc_chunk_reconstruction_claim<F>(
    log_k_chunk: usize,
) -> Option<JoltRelationClaims<F>>
where
    F: RingCore + FromPrimitiveInt,
{
    let chunking = unsigned_inc_chunking(log_k_chunk)?;
    let gamma =
        unsigned_inc_chunk_reconstruction_challenge(UnsignedIncChunkReconstructionChallenge::Gamma);
    let eq_booleanity_address = unsigned_inc_chunk_reconstruction_public(
        UnsignedIncChunkReconstructionPublic::EqBooleanityAddress,
    );
    let identity_at_address = unsigned_inc_chunk_reconstruction_public(
        UnsignedIncChunkReconstructionPublic::IdentityAtAddress,
    );
    let delta = gamma.clone().pow(2 * chunking.chunk_count);
    let lower_value = opening(unsigned_inc_opening())
        - constant(F::from_u128(1u128 << 64)) * opening(unsigned_inc_msb_opening());

    let mut input = delta.clone() * lower_value;
    let mut output = JoltExpr::zero();
    let mut place = F::one();
    for index in 0..chunking.chunk_count {
        input = input
            + gamma.clone().pow(2 * index)
            + gamma.clone().pow(2 * index + 1) * opening(unsigned_inc_chunk_opening(index));
        let output_coeff = gamma.clone().pow(2 * index)
            + gamma.clone().pow(2 * index + 1) * eq_booleanity_address.clone()
            + delta.clone() * constant(place) * identity_at_address.clone();
        output = output + output_coeff * opening(unsigned_inc_chunk_opening(index));
        place *= F::from_u64(chunking.radix);
    }

    Some(
        JoltRelationClaims::new(
            JoltRelationId::UnsignedIncChunkReconstruction,
            JoltSumcheckSpec::boolean(log_k_chunk, 3),
            input,
            output,
        )
        .with_input_challenges([JoltChallengeId::from(
            UnsignedIncChunkReconstructionChallenge::Gamma,
        )]),
    )
}

fn inc_virtualization_challenge<F>(id: IncVirtualizationChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn inc_virtualization_public<F>(id: IncVirtualizationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn unsigned_inc_chunk_reconstruction_challenge<F>(
    id: UnsignedIncChunkReconstructionChallenge,
) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn unsigned_inc_chunk_reconstruction_public<F>(
    id: UnsignedIncChunkReconstructionPublic,
) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}
