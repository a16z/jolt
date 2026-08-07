impl<F: ::jolt_field::Field> ::jolt_claims::SumcheckChallenges<F> for DemoChallenges<F> {
    fn from_transcript_values<__I: ::core::iter::Iterator<Item = F>>(
        values: __I,
    ) -> ::core::result::Result<Self, ::jolt_claims::ChallengeDrawError> {
        let mut __values = values;
        let tau = __values
            .next()
            .ok_or(::jolt_claims::ChallengeDrawError::StreamExhausted {
                required: 2usize,
                populated: 0usize,
            })?;
        let r_inner = __values
            .next()
            .ok_or(::jolt_claims::ChallengeDrawError::StreamExhausted {
                required: 2usize,
                populated: 1usize,
            })?;
        ::core::result::Result::Ok(Self { tau, r_inner })
    }
    fn resolve_challenge(
        &self,
        id: &::jolt_claims::protocols::jolt::JoltChallengeId,
    ) -> ::core::option::Option<F> {
        if *id
            == ::jolt_claims::protocols::jolt::JoltChallengeId::from(
                SpartanChallenges::Tau,
            )
        {
            return ::core::option::Option::Some(self.tau);
        }
        if *id
            == ::jolt_claims::protocols::jolt::JoltChallengeId::from(
                SpartanChallenges::RInner,
            )
        {
            return ::core::option::Option::Some(self.r_inner);
        }
        ::core::option::Option::None
    }
}
