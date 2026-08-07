impl<F: ::jolt_field::Field> ::jolt_claims::InputClaims<F> for DemoInputClaims<F> {
    fn canonical_order(
        &self,
    ) -> ::std::vec::Vec<::jolt_claims::protocols::jolt::JoltOpeningId> {
        ::core::iter::empty::<::jolt_claims::protocols::jolt::JoltOpeningId>()
            .chain(
                ::core::iter::once(
                    ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                        ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::PC,
                        ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                    ),
                ),
            )
            .chain(
                self
                    .ram_inc
                    .as_ref()
                    .map(|_| ::jolt_claims::protocols::jolt::JoltOpeningId::committed(
                        ::jolt_claims::protocols::jolt::JoltCommittedPolynomial::RamInc,
                        ::jolt_claims::protocols::jolt::JoltRelationId::RamReadWriteChecking,
                    )),
            )
            .chain(
                self
                    .table_flags
                    .iter()
                    .enumerate()
                    .map(|(index, _)| ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                        ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::LookupTableFlag(
                            index,
                        ),
                        ::jolt_claims::protocols::jolt::JoltRelationId::InstructionReadRaf,
                    )),
            )
            .chain(
                ::core::iter::once(
                    ::jolt_claims::protocols::jolt::JoltOpeningId::untrusted_advice(
                        ::jolt_claims::protocols::jolt::JoltRelationId::AdviceClaimReduction,
                    ),
                ),
            )
            .collect()
    }
    fn resolve_input(
        &self,
        id: &::jolt_claims::protocols::jolt::JoltOpeningId,
    ) -> ::core::option::Option<F> {
        if *id
            == ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::PC,
                ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
            )
        {
            return ::core::option::Option::Some(self.pc);
        }
        if *id
            == ::jolt_claims::protocols::jolt::JoltOpeningId::committed(
                ::jolt_claims::protocols::jolt::JoltCommittedPolynomial::RamInc,
                ::jolt_claims::protocols::jolt::JoltRelationId::RamReadWriteChecking,
            )
        {
            return self.ram_inc;
        }
        for (index, __value) in self.table_flags.iter().enumerate() {
            if *id
                == ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                    ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::LookupTableFlag(
                        index,
                    ),
                    ::jolt_claims::protocols::jolt::JoltRelationId::InstructionReadRaf,
                )
            {
                return ::core::option::Option::Some(*__value);
            }
        }
        if *id
            == ::jolt_claims::protocols::jolt::JoltOpeningId::untrusted_advice(
                ::jolt_claims::protocols::jolt::JoltRelationId::AdviceClaimReduction,
            )
        {
            return ::core::option::Option::Some(self.advice);
        }
        ::core::option::Option::None
    }
}
impl<F: ::jolt_field::Field> DemoInputClaims<::std::vec::Vec<F>> {
    pub fn pc(&self) -> &[F] {
        &self.pc
    }
    pub fn ram_inc(&self) -> ::core::option::Option<&[F]> {
        self.ram_inc.as_deref()
    }
    pub fn table_flags(&self) -> &[::std::vec::Vec<F>] {
        &self.table_flags
    }
    pub fn advice(&self) -> &[F] {
        &self.advice
    }
}
