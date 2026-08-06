impl<F: ::jolt_field::Field> ::jolt_claims::OutputClaims<F> for DemoOutputClaims<F> {
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
                ::core::iter::once(
                    ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                        ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::OpFlags(
                            CircuitFlags::VirtualInstruction,
                        ),
                        ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                    ),
                ),
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
                        ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                    )),
            )
            .chain(
                self
                    .ram_inc
                    .as_ref()
                    .map(|_| ::jolt_claims::protocols::jolt::JoltOpeningId::committed(
                        ::jolt_claims::protocols::jolt::JoltCommittedPolynomial::RamInc,
                        ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                    )),
            )
            .chain(
                ::core::iter::once(
                    ::jolt_claims::protocols::jolt::JoltOpeningId::trusted_advice(
                        ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                    ),
                ),
            )
            .collect()
    }
    fn resolve_output(
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
            == ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::OpFlags(
                    CircuitFlags::VirtualInstruction,
                ),
                ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
            )
        {
            return ::core::option::Option::Some(self.virtual_flag);
        }
        for (index, __value) in self.table_flags.iter().enumerate() {
            if *id
                == ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                    ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::LookupTableFlag(
                        index,
                    ),
                    ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                )
            {
                return ::core::option::Option::Some(*__value);
            }
        }
        if let ::core::option::Option::Some(__value) = &self.ram_inc {
            if *id
                == ::jolt_claims::protocols::jolt::JoltOpeningId::committed(
                    ::jolt_claims::protocols::jolt::JoltCommittedPolynomial::RamInc,
                    ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                )
            {
                return ::core::option::Option::Some(*__value);
            }
        }
        if *id
            == ::jolt_claims::protocols::jolt::JoltOpeningId::trusted_advice(
                ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
            )
        {
            return ::core::option::Option::Some(self.advice);
        }
        ::core::option::Option::None
    }
    fn from_opening_values(
        mut resolve: impl ::core::ops::FnMut(
            &::jolt_claims::protocols::jolt::JoltOpeningId,
        ) -> ::core::option::Option<F>,
    ) -> ::core::result::Result<
        Self,
        ::jolt_claims::MissingOpeningValue<::jolt_claims::protocols::jolt::JoltOpeningId>,
    > {
        ::core::result::Result::Ok(Self {
            pc: {
                let __id = ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                    ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::PC,
                    ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                );
                match resolve(&__id) {
                    ::core::option::Option::Some(__value) => __value,
                    ::core::option::Option::None => {
                        return ::core::result::Result::Err(::jolt_claims::MissingOpeningValue {
                            id: __id,
                        });
                    }
                }
            },
            virtual_flag: {
                let __id = ::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                    ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::OpFlags(
                        CircuitFlags::VirtualInstruction,
                    ),
                    ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                );
                match resolve(&__id) {
                    ::core::option::Option::Some(__value) => __value,
                    ::core::option::Option::None => {
                        return ::core::result::Result::Err(::jolt_claims::MissingOpeningValue {
                            id: __id,
                        });
                    }
                }
            },
            table_flags: {
                let mut __values = ::std::vec::Vec::new();
                let mut index = 0usize;
                while let ::core::option::Option::Some(__value) = resolve(
                    &::jolt_claims::protocols::jolt::JoltOpeningId::virtual_polynomial(
                        ::jolt_claims::protocols::jolt::JoltVirtualPolynomial::LookupTableFlag(
                            index,
                        ),
                        ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                    ),
                ) {
                    __values.push(__value);
                    index += 1;
                }
                __values
            },
            ram_inc: resolve(
                &::jolt_claims::protocols::jolt::JoltOpeningId::committed(
                    ::jolt_claims::protocols::jolt::JoltCommittedPolynomial::RamInc,
                    ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                ),
            ),
            advice: {
                let __id = ::jolt_claims::protocols::jolt::JoltOpeningId::trusted_advice(
                    ::jolt_claims::protocols::jolt::JoltRelationId::SpartanOuter,
                );
                match resolve(&__id) {
                    ::core::option::Option::Some(__value) => __value,
                    ::core::option::Option::None => {
                        return ::core::result::Result::Err(::jolt_claims::MissingOpeningValue {
                            id: __id,
                        });
                    }
                }
            },
        })
    }
}
impl<F: ::jolt_field::Field> DemoOutputClaims<::std::vec::Vec<F>> {
    pub fn pc(&self) -> &[F] {
        &self.pc
    }
    pub fn virtual_flag(&self) -> &[F] {
        &self.virtual_flag
    }
    pub fn table_flags(&self) -> &[::std::vec::Vec<F>] {
        &self.table_flags
    }
    pub fn ram_inc(&self) -> ::core::option::Option<&[F]> {
        self.ram_inc.as_deref()
    }
    pub fn advice(&self) -> &[F] {
        &self.advice
    }
}
