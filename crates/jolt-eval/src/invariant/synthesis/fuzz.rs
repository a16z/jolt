/// Macro that generates a libfuzzer fuzz target for an invariant.
///
/// Takes a concrete invariant expression. Setup is performed once;
/// each fuzz iteration produces an `Input` via `Arbitrary` and checks it.
///
/// # Usage
///
/// ```ignore
/// #![no_main]
/// use jolt_eval::invariant::split_eq_bind::SplitEqBindLowHighInvariant;
/// jolt_eval::fuzz_invariant!(SplitEqBindLowHighInvariant::default());
/// ```
#[macro_export]
macro_rules! fuzz_invariant {
    ($inv:expr) => {
        use $crate::Invariant as _;
        use $crate::InvariantTargets as _;

        // Assert at init time that this invariant includes the Fuzz target.
        fn __assert_fuzz_target<I: $crate::InvariantTargets>(inv: &I) {
            assert!(
                inv.targets()
                    .contains($crate::SynthesisTarget::Fuzz),
                "Invariant does not include SynthesisTarget::Fuzz"
            );
        }

        static __FUZZ_SETUP: ::std::sync::OnceLock<
            ::std::boxed::Box<dyn ::std::any::Any + ::std::marker::Send + ::std::marker::Sync>,
        > = ::std::sync::OnceLock::new();

        fn __fuzz_init<I: $crate::Invariant>(inv: &I) {
            __FUZZ_SETUP
                .set(::std::boxed::Box::new(inv.setup()))
                .ok();
        }

        fn __fuzz_check<I: $crate::Invariant>(inv: &I, data: &[u8]) {
            let setup = __FUZZ_SETUP
                .get()
                .expect("SETUP not initialized")
                .downcast_ref::<I::Setup>()
                .expect("wrong setup type");
            let mut u = $crate::arbitrary::Unstructured::new(data);
            if let Ok(input) = <I::Input as $crate::arbitrary::Arbitrary>::arbitrary(&mut u) {
                match inv.check(setup, input) {
                    Ok(()) | Err($crate::CheckError::InvalidInput(_)) => {}
                    Err($crate::CheckError::Violation(v)) => {
                        panic!("Invariant violated: {v}");
                    }
                }
            }
        }

        ::libfuzzer_sys::fuzz_target!(
            init: {
                __assert_fuzz_target(&$inv);
                __fuzz_init(&$inv);
            },
            |data: &[u8]| {
                __fuzz_check(&$inv, data);
            }
        );
    };
}
