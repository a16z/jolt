//! Architecture-specific Fp128 addition and subtraction kernels.

#[cfg(any(
    test,
    feature = "fuzzing",
    not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))
))]
use super::{join, split};
use super::{pack, Fp128};
#[cfg(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))]
use std::arch::asm;

impl<const P: u128> Fp128<P> {
    #[inline(always)]
    pub(super) fn add_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        {
            // Keep the reduction predicate in flags through `ccmp`.
            Self::add_raw_aarch64_dispatch(a, b)
        }

        #[cfg(all(feature = "asm", target_arch = "x86_64"))]
        {
            // Materialize the carry as a mask, then use `cmovne` for the
            // final branchless selection.
            Self::add_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
        {
            Self::add_raw_portable(a, b)
        }
    }

    #[cfg(any(
        test,
        feature = "fuzzing",
        not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))
    ))]
    #[inline(always)]
    pub(super) fn add_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // Compute s = a + b as two limbs.
        let (s0, carry0) = a[0].overflowing_add(b[0]);
        let (s1a, carry1a) = a[1].overflowing_add(b[1]);
        let (s1, carry1b) = s1a.overflowing_add(carry0 as u64);
        let overflow = carry1a | carry1b;

        // Since p = 2^128 - C and C < 2^64, reduction adds C to
        // the low limb and propagates the carry.
        let (r0, carry2) = s0.overflowing_add(Self::C_LO);
        let (r1, carry3) = s1.overflowing_add(carry2 as u64);

        pack(
            if overflow | carry3 { r0 } else { s0 },
            if overflow | carry3 { r1 } else { s1 },
        )
    }

    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn add_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        Self::add_raw_aarch64_asm(a, b)
    }

    /// Parameterized addition through the instruction body proved generically
    /// over every valid offset in HOL Light.
    ///
    /// Register contract for `fp128_add_body.inc`:
    ///
    /// - `x0:x1` starts with `a` and finishes with the result.
    /// - `x2:x3` contains `b` and is not changed.
    /// - `x4` contains `C = 2^128 - P` and is not changed.
    /// - `x5:x9` and the condition flags are temporary state.
    /// - The body does not access memory or the stack.
    ///
    /// The field representation requires canonical inputs. The machine theorem
    /// uses the same assumption. This function adds no runtime range check.
    /// The formal verification workflow checks the exact object and optimized
    /// public operation witness words.
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn add_raw_aarch64_asm(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [mut out_lo, mut out_hi] = a;
        let [b_lo, b_hi] = b;
        // SAFETY: The register contract is listed above. The body declares
        // every clobber and does not access memory or the stack.
        unsafe {
            asm!(
                include_str!("../../../asm/aarch64/fp128_add_body.inc"),
                inout("x0") out_lo,
                inout("x1") out_hi,
                in("x2") b_lo,
                in("x3") b_hi,
                in("x4") Self::C_LO,
                out("x5") _,
                out("x6") _,
                out("x7") _,
                out("x8") _,
                out("x9") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    #[inline(always)]
    fn add_raw_x86_64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        Self::add_raw_x86_64_asm(a, b)
    }

    /// Parameterized addition through the instruction body proved generically
    /// over every valid offset in HOL Light.
    ///
    /// Register contract for `fp128_add_body.inc`:
    ///
    /// - `rdi:rsi` starts with `a` and finishes with the result.
    /// - `rdx:rcx` contains `b` and is not changed.
    /// - `r8` contains `C = 2^128 - P` and is not changed.
    /// - `r9:r11` and the condition flags are temporary state.
    /// - The body does not access memory or the stack.
    ///
    /// The field representation requires canonical inputs. The machine theorem
    /// uses the same assumption. This function adds no runtime range check.
    /// The formal verification workflow checks the exact object bytes and the
    /// optimized public operation witness.
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    #[inline(always)]
    fn add_raw_x86_64_asm(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [mut out_lo, mut out_hi] = a;
        let [b_lo, b_hi] = b;
        // SAFETY: The register contract is listed above. The body declares
        // every clobber and does not access memory or the stack.
        unsafe {
            asm!(
                include_str!("../../../asm/x86_64/fp128_add_body.inc"),
                inout("rdi") out_lo,
                inout("rsi") out_hi,
                in("rdx") b_lo,
                in("rcx") b_hi,
                in("r8") Self::C_LO,
                out("r9") _,
                out("r10") _,
                out("r11") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[inline(always)]
    pub(super) fn sub_raw(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        #[cfg(all(feature = "asm", target_arch = "aarch64"))]
        {
            // Keep the borrow in flags and reduce by subtracting C.
            Self::sub_raw_aarch64_dispatch(a, b)
        }

        #[cfg(all(feature = "asm", target_arch = "x86_64"))]
        {
            // Turn the borrow into a mask, select 0 or C, and subtract it.
            Self::sub_raw_x86_64_dispatch(a, b)
        }

        #[cfg(not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64"))))]
        {
            Self::sub_raw_portable(a, b)
        }
    }

    #[cfg(any(
        test,
        feature = "fuzzing",
        not(all(feature = "asm", any(target_arch = "aarch64", target_arch = "x86_64")))
    ))]
    #[inline(always)]
    pub(super) const fn sub_raw_portable(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (diff, borrow) = join(a).overflowing_sub(join(b));
        split(if borrow { diff.wrapping_add(P) } else { diff })
    }

    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn sub_raw_aarch64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        Self::sub_raw_aarch64_asm(a, b)
    }

    /// Parameterized subtraction through the instruction body proved
    /// generically over every valid offset in HOL Light.
    ///
    /// Register contract for `fp128_sub_body.inc`:
    ///
    /// - `x0:x1` starts with `a` and finishes with the result.
    /// - `x2:x3` contains `b` and is not changed.
    /// - `x4` contains `C = 2^128 - P` and is not changed.
    /// - `x5:x7` and the condition flags are temporary state.
    /// - The body does not access memory or the stack.
    ///
    /// The field representation requires canonical inputs. The machine theorem
    /// uses the same assumption. This function adds no runtime range check.
    /// The formal verification workflow checks the exact object and optimized
    /// public operation witness words.
    #[cfg(all(feature = "asm", target_arch = "aarch64"))]
    #[inline(always)]
    fn sub_raw_aarch64_asm(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [mut out_lo, mut out_hi] = a;
        let [b_lo, b_hi] = b;
        // SAFETY: The register contract is listed above. The body declares
        // every clobber and does not access memory or the stack.
        unsafe {
            asm!(
                include_str!("../../../asm/aarch64/fp128_sub_body.inc"),
                inout("x0") out_lo,
                inout("x1") out_hi,
                in("x2") b_lo,
                in("x3") b_hi,
                in("x4") Self::C_LO,
                out("x5") _,
                out("x6") _,
                out("x7") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }

    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    #[inline(always)]
    fn sub_raw_x86_64_dispatch(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        Self::sub_raw_x86_64_asm(a, b)
    }

    /// Parameterized subtraction through the instruction body proved
    /// generically over every valid offset in HOL Light.
    ///
    /// Register contract for `fp128_sub_body.inc`:
    ///
    /// - `rdi:rsi` starts with `a` and finishes with the result.
    /// - `rdx:rcx` contains `b` and is not changed.
    /// - `r8` contains `C = 2^128 - P` and is not changed.
    /// - `r9` and the condition flags are temporary state.
    /// - The body does not access memory or the stack.
    ///
    /// The field representation requires canonical inputs. The machine theorem
    /// uses the same assumption. This function adds no runtime range check.
    /// The formal verification workflow checks the exact object bytes and the
    /// optimized public operation witness.
    #[cfg(all(feature = "asm", target_arch = "x86_64"))]
    #[inline(always)]
    fn sub_raw_x86_64_asm(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let [mut out_lo, mut out_hi] = a;
        let [b_lo, b_hi] = b;
        // SAFETY: The register contract is listed above. The body declares
        // every clobber and does not access memory or the stack.
        unsafe {
            asm!(
                include_str!("../../../asm/x86_64/fp128_sub_body.inc"),
                inout("rdi") out_lo,
                inout("rsi") out_hi,
                in("rdx") b_lo,
                in("rcx") b_hi,
                in("r8") Self::C_LO,
                out("r9") _,
                options(pure, nomem, nostack),
            );
        }
        pack(out_lo, out_hi)
    }
}
