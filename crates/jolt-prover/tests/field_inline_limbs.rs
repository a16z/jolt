//! Import policy, width, and dishonest-advice checks for the SDK limb
//! conversions, traced in the build's proof field (BN254 for Dory, fp128 for
//! Akita). Proven acceptance of the same guest lives in `e2e_matrix.rs`.
//!
//! Dishonest advice comes from the tracer's injection hook, which emits chosen
//! FIELD_ADVICE_LIMB limbs with the quotients the row constraint then forces.
//! Nonzero residuals stop honest tracing at the FIELD_ASSERT_ZERO that the
//! residual-quotient row also enforces; noncanonical limbs reach the guest's
//! `< p` check, which spoils the proof.

#![cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
#![expect(clippy::expect_used, reason = "integration tests should fail loudly")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use common::jolt_device::JoltDevice;
#[cfg(feature = "akita")]
use jolt_akita::AkitaField as Field;
use jolt_field::CanonicalBytes;
#[cfg(not(feature = "akita"))]
use jolt_field::Fr as Field;
use jolt_field::Ring;
use jolt_host::field_inline::canonical_limbs;
use jolt_host::Program;
use tracer::instruction::field_inline::advice_limb::inject_advice_limbs;
use tracer::instruction::Cycle;

/// Limb width of the proof field's modulus.
const MODULUS_LIMBS: usize = Field::NUM_BYTES / 8;

struct Run {
    device: JoltDevice,
    /// The trace holds spoil_proof's unsatisfiable assertion.
    spoiled: bool,
}

impl Run {
    fn output(&self) -> [u64; 6] {
        postcard::take_from_bytes(&self.device.outputs)
            .expect("decode the readout")
            .0
    }
}

/// Builds and traces the conformance guest, which imports `value` and reads it
/// back `readout_width` limbs wide before its fixed checks.
fn trace(value: [u64; 6], readout_width: u8) -> Run {
    let mut program = Program::new("field-limbs-guest");
    program.enable_field_inline();
    let mut inputs = postcard::to_stdvec(&value).expect("serialize value");
    inputs.extend(postcard::to_stdvec(&readout_width).expect("serialize readout width"));
    let (_, trace, _, device) = program.trace(&inputs, &[], &[]);
    // Honest VirtualAssertEQ rows panic the tracer on unequal operands, so an
    // unequal row is spoil_proof's.
    let spoiled = trace.iter().any(|cycle| {
        matches!(cycle, Cycle::VirtualAssertEQ(row)
            if row.register_state.rs1 != row.register_state.rs2)
    });
    Run { device, spoiled }
}

/// `p + offset` as six limbs, from the field's encoding of −1.
fn modulus_plus(offset: i64) -> [u64; 6] {
    let mut limbs = canonical_limbs::<Field, 6>(Field::from_u64(0) - Field::from_u64(1))
        .expect("the proof field fits six limbs");
    let mut carry = i128::from(offset) + 1;
    for limb in &mut limbs {
        let sum = i128::from(*limb) + carry;
        *limb = sum as u64;
        carry = sum >> 64;
    }
    limbs
}

fn limb_at(index: usize) -> [u64; 6] {
    let mut limbs = [0; 6];
    limbs[index] = 1;
    limbs
}

#[test]
fn noncanonical_imports_panic_without_spoiling() {
    for value in [
        modulus_plus(0),
        modulus_plus(1),
        limb_at(MODULUS_LIMBS),
        limb_at(5),
        [u64::MAX; 6],
    ] {
        let run = trace(value, 6);
        assert!(run.device.panic, "importing {value:x?} must panic");
        assert!(!run.spoiled, "an import is not prover advice: {value:x?}");
    }
}

#[test]
fn narrow_readouts_reject_values_that_do_not_fit() {
    let fits = trace([5, 0, 0, 0, 0, 0], 1);
    assert!(!fits.device.panic);
    assert_eq!(fits.output(), [5, 0, 0, 0, 0, 0]);

    let wide = trace(limb_at(1), 1);
    assert!(wide.device.panic, "2^64 does not fit one limb");
    assert!(!wide.spoiled);
}

/// Limbs below p that do not encode the register value leave a nonzero
/// quotient, which the assertion right after the last limb rejects.
#[test]
fn wrong_limbs_leave_a_nonzero_residual() {
    let value = modulus_plus(-1);
    let mut low_limb_off = value;
    low_limb_off[0] -= 1;
    let mut truncated = value;
    truncated[MODULUS_LIMBS - 1] = 0;
    for limbs in [low_limb_off, truncated] {
        let injection = inject_advice_limbs(&limbs[..MODULUS_LIMBS]);
        let failure = catch_unwind(AssertUnwindSafe(|| trace(value, 6)))
            .err()
            .expect("tracing must stop at the residual assertion");
        let message = failure
            .downcast_ref::<String>()
            .expect("assertion panics carry a formatted message");
        let addresses = injection.consumed_addresses();
        assert_eq!(addresses.len(), MODULUS_LIMBS);
        // The assertion directly follows the last limb, and the tracer reports
        // the pc after the asserting instruction.
        let residual_pc = addresses[MODULUS_LIMBS - 1] + 8;
        assert!(
            message.contains(&format!(
                "FIELD_ASSERT_ZERO of nonzero field register 15 at pc 0x{residual_pc:x}"
            )),
            "{limbs:x?}: {message}"
        );
    }
}

/// Integers congruent to the register value but at least p satisfy every
/// field-inline row; only the guest's `< p` check rejects them.
#[test]
fn noncanonical_advice_spoils_the_proof() {
    let five = [5, 0, 0, 0, 0, 0];
    for (value, limbs) in [([0; 6], modulus_plus(0)), (five, modulus_plus(5))] {
        let injection = inject_advice_limbs(&limbs[..MODULUS_LIMBS]);
        let run = trace(value, 6);
        assert_eq!(injection.consumed_addresses().len(), MODULUS_LIMBS);
        assert!(run.spoiled, "{limbs:x?} must spoil the proof");
        assert!(run.device.panic);
    }
}
