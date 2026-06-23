use jolt_akita::AkitaField;
use jolt_field::Field;

#[test]
fn akita_field_satisfies_jolt_field_bundle() {
    fn assert_field<F: Field>() {}
    assert_field::<AkitaField>();
}
