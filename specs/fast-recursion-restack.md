# Fast-recursion port to current field-inline

Local restack of PR1911 (`329e42ef26d9e2c8c103dd8f61cdb60855555475`) from
`321c58d38ccbdfce90f190192c921840e02ae0ae` onto field-inline
`6991852388732446dadeca072e68f9d625217a3c`. No remote mutation.

The guest arithmetic adapter now imports instruction encoding vocabulary from
`jolt-riscv::FieldInlineOp`; it does not retain removed load/store encodings.
Memory ingress clears the destination with `LoadImm(0)` and applies the current
`LoadAccumulateFromMemory` equation `dst := 2^64 * dst + word` from most to least
significant limb. Clearing occurs on every operand load, including repeated use
of the same field register and Montgomery constants.

Readout performs N advice equations `source = limb + 2^64 * quotient`, alternating
quotients through existing scratch registers, followed by `AssertZero` on the
last quotient. The original source register remains untouched. Summing the
equations binds the returned integer modulo the field; existing field wrappers
still reject integers at or above the modulus. No canonicality check was removed.

The native packed prover's `commit_field_inc_limbs` unconditionally commits its
limb object, including all-zero content. Provisioning therefore retains the
current base's mandatory limb-group rule. The advice-combination regression now
checks exactly all four legal combinations including the limb group, rather
than the old recursion patch's seven active/inactive shapes. Catalog decoding
and identity tests remain present.

Conflict resolutions retain both independent Cargo features, the current CI
matrix job name plus the recursion timeout/acceptance steps, the required new
imports, current ISA documentation, and both independent typo dictionaries.
A partial-clone retrieval failure interrupted the formatting-only commit during
replay; its formatting was restored afterward (including exact original
recursion CLI source, which the new base did not modify).

Validation pending: no Cargo build or runtime test has run during this repair.
Run the focused field/packed schedule checks, both recursion host feature paths,
then rebuild the guest and run retained native acceptance/rejection controls.
A preexisting guest ELF is invalid evidence for this ISA port. Existing composed
field-inline and packed e2e gates must pass before publication. Old instruction
names are absent from Rust source; source diff whitespace checks passed.
