use crate::{JoltInstructionKind, JoltInstructionTag, SourceInstructionKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceExtension {
    Rv64I,
    Rv64M,
    Rv64A,
    Rv64C,
    Zicsr,
    RvPrivileged,
    JoltCustom,
    JoltInline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JoltTargetExtension {
    IntegerCore,
    IntegerMultiply,
    ControlFlow,
    LoadStore64,
    Advice,
    HostIO,
    VirtualAssertions,
    VirtualArithmetic,
    VirtualShifts,
    BitManipulation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InlineExtension {
    Sha2,
    Keccak256,
    Blake2,
    Blake3,
    BigInt256,
    Secp256k1,
    Grumpkin,
    P256,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ProfileInstructionIndex(pub u16);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JoltInstructionProfile {
    pub source_extensions: &'static [SourceExtension],
    pub inline_extensions: &'static [InlineExtension],
}

pub const RV64IM_JOLT: JoltInstructionProfile = JoltInstructionProfile {
    source_extensions: &[
        SourceExtension::Rv64I,
        SourceExtension::Rv64M,
        SourceExtension::Zicsr,
        SourceExtension::RvPrivileged,
        SourceExtension::JoltCustom,
        SourceExtension::JoltInline,
    ],
    inline_extensions: &[],
};

pub const RV64IMAC_JOLT: JoltInstructionProfile = JoltInstructionProfile {
    source_extensions: &[
        SourceExtension::Rv64I,
        SourceExtension::Rv64M,
        SourceExtension::Rv64A,
        SourceExtension::Rv64C,
        SourceExtension::Zicsr,
        SourceExtension::RvPrivileged,
        SourceExtension::JoltCustom,
        SourceExtension::JoltInline,
    ],
    inline_extensions: &[],
};

pub const RV64IMAC_JOLT_ALL_INLINES: JoltInstructionProfile = JoltInstructionProfile {
    source_extensions: RV64IMAC_JOLT.source_extensions,
    inline_extensions: &[
        InlineExtension::Sha2,
        InlineExtension::Keccak256,
        InlineExtension::Blake2,
        InlineExtension::Blake3,
        InlineExtension::BigInt256,
        InlineExtension::Secp256k1,
        InlineExtension::Grumpkin,
        InlineExtension::P256,
    ],
};

impl JoltInstructionProfile {
    pub fn supports_source(self, kind: SourceInstructionKind) -> bool {
        match source_extension(kind) {
            None => true,
            Some(extension) => self.source_extensions.contains(&extension),
        }
    }

    pub fn supports_jolt(self, kind: JoltInstructionKind) -> bool {
        match jolt_target_extension(kind) {
            None => matches!(kind, JoltInstructionKind::NoOp),
            Some(extension) => self.supports_target_extension(extension),
        }
    }

    pub fn supports_inline(self, extension: InlineExtension) -> bool {
        self.inline_extensions.contains(&extension)
    }

    pub fn source_dense_index(
        self,
        kind: SourceInstructionKind,
    ) -> Option<ProfileInstructionIndex> {
        dense_index(
            SourceInstructionKind::ALL.iter().copied(),
            |candidate| self.supports_source(candidate),
            kind,
        )
    }

    pub fn jolt_dense_index(self, kind: JoltInstructionKind) -> Option<ProfileInstructionIndex> {
        dense_index(
            JoltInstructionKind::ALL.iter().copied(),
            |candidate| self.supports_jolt(candidate),
            kind,
        )
    }

    pub fn fingerprint(self) -> u64 {
        let mut hash = hash_byte(FNV_OFFSET_BASIS, FINGERPRINT_SCHEMA_VERSION);
        for extension in self.source_extensions {
            hash = hash_byte(hash, 0x01);
            hash = hash_byte(hash, source_extension_code(*extension));
        }
        for extension in self.inline_extensions {
            hash = hash_byte(hash, 0x02);
            hash = hash_byte(hash, inline_extension_code(*extension));
        }
        for kind in SourceInstructionKind::ALL {
            if self.supports_source(*kind) {
                hash = hash_str(hash_byte(hash, 0x03), kind.canonical_name());
            }
        }
        for kind in JoltInstructionKind::ALL {
            if self.supports_jolt(*kind) {
                hash = hash_tag(hash_byte(hash, 0x04), kind.tag());
                hash = hash_str(hash_byte(hash, 0x05), kind.canonical_name());
            }
        }
        hash
    }

    fn supports_target_extension(self, extension: JoltTargetExtension) -> bool {
        match extension {
            JoltTargetExtension::IntegerCore
            | JoltTargetExtension::ControlFlow
            | JoltTargetExtension::LoadStore64
            | JoltTargetExtension::VirtualAssertions
            | JoltTargetExtension::VirtualArithmetic
            | JoltTargetExtension::VirtualShifts => {
                self.source_extensions.contains(&SourceExtension::Rv64I)
            }
            JoltTargetExtension::IntegerMultiply => {
                self.source_extensions.contains(&SourceExtension::Rv64M)
            }
            JoltTargetExtension::Advice => {
                self.source_extensions.contains(&SourceExtension::Rv64M)
                    || self.source_extensions.contains(&SourceExtension::Rv64A)
                    || self
                        .source_extensions
                        .contains(&SourceExtension::JoltCustom)
            }
            JoltTargetExtension::HostIO => self
                .source_extensions
                .contains(&SourceExtension::JoltCustom),
            JoltTargetExtension::BitManipulation => {
                self.source_extensions
                    .contains(&SourceExtension::JoltCustom)
                    || !self.inline_extensions.is_empty()
            }
        }
    }
}

pub const fn source_extension(kind: SourceInstructionKind) -> Option<SourceExtension> {
    match kind {
        SourceInstructionKind::NoOp | SourceInstructionKind::Unimpl => None,
        SourceInstructionKind::Inline => Some(SourceExtension::JoltInline),
        SourceInstructionKind::ADD
        | SourceInstructionKind::ADDI
        | SourceInstructionKind::ADDIW
        | SourceInstructionKind::ADDW
        | SourceInstructionKind::AND
        | SourceInstructionKind::ANDI
        | SourceInstructionKind::AUIPC
        | SourceInstructionKind::BEQ
        | SourceInstructionKind::BGE
        | SourceInstructionKind::BGEU
        | SourceInstructionKind::BLT
        | SourceInstructionKind::BLTU
        | SourceInstructionKind::BNE
        | SourceInstructionKind::EBREAK
        | SourceInstructionKind::ECALL
        | SourceInstructionKind::FENCE
        | SourceInstructionKind::JAL
        | SourceInstructionKind::JALR
        | SourceInstructionKind::LB
        | SourceInstructionKind::LBU
        | SourceInstructionKind::LD
        | SourceInstructionKind::LH
        | SourceInstructionKind::LHU
        | SourceInstructionKind::LUI
        | SourceInstructionKind::LW
        | SourceInstructionKind::LWU
        | SourceInstructionKind::OR
        | SourceInstructionKind::ORI
        | SourceInstructionKind::SB
        | SourceInstructionKind::SD
        | SourceInstructionKind::SH
        | SourceInstructionKind::SLL
        | SourceInstructionKind::SLLI
        | SourceInstructionKind::SLLIW
        | SourceInstructionKind::SLLW
        | SourceInstructionKind::SLT
        | SourceInstructionKind::SLTI
        | SourceInstructionKind::SLTIU
        | SourceInstructionKind::SLTU
        | SourceInstructionKind::SRA
        | SourceInstructionKind::SRAI
        | SourceInstructionKind::SRAIW
        | SourceInstructionKind::SRAW
        | SourceInstructionKind::SRL
        | SourceInstructionKind::SRLI
        | SourceInstructionKind::SRLIW
        | SourceInstructionKind::SRLW
        | SourceInstructionKind::SUB
        | SourceInstructionKind::SUBW
        | SourceInstructionKind::SW
        | SourceInstructionKind::XOR
        | SourceInstructionKind::XORI => Some(SourceExtension::Rv64I),
        SourceInstructionKind::MUL
        | SourceInstructionKind::MULH
        | SourceInstructionKind::MULHSU
        | SourceInstructionKind::MULHU
        | SourceInstructionKind::MULW
        | SourceInstructionKind::DIV
        | SourceInstructionKind::DIVU
        | SourceInstructionKind::DIVUW
        | SourceInstructionKind::DIVW
        | SourceInstructionKind::REM
        | SourceInstructionKind::REMU
        | SourceInstructionKind::REMUW
        | SourceInstructionKind::REMW => Some(SourceExtension::Rv64M),
        SourceInstructionKind::LRW
        | SourceInstructionKind::SCW
        | SourceInstructionKind::AMOSWAPW
        | SourceInstructionKind::AMOADDW
        | SourceInstructionKind::AMOANDW
        | SourceInstructionKind::AMOORW
        | SourceInstructionKind::AMOXORW
        | SourceInstructionKind::AMOMINW
        | SourceInstructionKind::AMOMAXW
        | SourceInstructionKind::AMOMINUW
        | SourceInstructionKind::AMOMAXUW
        | SourceInstructionKind::LRD
        | SourceInstructionKind::SCD
        | SourceInstructionKind::AMOSWAPD
        | SourceInstructionKind::AMOADDD
        | SourceInstructionKind::AMOANDD
        | SourceInstructionKind::AMOORD
        | SourceInstructionKind::AMOXORD
        | SourceInstructionKind::AMOMIND
        | SourceInstructionKind::AMOMAXD
        | SourceInstructionKind::AMOMINUD
        | SourceInstructionKind::AMOMAXUD => Some(SourceExtension::Rv64A),
        SourceInstructionKind::CSRRW | SourceInstructionKind::CSRRS => Some(SourceExtension::Zicsr),
        SourceInstructionKind::MRET => Some(SourceExtension::RvPrivileged),
        SourceInstructionKind::ANDN
        | SourceInstructionKind::AdviceLB
        | SourceInstructionKind::AdviceLD
        | SourceInstructionKind::AdviceLH
        | SourceInstructionKind::AdviceLW
        | SourceInstructionKind::VirtualAdvice
        | SourceInstructionKind::VirtualAdviceLen
        | SourceInstructionKind::VirtualAdviceLoad
        | SourceInstructionKind::VirtualAssertEQ
        | SourceInstructionKind::VirtualAssertHalfwordAlignment
        | SourceInstructionKind::VirtualAssertWordAlignment
        | SourceInstructionKind::VirtualAssertLTE
        | SourceInstructionKind::VirtualHostIO
        | SourceInstructionKind::VirtualAssertValidDiv0
        | SourceInstructionKind::VirtualAssertValidUnsignedRemainder
        | SourceInstructionKind::VirtualAssertMulUNoOverflow
        | SourceInstructionKind::VirtualChangeDivisor
        | SourceInstructionKind::VirtualChangeDivisorW
        | SourceInstructionKind::VirtualLW
        | SourceInstructionKind::VirtualSW
        | SourceInstructionKind::VirtualZeroExtendWord
        | SourceInstructionKind::VirtualSignExtendWord
        | SourceInstructionKind::VirtualPow2W
        | SourceInstructionKind::VirtualPow2IW
        | SourceInstructionKind::VirtualMovsign
        | SourceInstructionKind::VirtualMULI
        | SourceInstructionKind::VirtualPow2
        | SourceInstructionKind::VirtualPow2I
        | SourceInstructionKind::VirtualRev8W
        | SourceInstructionKind::VirtualROTRI
        | SourceInstructionKind::VirtualROTRIW
        | SourceInstructionKind::VirtualShiftRightBitmask
        | SourceInstructionKind::VirtualShiftRightBitmaskI
        | SourceInstructionKind::VirtualSRA
        | SourceInstructionKind::VirtualSRAI
        | SourceInstructionKind::VirtualSRL
        | SourceInstructionKind::VirtualSRLI
        | SourceInstructionKind::VirtualXORROT32
        | SourceInstructionKind::VirtualXORROT24
        | SourceInstructionKind::VirtualXORROT16
        | SourceInstructionKind::VirtualXORROT63
        | SourceInstructionKind::VirtualXORROTW16
        | SourceInstructionKind::VirtualXORROTW12
        | SourceInstructionKind::VirtualXORROTW8
        | SourceInstructionKind::VirtualXORROTW7 => Some(SourceExtension::JoltCustom),
    }
}

pub const fn jolt_target_extension(kind: JoltInstructionKind) -> Option<JoltTargetExtension> {
    match kind {
        JoltInstructionKind::NoOp => None,
        JoltInstructionKind::ADD
        | JoltInstructionKind::ADDI
        | JoltInstructionKind::AND
        | JoltInstructionKind::ANDI
        | JoltInstructionKind::AUIPC
        | JoltInstructionKind::LUI
        | JoltInstructionKind::OR
        | JoltInstructionKind::ORI
        | JoltInstructionKind::SLT
        | JoltInstructionKind::SLTI
        | JoltInstructionKind::SLTIU
        | JoltInstructionKind::SLTU
        | JoltInstructionKind::SUB
        | JoltInstructionKind::XOR
        | JoltInstructionKind::XORI => Some(JoltTargetExtension::IntegerCore),
        JoltInstructionKind::MUL | JoltInstructionKind::MULHU => {
            Some(JoltTargetExtension::IntegerMultiply)
        }
        JoltInstructionKind::BEQ
        | JoltInstructionKind::BGE
        | JoltInstructionKind::BGEU
        | JoltInstructionKind::BLT
        | JoltInstructionKind::BLTU
        | JoltInstructionKind::BNE
        | JoltInstructionKind::FENCE
        | JoltInstructionKind::JAL
        | JoltInstructionKind::JALR => Some(JoltTargetExtension::ControlFlow),
        JoltInstructionKind::LD | JoltInstructionKind::SD => Some(JoltTargetExtension::LoadStore64),
        JoltInstructionKind::VirtualAdvice
        | JoltInstructionKind::VirtualAdviceLen
        | JoltInstructionKind::VirtualAdviceLoad => Some(JoltTargetExtension::Advice),
        JoltInstructionKind::VirtualHostIO => Some(JoltTargetExtension::HostIO),
        JoltInstructionKind::VirtualAssertEQ
        | JoltInstructionKind::VirtualAssertHalfwordAlignment
        | JoltInstructionKind::VirtualAssertWordAlignment
        | JoltInstructionKind::VirtualAssertLTE
        | JoltInstructionKind::VirtualAssertValidDiv0
        | JoltInstructionKind::VirtualAssertValidUnsignedRemainder
        | JoltInstructionKind::VirtualAssertMulUNoOverflow => {
            Some(JoltTargetExtension::VirtualAssertions)
        }
        JoltInstructionKind::VirtualChangeDivisor
        | JoltInstructionKind::VirtualChangeDivisorW
        | JoltInstructionKind::VirtualZeroExtendWord
        | JoltInstructionKind::VirtualSignExtendWord
        | JoltInstructionKind::VirtualPow2W
        | JoltInstructionKind::VirtualPow2IW
        | JoltInstructionKind::VirtualMovsign
        | JoltInstructionKind::VirtualMULI
        | JoltInstructionKind::VirtualPow2
        | JoltInstructionKind::VirtualPow2I => Some(JoltTargetExtension::VirtualArithmetic),
        JoltInstructionKind::VirtualROTRI
        | JoltInstructionKind::VirtualROTRIW
        | JoltInstructionKind::VirtualShiftRightBitmask
        | JoltInstructionKind::VirtualShiftRightBitmaskI
        | JoltInstructionKind::VirtualSRA
        | JoltInstructionKind::VirtualSRAI
        | JoltInstructionKind::VirtualSRL
        | JoltInstructionKind::VirtualSRLI => Some(JoltTargetExtension::VirtualShifts),
        JoltInstructionKind::ANDN
        | JoltInstructionKind::VirtualRev8W
        | JoltInstructionKind::VirtualXORROT32
        | JoltInstructionKind::VirtualXORROT24
        | JoltInstructionKind::VirtualXORROT16
        | JoltInstructionKind::VirtualXORROT63
        | JoltInstructionKind::VirtualXORROTW16
        | JoltInstructionKind::VirtualXORROTW12
        | JoltInstructionKind::VirtualXORROTW8
        | JoltInstructionKind::VirtualXORROTW7 => Some(JoltTargetExtension::BitManipulation),
    }
}

fn dense_index<I, K>(
    kinds: I,
    mut is_supported: impl FnMut(K) -> bool,
    needle: K,
) -> Option<ProfileInstructionIndex>
where
    I: IntoIterator<Item = K>,
    K: Copy + Eq,
{
    let mut index = 0u16;
    for candidate in kinds {
        if !is_supported(candidate) {
            continue;
        }
        if candidate == needle {
            return Some(ProfileInstructionIndex(index));
        }
        index = index.checked_add(1)?;
    }
    None
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const FINGERPRINT_SCHEMA_VERSION: u8 = 1;

const fn hash_byte(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV_PRIME)
}

const fn hash_tag(hash: u64, tag: JoltInstructionTag) -> u64 {
    let bytes = tag.0.to_le_bytes();
    hash_byte(hash_byte(hash, bytes[0]), bytes[1])
}

fn hash_str(mut hash: u64, value: &str) -> u64 {
    for byte in value.as_bytes() {
        hash = hash_byte(hash, *byte);
    }
    hash
}

const fn source_extension_code(extension: SourceExtension) -> u8 {
    match extension {
        SourceExtension::Rv64I => 0,
        SourceExtension::Rv64M => 1,
        SourceExtension::Rv64A => 2,
        SourceExtension::Rv64C => 3,
        SourceExtension::Zicsr => 4,
        SourceExtension::RvPrivileged => 5,
        SourceExtension::JoltCustom => 6,
        SourceExtension::JoltInline => 7,
    }
}

const fn inline_extension_code(extension: InlineExtension) -> u8 {
    match extension {
        InlineExtension::Sha2 => 0,
        InlineExtension::Keccak256 => 1,
        InlineExtension::Blake2 => 2,
        InlineExtension::Blake3 => 3,
        InlineExtension::BigInt256 => 4,
        InlineExtension::Secp256k1 => 5,
        InlineExtension::Grumpkin => 6,
        InlineExtension::P256 => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profile_matches_current_source_shape() {
        assert!(RV64IMAC_JOLT.supports_source(SourceInstructionKind::ADD));
        assert!(RV64IMAC_JOLT.supports_source(SourceInstructionKind::ADDW));
        assert!(RV64IMAC_JOLT.supports_source(SourceInstructionKind::AMOADDW));
        assert!(RV64IMAC_JOLT.supports_source(SourceInstructionKind::Inline));
        assert!(!RV64IM_JOLT.supports_source(SourceInstructionKind::AMOADDW));
    }

    #[test]
    fn target_legality_rejects_source_only_rows() {
        assert!(RV64IMAC_JOLT.supports_jolt(JoltInstructionKind::ADD));
        assert!(RV64IMAC_JOLT.supports_jolt(JoltInstructionKind::LD));
        assert!(RV64IMAC_JOLT.supports_jolt(JoltInstructionKind::VirtualAssertEQ));

        assert_eq!(SourceInstructionKind::Unimpl.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::ADDW.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::LW.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::SW.jolt_kind(), None);
        assert_eq!(SourceInstructionKind::Inline.jolt_kind(), None);
    }

    #[test]
    fn dense_indexes_are_profile_local_and_contiguous() {
        assert_eq!(
            RV64IMAC_JOLT.source_dense_index(SourceInstructionKind::NoOp),
            Some(ProfileInstructionIndex(0))
        );
        assert_eq!(
            RV64IMAC_JOLT.jolt_dense_index(JoltInstructionKind::NoOp),
            Some(ProfileInstructionIndex(0))
        );
        assert_eq!(
            RV64IM_JOLT.source_dense_index(SourceInstructionKind::AMOADDW),
            None
        );
        assert_eq!(SourceInstructionKind::ADDW.jolt_kind(), None);

        let supported = JoltInstructionKind::ALL
            .iter()
            .copied()
            .filter(|kind| RV64IMAC_JOLT.supports_jolt(*kind))
            .collect::<Vec<_>>();
        for (expected, kind) in supported.into_iter().enumerate() {
            assert_eq!(
                RV64IMAC_JOLT.jolt_dense_index(kind),
                Some(ProfileInstructionIndex(expected as u16))
            );
        }
    }

    #[test]
    fn profile_fingerprint_changes_with_legality_sets() {
        assert_ne!(RV64IM_JOLT.fingerprint(), RV64IMAC_JOLT.fingerprint());
        assert_ne!(
            RV64IMAC_JOLT.fingerprint(),
            RV64IMAC_JOLT_ALL_INLINES.fingerprint()
        );
    }
}
