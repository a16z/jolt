//! The Jolt protocol's diagnostic transcript sites.
//!
//! Each protocol region sets its site on entry, on both the prover and the
//! verifier, so a `logging` build attributes every transcript event to a
//! region. Sites never reach the sponge.

use jolt_transcript::SiteId;

pub const PREAMBLE: SiteId = SiteId::label("jolt/preamble");
pub const COMMITMENTS: SiteId = SiteId::label("jolt/commitments");
pub const STAGE1: SiteId = SiteId::label("jolt/stage1");
pub const STAGE2: SiteId = SiteId::label("jolt/stage2");
pub const STAGE3: SiteId = SiteId::label("jolt/stage3");
pub const STAGE4: SiteId = SiteId::label("jolt/stage4");
pub const STAGE5: SiteId = SiteId::label("jolt/stage5");
pub const STAGE6A: SiteId = SiteId::label("jolt/stage6a");
pub const STAGE6B: SiteId = SiteId::label("jolt/stage6b");
pub const STAGE7: SiteId = SiteId::label("jolt/stage7");
pub const STAGE8: SiteId = SiteId::label("jolt/stage8");
pub const BLINDFOLD: SiteId = SiteId::label("jolt/blindfold");
