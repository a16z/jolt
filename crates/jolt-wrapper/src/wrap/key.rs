use std::collections::{BTreeMap, HashMap};

use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::HyperKZGProverSetup;
use jolt_r1cs::Variable;

use crate::hash_table::layout::MESSAGE;
use crate::hash_table::terms::{
    challenge125, challenge_scalar128, fr_word, fr_word_shifted, AffineForm as HashAffineForm,
    WIRED_BIT_BASE, WIRED_WORD_BASE,
};
use crate::hash_table::{
    ByteSource, Decoder, ElementKind as HashElementKind, HashTable, LinkMap, PublicInputs,
    T1Challenges, VkColumn,
};
use crate::limb_table::dory::{input_elements, ElementKind as LimbElementKind, InputElement};
use crate::limb_table::relation::Col as LimbCol;
use crate::limb_table::schedule::Layout as LimbTableLayout;
use crate::limb_table::stream::{
    commitment_phases as limb_commitment_phases, LimbTableKey, T2Challenges, PHASE_CHALLENGES,
};
use crate::links::{CopyLink, CopyLinkSide, CopyLinkTermSide, WIRES};
use crate::profile::WrapperProfile;
use crate::relation::{build_relation, Relation, ScheduleEntry, SqueezeKind};
use crate::stream::{
    commit_packed, AffineForm, AssemblyMemberStatement, AssemblyStatement, Column, ColumnId,
    Commitment, CommitmentPhase, StageMemberSpec,
};

use super::{
    hash_public_statement, CopyExporterPlan, CopyKey, DoryLinkPlacement, LeftLinkValue,
    LimbExporterPlan, LimbLinkValue, ScalarExporterPlan, WrapAssemblyPlan, WrapError, WrapHashKey,
    WrapLimbKey,
};

pub(super) struct KeyAssembly {
    pub(super) statement: AssemblyStatement,
    pub(super) limb: WrapLimbKey,
    pub(super) dory_link: DoryLinkPlacement,
    pub(super) plan: WrapAssemblyPlan,
    pub(super) copies: Vec<CopyKey>,
    pub(super) relation: Relation,
}

pub(super) fn build_key_assembly(
    profile: &WrapperProfile,
    hash: &WrapHashKey,
    hash_public: &PublicInputs,
    limb_table: LimbTableKey,
    mut public_inputs: Vec<Fr>,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<KeyAssembly, WrapError> {
    let packing = hash.table.packing();
    if limb_table.packing() != packing {
        return Err(WrapError::CommitmentPhases);
    }
    let rows = 1usize << hash.schedule().log_rows;
    let relation = build_relation(profile)?;
    if public_inputs.len() != relation.public.num_public {
        return Err(WrapError::StatementMismatch);
    }
    let copies = canonical_copy_keys(
        hash.links(),
        &relation,
        limb_table.layout(),
        &profile.commitment_link_order(),
        rows,
    )?;
    let hash_columns = hash.table.column_ids(0);
    let hash_groups = hash_columns
        .iter()
        .map(|column| column.group)
        .max()
        .ok_or(WrapError::CommitmentPhases)?
        + 1;
    let witness_base = hash_groups * packing;
    let mut copy_base = witness_base + packing;
    let copy_fixed_bases = copies
        .iter()
        .map(|_| {
            let base = copy_base;
            copy_base += (4 * WIRES).div_ceil(packing) * packing;
            base
        })
        .collect::<Vec<_>>();
    let phase_1a_groups = copy_base / packing;
    let t2_group_offset = phase_1a_groups;
    let t2_phases = limb_commitment_phases(packing);
    let t2_groups = t2_phases
        .iter()
        .map(|phase| phase.group_count)
        .sum::<usize>();
    let helper_count = 2 * copies.len();
    let helper_groups = helper_count.div_ceil(packing);
    let helper_base = (t2_group_offset + t2_groups) * packing;

    let t1_challenge_offset = 0;
    let theta_offset = t1_challenge_offset + T1Challenges::count(hash.schedule().log_rows);
    let t2_challenge_offset = theta_offset + 1;
    let copy_challenge_offset = t2_challenge_offset + PHASE_CHALLENGES[0];
    let rho_offset = copy_challenge_offset + 2 * copies.len();
    let r_stage_challenge_offset = rho_offset + 1 + T2Challenges::count() - PHASE_CHALLENGES[0];
    let weights_offset = r_stage_challenge_offset + copies.len() * hash.schedule().log_rows;
    let t2_member = 2 + copies.len();
    let dory_member = t2_member + 1;
    let carry_member = dory_member + 1;

    let mut commitment_phases = vec![CommitmentPhase {
        group_count: phase_1a_groups,
        challenge_count: theta_offset + 1,
    }];
    commitment_phases.extend(t2_phases.into_iter().enumerate().map(|(index, mut phase)| {
        if index == 0 {
            phase.challenge_count += 2 * copies.len() + 1;
        }
        phase
    }));
    let final_phase = commitment_phases
        .last_mut()
        .ok_or(WrapError::CommitmentPhases)?;
    final_phase.group_count += helper_groups;
    final_phase.challenge_count += copies.len() * (hash.schedule().log_rows + 3);
    let total_groups = commitment_phases
        .iter()
        .map(|phase| phase.group_count)
        .sum::<usize>();
    public_inputs.extend(hash_public_statement(hash_public));
    let members = std::iter::once(3)
        .chain(std::iter::once(3))
        .chain(std::iter::repeat_n(5, copies.len()))
        .chain([5, 2, 2])
        .map(|degree| AssemblyMemberStatement {
            input_claim: Fr::zero(),
            spec: StageMemberSpec {
                rounds: hash.schedule().log_rows,
                degree,
                offset: 0,
            },
        })
        .collect();

    let mut pinned_commitments = hash.pinned_commitments();
    for (copy, &base) in copies.iter().zip(&copy_fixed_bases) {
        pinned_commitments.extend(commit_key_columns(
            fixed_copy_columns(&copy.link),
            base / packing,
            packing,
            rows,
            setup,
        )?);
    }
    pinned_commitments.extend(limb_table.pinned_commitments(t2_group_offset));

    let witness_column = physical_id(witness_base, packing);
    let limb_columns = limb_table.column_ids(t2_group_offset);
    let copy_plans = copies
        .iter()
        .zip(&copy_fixed_bases)
        .enumerate()
        .map(|(index, (copy, &base))| {
            let left = CopyLinkTermSide {
                selectors: std::array::from_fn(|wire| physical_id(base + wire, packing)),
                ids: std::array::from_fn(|wire| {
                    column_form(physical_id(base + WIRES + wire, packing))
                }),
                values: copy.left.clone().map(|source| match source {
                    LeftLinkValue::Hash(form) => map_hash_form(&form, &hash_columns),
                    LeftLinkValue::Zero => zero_form(),
                }),
                helper: physical_id(helper_base + 2 * index, packing),
            };
            let right = CopyLinkTermSide {
                selectors: std::array::from_fn(|wire| {
                    physical_id(base + 2 * WIRES + wire, packing)
                }),
                ids: std::array::from_fn(|wire| {
                    column_form(physical_id(base + 3 * WIRES + wire, packing))
                }),
                values: copy
                    .right
                    .map(|source| limb_link_form(source, witness_column, &limb_columns)),
                helper: physical_id(helper_base + 2 * index + 1, packing),
            };
            CopyExporterPlan {
                link: copy.link.clone(),
                left,
                right,
                tau: r_stage_challenge_offset + index * hash.schedule().log_rows
                    ..r_stage_challenge_offset + (index + 1) * hash.schedule().log_rows,
                beta: copy_challenge_offset + 2 * index,
                gamma: copy_challenge_offset + 2 * index + 1,
                weights: weights_offset + 3 * index..weights_offset + 3 * (index + 1),
                member: 2 + index,
            }
        })
        .collect();
    let statement = AssemblyStatement {
        key_digest: hash.profile_digest,
        public_inputs,
        rows,
        column_count: total_groups * packing,
        k: packing,
        members,
        commitment_phases,
        pinned_commitments,
    };
    let plan = WrapAssemblyPlan {
        hash_columns,
        witness_column,
        carry_member,
        copies: copy_plans,
        limb: LimbExporterPlan {
            challenge_offset: t2_challenge_offset,
            theta_offset,
            rho_offset,
            columns: limb_columns,
            row_member: t2_member,
            link_member: dory_member,
        },
        scalar: ScalarExporterPlan {
            rows,
            positions: relation
                .link
                .dory
                .scalars
                .iter()
                .map(|(_, variable)| variable.index() - 1 - relation.public.num_public)
                .collect(),
            rho_offset,
            wire: witness_column,
            member: dory_member,
        },
        max_factors: 4,
    };
    Ok(KeyAssembly {
        statement,
        limb: WrapLimbKey::new(limb_table),
        dory_link: DoryLinkPlacement {
            challenge: rho_offset,
            theta: theta_offset,
            member: dory_member,
        },
        plan,
        copies,
        relation,
    })
}

fn commit_key_columns(
    mut columns: Vec<Column>,
    group_offset: usize,
    packing: usize,
    rows: usize,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<Vec<(usize, Commitment)>, WrapError> {
    while !columns.len().is_multiple_of(packing) {
        columns.push(Column::Bits(vec![0; rows]));
    }
    let committed = commit_packed(&columns, packing, setup)?;
    Ok(committed
        .commitments
        .into_iter()
        .enumerate()
        .map(|(index, commitment)| (group_offset + index, commitment))
        .collect())
}

fn physical_id(index: usize, packing: usize) -> ColumnId {
    ColumnId {
        group: index / packing,
        slot: index % packing,
    }
}

fn column_form(column: ColumnId) -> AffineForm {
    AffineForm {
        constant: Fr::zero(),
        weights: vec![(column, Fr::one())],
    }
}

fn zero_form() -> AffineForm {
    AffineForm {
        constant: Fr::zero(),
        weights: Vec::new(),
    }
}

fn map_hash_form(form: &HashAffineForm, columns: &[ColumnId]) -> AffineForm {
    AffineForm {
        constant: form.constant,
        weights: form
            .weights
            .iter()
            .map(|&(column, weight)| (columns[column], weight))
            .collect(),
    }
}

fn limb_link_form(source: LimbLinkValue, witness: ColumnId, limb: &[ColumnId]) -> AffineForm {
    match source {
        LimbLinkValue::Witness => column_form(witness),
        LimbLinkValue::Chunk(chunk) => column_form(limb[LimbCol::CHUNKS + chunk]),
        LimbLinkValue::Sign => column_form(limb[LimbCol::FLAG]),
        LimbLinkValue::Zero => zero_form(),
    }
}

fn fixed_copy_columns(link: &CopyLink) -> Vec<Column> {
    link.left
        .selectors
        .iter()
        .chain(&link.left.ids)
        .chain(&link.right.selectors)
        .chain(&link.right.ids)
        .cloned()
        .map(Column::Fr)
        .collect()
}

fn canonical_copy_keys(
    links: &LinkMap,
    relation: &Relation,
    limb: &LimbTableLayout,
    commitment_order: &[usize],
    rows: usize,
) -> Result<Vec<CopyKey>, WrapError> {
    let mut copies = vec![
        challenge_copy_key(links, relation, rows)?,
        absorbed_word_copy_key(links, relation, rows)?,
    ];
    copies.extend(element_copy_keys(links, limb, commitment_order, rows)?);
    Ok(copies)
}

fn challenge_copy_key(
    links: &LinkMap,
    relation: &Relation,
    rows: usize,
) -> Result<CopyKey, WrapError> {
    let relation_squeezes = relation
        .link
        .schedule
        .iter()
        .filter_map(|entry| match entry {
            ScheduleEntry::Squeeze { kind, var } => Some((*kind, *var)),
            ScheduleEntry::Bytes(_) | ScheduleEntry::Fr(_) | ScheduleEntry::Opaque { .. } => None,
        })
        .collect::<Vec<_>>();
    if links.challenges.len() != relation_squeezes.len() {
        return Err(WrapError::T1MemberLayout);
    }
    let mut left_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut left_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_slots = HashMap::new();
    for (index, ((squeeze, left_row), (kind, variable))) in
        links.challenges.iter().zip(relation_squeezes).enumerate()
    {
        let wire = match (squeeze.decoder, kind) {
            (Decoder::Challenge125, SqueezeKind::Challenge) => 0,
            (Decoder::Scalar128, SqueezeKind::Scalar) => 1,
            _ => return Err(WrapError::T1MemberLayout),
        };
        let id = Fr::from_u64(index as u64 + 1);
        left_selectors[wire][*left_row] = Fr::one();
        left_ids[wire][*left_row] = id;
        let right_row = witness_row(relation, variable)?;
        let right_wire = right_slots.entry(right_row).or_insert(0usize);
        if *right_wire >= WIRES {
            return Err(WrapError::T1MemberLayout);
        }
        right_selectors[*right_wire][right_row] = Fr::one();
        right_ids[*right_wire][right_row] = id;
        *right_wire += 1;
    }
    Ok(CopyKey {
        link: CopyLink::new(
            CopyLinkSide::new(left_selectors, left_ids)?,
            CopyLinkSide::new(right_selectors, right_ids)?,
        )?,
        left: [
            LeftLinkValue::Hash(challenge125()),
            LeftLinkValue::Hash(challenge_scalar128()),
            LeftLinkValue::Zero,
        ],
        right: [
            LimbLinkValue::Witness,
            LimbLinkValue::Witness,
            LimbLinkValue::Witness,
        ],
    })
}

fn absorbed_word_copy_key(
    links: &LinkMap,
    relation: &Relation,
    rows: usize,
) -> Result<CopyKey, WrapError> {
    let absorbed = relation
        .link
        .schedule
        .iter()
        .filter_map(|entry| match entry {
            ScheduleEntry::Fr(variable) => Some(*variable),
            ScheduleEntry::Bytes(_)
            | ScheduleEntry::Opaque { .. }
            | ScheduleEntry::Squeeze { .. } => None,
        })
        .collect::<Vec<_>>();
    let mut left_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut left_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
    let mut right_slots = HashMap::new();
    for &(index, row) in &links.wires {
        let id = Fr::from_u64(u64::from(index) + 1);
        left_selectors[0][row] = Fr::one();
        left_ids[0][row] = id;
        let variable = *absorbed
            .get(index as usize)
            .ok_or(WrapError::T1MemberLayout)?;
        let right_row = witness_row(relation, variable)?;
        let right_wire = right_slots.entry(right_row).or_insert(0usize);
        if *right_wire >= WIRES {
            return Err(WrapError::T1MemberLayout);
        }
        right_selectors[*right_wire][right_row] = Fr::one();
        right_ids[*right_wire][right_row] = id;
        *right_wire += 1;
    }
    for &(index, row) in &links.wires_shifted {
        let id = Fr::from_u64(u64::from(index) + 1);
        left_selectors[1][row] = Fr::one();
        left_ids[1][row] = id;
        let variable = *absorbed
            .get(index as usize)
            .ok_or(WrapError::T1MemberLayout)?;
        let right_row = witness_row(relation, variable)?;
        let right_wire = right_slots.entry(right_row).or_insert(0usize);
        if *right_wire >= WIRES {
            return Err(WrapError::T1MemberLayout);
        }
        right_selectors[*right_wire][right_row] = Fr::one();
        right_ids[*right_wire][right_row] = id;
        *right_wire += 1;
    }
    Ok(CopyKey {
        link: CopyLink::new(
            CopyLinkSide::new(left_selectors, left_ids)?,
            CopyLinkSide::new(right_selectors, right_ids)?,
        )?,
        left: [
            LeftLinkValue::Hash(fr_word()),
            LeftLinkValue::Hash(fr_word_shifted()),
            LeftLinkValue::Zero,
        ],
        right: [
            LimbLinkValue::Witness,
            LimbLinkValue::Witness,
            LimbLinkValue::Witness,
        ],
    })
}

fn witness_row(relation: &Relation, variable: Variable) -> Result<usize, WrapError> {
    variable
        .index()
        .checked_sub(1 + relation.public.num_public)
        .ok_or(WrapError::T1MemberLayout)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum HashLinkValue {
    Half {
        half: usize,
        swapped: bool,
        bits: usize,
    },
    Bit(usize),
}

impl HashLinkValue {
    fn form(self) -> HashAffineForm {
        match self {
            Self::Half {
                half,
                swapped,
                bits,
            } => hash_halfword(half, swapped, bits),
            Self::Bit(bit) => hash_bit(bit),
        }
    }
}

#[derive(Clone, Copy)]
struct LinkEdge {
    left_row: usize,
    right_row: usize,
    left: HashLinkValue,
    right: LimbLinkValue,
    id: u64,
}

struct ElementTarget {
    rows: Vec<usize>,
    sign_row: Option<usize>,
}

fn element_copy_keys(
    links: &LinkMap,
    layout: &LimbTableLayout,
    commitment_order: &[usize],
    rows: usize,
) -> Result<Vec<CopyKey>, WrapError> {
    let targets = element_targets(layout, commitment_order)?;
    let mut positions = HashMap::new();
    for &(source, row, byte_in_word) in &links.bytes {
        if let ByteSource::Element { kind, index, byte } = source {
            if positions
                .insert((kind, index, byte), (row, usize::from(byte_in_word)))
                .is_some()
            {
                return Err(WrapError::T1MemberLayout);
            }
        }
    }
    let mut edges = Vec::new();
    let mut next_id = 1u64;
    for (&(kind, index), target) in &targets {
        let bytes = match kind {
            HashElementKind::CommitmentGt | HashElementKind::DoryGt => 384,
            HashElementKind::DoryG1 => 32,
            HashElementKind::DoryG2 => 64,
        };
        for byte in (0..bytes).step_by(2) {
            let &(left_row, first) = positions
                .get(&(kind, index, byte as u16))
                .ok_or(WrapError::T1MemberLayout)?;
            let &(second_row, second) = positions
                .get(&(kind, index, byte as u16 + 1))
                .ok_or(WrapError::T1MemberLayout)?;
            if left_row != second_row || second != first + 1 || !matches!(first, 0 | 2) {
                return Err(WrapError::T1MemberLayout);
            }
            let source_half = byte / 2;
            let (target_half, swapped) = if kind == HashElementKind::CommitmentGt {
                (bytes / 2 - 1 - source_half, true)
            } else {
                (source_half, false)
            };
            let coordinate = target_half / 16;
            let chunk = target_half % 16;
            let right_row = *target
                .rows
                .get(coordinate)
                .ok_or(WrapError::T1MemberLayout)?;
            let is_curve_top = matches!(kind, HashElementKind::DoryG1 | HashElementKind::DoryG2)
                && source_half == bytes / 2 - 1;
            edges.push(LinkEdge {
                left_row,
                right_row,
                left: HashLinkValue::Half {
                    half: first / 2,
                    swapped,
                    bits: if is_curve_top { 14 } else { 16 },
                },
                right: LimbLinkValue::Chunk(chunk),
                id: next_id,
            });
            next_id += 1;
            if is_curve_top {
                let sign_row = target.sign_row.ok_or(WrapError::T1MemberLayout)?;
                edges.push(LinkEdge {
                    left_row,
                    right_row: sign_row,
                    left: HashLinkValue::Bit(8 * second + 7),
                    right: LimbLinkValue::Sign,
                    id: next_id,
                });
                next_id += 1;
                edges.push(LinkEdge {
                    left_row,
                    right_row: sign_row,
                    left: HashLinkValue::Bit(8 * second + 6),
                    right: LimbLinkValue::Zero,
                    id: next_id,
                });
                next_id += 1;
            }
        }
    }
    let mut groups: Vec<(Vec<HashLinkValue>, Vec<LimbLinkValue>, Vec<LinkEdge>)> = Vec::new();
    for edge in edges {
        let group = groups.iter().position(|(left, right, _)| {
            (left.contains(&edge.left) || left.len() < WIRES)
                && (right.contains(&edge.right) || right.len() < WIRES)
        });
        let index = group.unwrap_or_else(|| {
            groups.push((Vec::new(), Vec::new(), Vec::new()));
            groups.len() - 1
        });
        let group = &mut groups[index];
        if !group.0.contains(&edge.left) {
            group.0.push(edge.left);
        }
        if !group.1.contains(&edge.right) {
            group.1.push(edge.right);
        }
        group.2.push(edge);
    }
    groups
        .into_iter()
        .map(|(mut left_keys, mut right_keys, edges)| {
            left_keys.resize(WIRES, HashLinkValue::Bit(0));
            right_keys.resize(WIRES, LimbLinkValue::Zero);
            let mut left_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            let mut left_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            let mut right_selectors = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            let mut right_ids = std::array::from_fn(|_| vec![Fr::zero(); rows]);
            for edge in edges {
                let left = left_keys
                    .iter()
                    .position(|key| *key == edge.left)
                    .ok_or(WrapError::T1MemberLayout)?;
                let right = right_keys
                    .iter()
                    .position(|key| *key == edge.right)
                    .ok_or(WrapError::T1MemberLayout)?;
                let id = Fr::from_u64(edge.id);
                left_selectors[left][edge.left_row] = Fr::one();
                left_ids[left][edge.left_row] = id;
                right_selectors[right][edge.right_row] = Fr::one();
                right_ids[right][edge.right_row] = id;
            }
            let left: [HashLinkValue; WIRES] = left_keys
                .try_into()
                .map_err(|_| WrapError::T1MemberLayout)?;
            let right: [LimbLinkValue; WIRES] = right_keys
                .try_into()
                .map_err(|_| WrapError::T1MemberLayout)?;
            Ok(CopyKey {
                link: CopyLink::new(
                    CopyLinkSide::new(left_selectors, left_ids)?,
                    CopyLinkSide::new(right_selectors, right_ids)?,
                )?,
                left: left.map(|value| LeftLinkValue::Hash(value.form())),
                right,
            })
        })
        .collect()
}

fn element_targets(
    layout: &LimbTableLayout,
    commitment_order: &[usize],
) -> Result<BTreeMap<(HashElementKind, u32), ElementTarget>, WrapError> {
    let mut by_element = HashMap::new();
    let mut cursor = 0;
    for &element in &layout.input_order {
        let coordinates = element.kind().coords();
        let end = cursor + coordinates;
        let rows = layout
            .program
            .input_rows
            .get(cursor..end)
            .ok_or(WrapError::T1MemberLayout)?
            .iter()
            .map(|row| *row as usize)
            .collect();
        cursor = end;
        let sign_row = layout
            .sign_rows
            .iter()
            .find_map(|(candidate, row)| (*candidate == element).then_some(*row as usize));
        if by_element
            .insert(element, ElementTarget { rows, sign_row })
            .is_some()
        {
            return Err(WrapError::T1MemberLayout);
        }
    }
    let mut targets = BTreeMap::new();
    let mut dory = [0u32; 3];
    for element in input_elements(layout.check.sigma, layout.check.n) {
        let key = if let InputElement::Commitment(index) = element {
            let transcript_index = *commitment_order
                .get(index)
                .ok_or(WrapError::T1MemberLayout)?;
            (HashElementKind::CommitmentGt, transcript_index as u32)
        } else {
            let slot = match element.kind() {
                LimbElementKind::Gt => 0,
                LimbElementKind::G1 => 1,
                LimbElementKind::G2 => 2,
            };
            let kind = [
                HashElementKind::DoryGt,
                HashElementKind::DoryG1,
                HashElementKind::DoryG2,
            ][slot];
            let key = (kind, dory[slot]);
            dory[slot] += 1;
            key
        };
        let target = by_element
            .remove(&element)
            .ok_or(WrapError::T1MemberLayout)?;
        if targets.insert(key, target).is_some() {
            return Err(WrapError::T1MemberLayout);
        }
    }
    if cursor != layout.program.input_rows.len() || !by_element.is_empty() {
        return Err(WrapError::T1MemberLayout);
    }
    Ok(targets)
}

fn hash_halfword(half: usize, swapped: bool, bits: usize) -> HashAffineForm {
    let mut form = HashAffineForm::default();
    for output in 0..bits {
        let input = if swapped { output ^ 8 } else { output };
        form.add(MESSAGE + 16 * half + input, Fr::from_u64(1 << output));
    }
    form
}

fn hash_bit(bit: usize) -> HashAffineForm {
    let mut form = HashAffineForm::default();
    form.add(MESSAGE + bit, Fr::one());
    form
}

pub(super) fn materialize_hash_form(form: &HashAffineForm, table: &HashTable) -> Vec<Fr> {
    let mut values = vec![form.constant; table.rows()];
    for &(column, weight) in &form.weights {
        for (row, value) in values.iter_mut().enumerate() {
            *value += weight * hash_column_value(table, column, row);
        }
    }
    values
}

fn hash_column_value(table: &HashTable, column: usize, row: usize) -> Fr {
    if column < WIRED_BIT_BASE {
        Fr::from_u64(u64::from(table.bits[column][row]))
    } else if column < WIRED_WORD_BASE {
        Fr::from_u64(u64::from(table.wired_bits[column - WIRED_BIT_BASE][row]))
    } else if column < WIRED_WORD_BASE + table.wired_words.len() {
        Fr::from_u64(u64::from(table.wired_words[column - WIRED_WORD_BASE][row]))
    } else {
        let vk = VkColumn::ALL[column - WIRED_WORD_BASE - table.wired_words.len()];
        Fr::from_u64(u64::from(table.vk.value(vk, row)))
    }
}
