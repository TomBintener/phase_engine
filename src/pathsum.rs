//! Foreign Function Interface (FFI) Bridge for Egglog Integration.
//!
//! This module connects the high-performance Rust implementation of the path sum
//! (`EvaluatedPathSum`) to the `egglog` equality saturation engine. It defines a
//! custom `egglog` Sort called `PathSum` and registers primitives that map
//! `egglog` function calls (e.g., `rust_apply_cx_ffi`) directly to the
//! corresponding Rust methods.
//!
//! # FFI Panic Shields
//! Since `egglog` explores many possible circuit rewrites simultaneously, it might
//! temporarily generate invalid gate applications (e.g., applying a gate to an
//! out-of-bounds qubit). The functions in this bridge include "panic shields"
//! that gracefully catch these invalid operations and return the unmodified state
//! rather than crashing the Rust runtime.
//!
//! # Eager Reduction
//! Every gate application logic function eagerly calls `reduce()` on the state
//! after applying the gate. This enforces that the path sum is maintained in its
//! canonical form at all times within the e-graph, which is necessary for
//! the engine to correctly identify equivalent operators.
//!
//! # Linear reversible synthesis
//! `rust_synthesize_pmh_*` calls `synthesize_cnot_matrix` on the matching
//! width: Patel–Markov–Hayes sectioned elimination (quant-ph/0302002), with
//! Gauss–Jordan as a fallback. Gray/Steiner residuals use the same helper.

use crate::ast::Literal;
use crate::engine::engine_128::EvaluatedPathSum as EvaluatedPathSum128;
use crate::engine::engine_64::EvaluatedPathSum as EvaluatedPathSum64;
use crate::engine::{engine_128, engine_64};
use crate::prelude::BaseSort;
use crate::sort::{BaseValues, Boxed};
use crate::{add_primitive, EGraph, Value};
use crate::{TermDag, TermId};
use std::sync::Arc;

// Define type aliases for the 64-bit and 128-bit engines.
//
// The interned state is Arc-wrapped: `Arc<T>` delegates `Hash`/`Eq`/`Debug`
// to `T`, so interning semantics and canonical-form equality are bit-identical
// to the plain `Boxed<EvaluatedPathSum>`, while `InternTable::get_cloned` (one
// deep clone per FFI argument) and the intern-miss store both become refcount
// bumps. Mutating wrappers pay exactly one content copy via
// `Arc::unwrap_or_clone`; read-only wrappers pay zero.
pub type PSum64 = Boxed<Arc<engine_64::EvaluatedPathSum>>;
pub type PSum128 = Boxed<Arc<engine_128::EvaluatedPathSum>>;

// --- Logic for the 64-bit Engine ---

pub fn id_pathsum_logic_64(num_qubits: i64) -> PSum64 {
    if num_qubits <= 0 {
        PSum64::new(Arc::new(engine_64::EvaluatedPathSum::new_id(0)))
    } else {
        PSum64::new(Arc::new(engine_64::EvaluatedPathSum::new_id(
            num_qubits as u32,
        )))
    }
}

pub fn apply_gate_logic_64<F>(state: PSum64, q: i64, op: F) -> PSum64
where
    F: Fn(&mut engine_64::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    // One content copy exactly where the gate needs an owned state: moves the
    // value out when this Arc is unique, clones otherwise.
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    op(&mut new_state, q as usize);
    new_state.reduce();
    PSum64::new(Arc::new(new_state))
}

pub fn apply_gate_no_reduce_logic_64<F>(state: PSum64, q: i64, op: F) -> PSum64
where
    F: Fn(&mut engine_64::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    op(&mut new_state, q as usize);
    // Affine updates (X) skip `reduce()` but may leave a constant on a row
    // that holds a path variable; restore the canonical gauge.
    new_state.canonicalize_gauge_after_row_op();
    PSum64::new(Arc::new(new_state))
}

pub fn apply_cx_logic_64(state: PSum64, qc: i64, qt: i64) -> PSum64 {
    if qc == qt
        || qc < 0
        || qt < 0
        || qc as usize >= state.num_qubits as usize
        || qt as usize >= state.num_qubits as usize
    {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    new_state.apply_cx(qc as usize, qt as usize);
    // CX is a row operation and needs no `reduce()`, but it can move a path
    // variable into another row; restore the canonical gauge (no-op without
    // path variables).
    new_state.canonicalize_gauge_after_row_op();
    PSum64::new(Arc::new(new_state))
}

pub fn apply_rz_logic_64(state: PSum64, q: i64, theta_bits: i64) -> PSum64 {
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    new_state.apply_rz(q as usize, f64::from_bits(theta_bits as u64));
    new_state.reduce();
    PSum64::new(Arc::new(new_state))
}

const SNAP_PRECISION: f64 = 100_000_000.0;
#[inline(always)]
fn snap_phase(val: f64) -> f64 {
    (val * SNAP_PRECISION).round() / SNAP_PRECISION
}

pub fn rust_add_rz_bits_logic(a: i64, b: i64) -> i64 {
    let f_a = f64::from_bits(a as u64);
    let f_b = f64::from_bits(b as u64);
    let sum = (f_a + f_b) % (2.0 * std::f64::consts::PI);
    let bounded = if sum < 0.0 {
        sum + 2.0 * std::f64::consts::PI
    } else {
        sum
    };
    let snapped = snap_phase(bounded);
    snapped.to_bits() as i64
}

pub fn rust_negate_rz_bits_logic(a: i64) -> i64 {
    let f_a = f64::from_bits(a as u64);
    let neg = -f_a;
    let bounded = if neg < 0.0 {
        neg + 2.0 * std::f64::consts::PI
    } else {
        neg
    };
    let snapped = snap_phase(bounded);
    snapped.to_bits() as i64
}

// --- Logic for the 128-bit Engine ---

pub fn id_pathsum_logic_128(num_qubits: i64) -> PSum128 {
    if num_qubits <= 0 {
        PSum128::new(Arc::new(engine_128::EvaluatedPathSum::new_id(0)))
    } else {
        PSum128::new(Arc::new(engine_128::EvaluatedPathSum::new_id(
            num_qubits as u32,
        )))
    }
}

pub fn apply_gate_logic_128<F>(state: PSum128, q: i64, op: F) -> PSum128
where
    F: Fn(&mut engine_128::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    // One content copy exactly where the gate needs an owned state: moves the
    // value out when this Arc is unique, clones otherwise.
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    op(&mut new_state, q as usize);
    new_state.reduce();
    PSum128::new(Arc::new(new_state))
}

pub fn apply_gate_no_reduce_logic_128<F>(state: PSum128, q: i64, op: F) -> PSum128
where
    F: Fn(&mut engine_128::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    op(&mut new_state, q as usize);
    // Affine updates (X) skip `reduce()` but may leave a constant on a row
    // that holds a path variable; restore the canonical gauge.
    new_state.canonicalize_gauge_after_row_op();
    PSum128::new(Arc::new(new_state))
}

pub fn apply_cx_logic_128(state: PSum128, qc: i64, qt: i64) -> PSum128 {
    if qc == qt
        || qc < 0
        || qt < 0
        || qc as usize >= state.num_qubits as usize
        || qt as usize >= state.num_qubits as usize
    {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    new_state.apply_cx(qc as usize, qt as usize);
    // CX is a row operation and needs no `reduce()`, but it can move a path
    // variable into another row; restore the canonical gauge (no-op without
    // path variables).
    new_state.canonicalize_gauge_after_row_op();
    PSum128::new(Arc::new(new_state))
}

pub fn apply_rz_logic_128(state: PSum128, q: i64, theta_bits: i64) -> PSum128 {
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    new_state.apply_rz(q as usize, f64::from_bits(theta_bits as u64));
    new_state.reduce();
    PSum128::new(Arc::new(new_state))
}

fn rank_m_minus_i_64(state: &EvaluatedPathSum64) -> i64 {
    let mut m = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        let mut row = state.out_state[i].variable_mask;
        row ^= 1 << i;
        m.push(row);
    }
    let mut rank = 0;
    for c in 0..state.num_qubits as usize {
        let mut pivot = rank;
        while pivot < state.num_qubits as usize && (m[pivot] & (1 << c)) == 0 {
            pivot += 1;
        }
        if pivot == state.num_qubits as usize {
            continue;
        }
        m.swap(rank, pivot);
        for r in 0..state.num_qubits as usize {
            if r != rank && (m[r] & (1 << c)) != 0 {
                m[r] ^= m[rank];
            }
        }
        rank += 1;
    }
    rank as i64
}

fn rank_m_minus_i_128(state: &EvaluatedPathSum128) -> i64 {
    let mut m = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        let mut row = state.out_state[i].variable_mask;
        row ^= 1 << i;
        m.push(row);
    }
    let mut rank = 0;
    for c in 0..state.num_qubits as usize {
        let mut pivot = rank;
        while pivot < state.num_qubits as usize && (m[pivot] & (1 << c)) == 0 {
            pivot += 1;
        }
        if pivot == state.num_qubits as usize {
            continue;
        }
        m.swap(rank, pivot);
        for r in 0..state.num_qubits as usize {
            if r != rank && (m[r] & (1 << c)) != 0 {
                m[r] ^= m[rank];
            }
        }
        rank += 1;
    }
    rank as i64
}
/// Continuous parities must be linear XORs of *input* qubits. Path-variable
/// bits or product monomials cannot be fed to Steiner/Gray: `variable_mask`
/// is the OR of monomials, so a live path var would be silently dropped and
/// the angle attached to leftover qubit bits.
fn continuous_parities_are_qubit_linear_64(state: &EvaluatedPathSum64) -> bool {
    let n = state.num_qubits as usize;
    let valid_mask: u64 = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    for &p in &state.continuous_poly.parities {
        if p == 0 || (p & !valid_mask) != 0 {
            return false;
        }
    }
    true
}

fn continuous_parities_are_qubit_linear_128(state: &EvaluatedPathSum128) -> bool {
    let n = state.num_qubits as usize;
    let valid_mask: u128 = if n >= 128 { u128::MAX } else { (1u128 << n) - 1 };
    for &p in &state.continuous_poly.parities {
        if p == 0 || (p & !valid_mask) != 0 {
            return false;
        }
    }
    true
}

/// Qubits whose `out_state` carries a constant-1 term (net X / affine offset).
/// Linear synthesizers only see `variable_mask`; callers append a trailing X layer.
fn affine_x_offsets_64(state: &PSum64) -> Vec<usize> {
    state
        .out_state
        .iter()
        .enumerate()
        .filter(|(_, poly)| poly.terms.contains(&0))
        .map(|(q, _)| q)
        .collect()
}

fn affine_x_offsets_128(state: &PSum128) -> Vec<usize> {
    state
        .out_state
        .iter()
        .enumerate()
        .filter(|(_, poly)| poly.terms.contains(&0))
        .map(|(q, _)| q)
        .collect()
}

fn append_x_layer(instructions: &mut Vec<String>, offsets: &[usize]) {
    for &q in offsets {
        instructions.push(format!("x {}", q));
    }
}

/// Normalize a phase angle into `(0, 2π)` like GraySynth so Mobius-derived
/// (possibly negative) coefficients emit IEEE bit patterns that fit in `i64`
/// and round-trip through `apply_rz` / the Python injector.
fn normalized_rz_bits(angle: f64) -> Option<u64> {
    let tau = std::f64::consts::TAU;
    let mut a = angle % tau;
    if a < 0.0 {
        a += tau;
    }
    if a.abs() < 1e-12 || (tau - a).abs() < 1e-12 {
        None
    } else {
        Some(a.to_bits())
    }
}

pub fn synthesize_steiner_logic_64(
    state: PSum64,
    gate_count: i64,
    hw_cost: i64,
    topology_str: String,
    cnot_weight: i64,
    rz_weight: i64,
) -> String {
    if gate_count <= 2 {
        return "None".to_string();
    }
    if state.is_overflowed() {
        return "None".to_string();
    }
    if !continuous_parities_are_qubit_linear_64(&state) {
        return "None".to_string();
    }
    let valid_mask = (1_u64 << state.num_qubits) - 1;
    for poly in &state.out_state {
        if (poly.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
    }
    // Affine/X offsets are synthesizable: linear CNOT/RZ block + trailing X layer.
    let x_offsets = affine_x_offsets_64(&state);

    let topo = crate::engine::engine_64::Topology::new(state.num_qubits as usize, &topology_str);
    let mut instructions = Vec::new();
    // Discrete phase poly is monomial-basis; lift to parities via Mobius
    // (same as Gray) before building per-term Steiner trees.
    let mut parities: Vec<(u64, f64)> = Vec::new();
    let mut monomials: Vec<(u64, f64)> = Vec::new();
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 61;
        let mask = term.0 & 0x1FFF_FFFF_FFFF_FFFF;
        if (mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        monomials.push((mask, (phase_unit as f64) * std::f64::consts::PI / 4.0));
    }
    match crate::engine::engine_64::phase_monomials_to_parities(&monomials) {
        Some(disc) => parities.extend(disc),
        None => return "None".to_string(),
    }
    for (i, &mask) in state.continuous_poly.parities.iter().enumerate() {
        let angle = crate::engine::ticks_to_angle(state.continuous_poly.phases[i]);
        parities.push((mask, angle));
    }
    // Merge duplicate parity masks (Mobius + continuous can collide).
    {
        use std::collections::HashMap;
        let mut merged: HashMap<u64, f64> = HashMap::new();
        for (mask, angle) in parities.drain(..) {
            if mask != 0 {
                *merged.entry(mask).or_insert(0.0) += angle;
            }
        }
        parities = merged.into_iter().collect();
    }

    let mut total_cnots = 0;
    for (mask, angle) in parities {
        let Some(rz_bits) = normalized_rz_bits(angle) else {
            continue;
        };
        let mut terminals = Vec::new();
        for i in 0..state.num_qubits {
            if (mask & (1 << i)) != 0 {
                terminals.push(i as usize);
            }
        }

        if terminals.is_empty() {
            continue;
        } else if terminals.len() == 1 {
            instructions.push(format!("rz {},{}", terminals[0], rz_bits as i64));
        } else {
            let edges = topo.steiner_tree_edges(&terminals);
            let root = terminals[0];
            let mut tree_adj = vec![vec![]; state.num_qubits as usize];
            for &(u, v) in &edges {
                tree_adj[u].push(v);
                tree_adj[v].push(u);
            }

            let mut visited = vec![false; state.num_qubits as usize];
            let mut post_order = Vec::new();
            let mut parent = vec![usize::MAX; state.num_qubits as usize];

            fn dfs(
                u: usize,
                p: usize,
                adj: &Vec<Vec<usize>>,
                vis: &mut Vec<bool>,
                po: &mut Vec<usize>,
                par: &mut Vec<usize>,
            ) {
                vis[u] = true;
                par[u] = p;
                for &v in &adj[u] {
                    if v != p && !vis[v] {
                        dfs(v, u, adj, vis, po, par);
                    }
                }
                po.push(u);
            }

            dfs(
                root,
                usize::MAX,
                &tree_adj,
                &mut visited,
                &mut post_order,
                &mut parent,
            );

            for &u in &post_order {
                if u != root {
                    let p = parent[u];
                    instructions.push(format!("cx {},{}", u, p));
                    total_cnots += 1;
                }
            }

            instructions.push(format!("rz {},{}", root, rz_bits as i64));

            for &u in post_order.iter().rev() {
                if u != root {
                    let p = parent[u];
                    instructions.push(format!("cx {},{}", u, p));
                    total_cnots += 1;
                }
            }
        }
    }

    let mut matrix = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        matrix.push(state.out_state[i].variable_mask);
    }

    if let Ok(cnots) =
        crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize)
    {
        let mut final_cnots = Vec::new();
        for (c, t) in cnots {
            crate::engine::engine_64::apply_remote_cnot(c, t, &topo, &mut final_cnots);
        }
        for (c, t) in &final_cnots {
            instructions.push(format!("cx {},{}", c, t));
        }
        total_cnots += final_cnots.len() as i64;
    } else {
        return "None".to_string();
    }
    append_x_layer(&mut instructions, &x_offsets);

    let synth_cost =
        ((instructions.len() as i64) - total_cnots) * rz_weight + total_cnots * cnot_weight;
    if synth_cost >= hw_cost {
        return "None".to_string();
    }

    if instructions.is_empty() {
        return "empty".to_string();
    }

    instructions.join(";")
}

pub fn synthesize_pmh_logic_64(state: PSum64, gate_count: i64) -> String {
    if gate_count <= 2 {
        return "None".to_string();
    }
    if state.is_overflowed() {
        return "None".to_string();
    }

    // 1. Purity Check
    // 1a. No discrete or continuous phase polynomial terms (rules out Z/S/T/H/Rz gates).
    if !state.phase_poly.terms.is_empty() || !state.continuous_poly.parities.is_empty() {
        return "None".to_string();
    }
    // 1b. No path variables in out_state (rules out H gates which introduce fresh path vars).
    let valid_mask = (1_u64 << state.num_qubits) - 1;
    for poly in &state.out_state {
        if (poly.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
    }
    // 1c. Affine/X offsets (monomial `0`) are kept and emitted as a trailing X layer.
    let x_offsets = affine_x_offsets_64(&state);

    // 2. Strict CNOT block boundary (pure-X / identity has rank 0 — still allow
    //    a trailing X-only answer when offsets are present).
    let rank = rank_m_minus_i_64(&state);
    if rank == 0 && x_offsets.is_empty() {
        return "None".to_string();
    }

    // 3. Rank(M - I) lower bound
    let r = rank;
    if rank > 0 && gate_count <= r {
        return "None".to_string();
    }

    // 4. PMH Synthesis
    let mut matrix = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        matrix.push(state.out_state[i].variable_mask);
    }

    if let Ok(cnots) =
        crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize)
    {
        let mut instructions: Vec<String> = cnots
            .iter()
            .map(|&(c, t)| format!("cx {},{}", c, t))
            .collect();
        append_x_layer(&mut instructions, &x_offsets);
        if (instructions.len() as i64) < gate_count {
            if instructions.is_empty() {
                return "empty".to_string();
            }
            // Named form whenever X appears; legacy "c,t;c,t" when CX-only.
            if x_offsets.is_empty() {
                return cnots
                    .iter()
                    .map(|&(c, t)| format!("{},{}", c, t))
                    .collect::<Vec<_>>()
                    .join(";");
            }
            return instructions.join(";");
        }
    } else if rank == 0 && !x_offsets.is_empty() {
        let mut instructions = Vec::new();
        append_x_layer(&mut instructions, &x_offsets);
        if (instructions.len() as i64) < gate_count {
            return instructions.join(";");
        }
    }

    "None".to_string()
}

/// GraySynth-style CX+RZ resynthesis on all-to-all connectivity (64-bit).
///
/// Unlike `synthesize_pmh` this accepts states *with* phase terms (the whole
/// point: RZ-bearing Hadamard-free blocks), and unlike `synthesize_steiner`
/// it needs no topology and shares parity prefixes across terms instead of
/// building and unbuilding one CNOT tree per term. Discrete phase monomials
/// are lifted to the parity basis by Mobius inversion first (the canonical
/// poly is monomial-based; treating masks as parities would be wrong for any
/// multi-bit mask).
///
/// Sound by construction downstream: proposals only merge via `state_union`
/// when the engine proves state equality.
pub fn synthesize_gray_logic_64(
    state: PSum64,
    gate_count: i64,
    hw_cost: i64,
    cnot_weight: i64,
    rz_weight: i64,
) -> String {
    if gate_count <= 2 {
        return "None".to_string();
    }
    if state.is_overflowed() {
        return "None".to_string();
    }
    if !continuous_parities_are_qubit_linear_64(&state) {
        return "None".to_string();
    }
    let n = state.num_qubits as usize;
    let valid_mask: u64 = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };

    // Purity: linear out_state over input variables; constant-1 (affine/X)
    // terms are allowed and emitted as a trailing X layer.
    let mut x_offsets = Vec::new();
    for (q, poly) in state.out_state.iter().enumerate() {
        if (poly.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        for &t in &poly.terms {
            if t == 0 {
                x_offsets.push(q);
            } else if t.count_ones() != 1 {
                return "None".to_string();
            }
        }
    }

    let mut parities: Vec<(u64, f64)> = Vec::new();
    for (i, &mask) in state.continuous_poly.parities.iter().enumerate() {
        parities.push((mask, crate::engine::ticks_to_angle(state.continuous_poly.phases[i])));
    }

    let mut monomials: Vec<(u64, f64)> = Vec::new();
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 61;
        let mask = term.0 & 0x1FFF_FFFF_FFFF_FFFF;
        if (mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        monomials.push((mask, (phase_unit as f64) * std::f64::consts::PI / 4.0));
    }
    match crate::engine::engine_64::phase_monomials_to_parities(&monomials) {
        Some(disc) => parities.extend(disc),
        None => return "None".to_string(),
    }

    let target: Vec<u64> = (0..n).map(|i| state.out_state[i].variable_mask).collect();
    match crate::engine::engine_64::synthesize_gray_network(&parities, &target, n) {
        Some((mut instructions, total_cnots)) => {
            append_x_layer(&mut instructions, &x_offsets);
            let rz_count = instructions.len() as i64 - total_cnots;
            let synth_cost = rz_count * rz_weight + total_cnots * cnot_weight;
            if synth_cost >= hw_cost {
                return "None".to_string();
            }
            if instructions.is_empty() {
                return "empty".to_string();
            }
            instructions.join(";")
        }
        None => "None".to_string(),
    }
}

pub fn synthesize_steiner_logic_128(
    state: PSum128,
    gate_count: i64,
    hw_cost: i64,
    topology_str: String,
    cnot_weight: i64,
    rz_weight: i64,
) -> String {
    if gate_count <= 2 {
        return "None".to_string();
    }
    if state.is_overflowed() {
        return "None".to_string();
    }
    if !continuous_parities_are_qubit_linear_128(&state) {
        return "None".to_string();
    }
    let valid_mask = (1_u128 << state.num_qubits) - 1;
    for poly in &state.out_state {
        if (poly.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
    }
    let x_offsets = affine_x_offsets_128(&state);

    let topo = crate::engine::engine_128::Topology::new(state.num_qubits as usize, &topology_str);
    let mut instructions = Vec::new();
    // Discrete phase poly is monomial-basis; lift to parities via Mobius
    // (same as Gray) before building per-term Steiner trees.
    let mut parities: Vec<(u128, f64)> = Vec::new();
    let mut monomials: Vec<(u128, f64)> = Vec::new();
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 125;
        let mask = term.0 & 0x1FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF;
        if (mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        monomials.push((mask, (phase_unit as f64) * std::f64::consts::PI / 4.0));
    }
    match crate::engine::engine_128::phase_monomials_to_parities(&monomials) {
        Some(disc) => parities.extend(disc),
        None => return "None".to_string(),
    }
    for (i, &mask) in state.continuous_poly.parities.iter().enumerate() {
        let angle = crate::engine::ticks_to_angle(state.continuous_poly.phases[i]);
        parities.push((mask, angle));
    }
    {
        use std::collections::HashMap;
        let mut merged: HashMap<u128, f64> = HashMap::new();
        for (mask, angle) in parities.drain(..) {
            if mask != 0 {
                *merged.entry(mask).or_insert(0.0) += angle;
            }
        }
        parities = merged.into_iter().collect();
    }

    let mut total_cnots = 0;
    for (mask, angle) in parities {
        let Some(rz_bits) = normalized_rz_bits(angle) else {
            continue;
        };
        let mut terminals = Vec::new();
        for i in 0..state.num_qubits {
            if (mask & (1u128 << i)) != 0 {
                terminals.push(i as usize);
            }
        }

        if terminals.is_empty() {
            continue;
        } else if terminals.len() == 1 {
            instructions.push(format!("rz {},{}", terminals[0], rz_bits as i64));
        } else {
            let edges = topo.steiner_tree_edges(&terminals);
            let root = terminals[0];
            let mut tree_adj = vec![vec![]; state.num_qubits as usize];
            for &(u, v) in &edges {
                tree_adj[u].push(v);
                tree_adj[v].push(u);
            }

            let mut visited = vec![false; state.num_qubits as usize];
            let mut post_order = Vec::new();
            let mut parent = vec![usize::MAX; state.num_qubits as usize];

            fn dfs(
                u: usize,
                p: usize,
                adj: &Vec<Vec<usize>>,
                vis: &mut Vec<bool>,
                po: &mut Vec<usize>,
                par: &mut Vec<usize>,
            ) {
                vis[u] = true;
                par[u] = p;
                for &v in &adj[u] {
                    if v != p && !vis[v] {
                        dfs(v, u, adj, vis, po, par);
                    }
                }
                po.push(u);
            }

            dfs(
                root,
                usize::MAX,
                &tree_adj,
                &mut visited,
                &mut post_order,
                &mut parent,
            );

            for &u in &post_order {
                if u != root {
                    let p = parent[u];
                    instructions.push(format!("cx {},{}", u, p));
                    total_cnots += 1;
                }
            }

            instructions.push(format!("rz {},{}", root, rz_bits as i64));

            for &u in post_order.iter().rev() {
                if u != root {
                    let p = parent[u];
                    instructions.push(format!("cx {},{}", u, p));
                    total_cnots += 1;
                }
            }
        }
    }

    let mut matrix = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        matrix.push(state.out_state[i].variable_mask);
    }

    if let Ok(cnots) =
        crate::engine::engine_128::synthesize_cnot_matrix(matrix, state.num_qubits as usize)
    {
        let mut final_cnots = Vec::new();
        for (c, t) in cnots {
            crate::engine::engine_128::apply_remote_cnot(c, t, &topo, &mut final_cnots);
        }
        for (c, t) in &final_cnots {
            instructions.push(format!("cx {},{}", c, t));
        }
        total_cnots += final_cnots.len() as i64;
    } else {
        return "None".to_string();
    }
    append_x_layer(&mut instructions, &x_offsets);

    let synth_cost =
        ((instructions.len() as i64) - total_cnots) * rz_weight + total_cnots * cnot_weight;
    if synth_cost >= hw_cost {
        return "None".to_string();
    }

    if instructions.is_empty() {
        return "empty".to_string();
    }

    instructions.join(";")
}

/// GraySynth-style CX+RZ resynthesis on all-to-all connectivity (128-bit).
/// See `synthesize_gray_logic_64` for the rationale and guards.
pub fn synthesize_gray_logic_128(
    state: PSum128,
    gate_count: i64,
    hw_cost: i64,
    cnot_weight: i64,
    rz_weight: i64,
) -> String {
    if gate_count <= 2 {
        return "None".to_string();
    }
    if state.is_overflowed() {
        return "None".to_string();
    }
    if !continuous_parities_are_qubit_linear_128(&state) {
        return "None".to_string();
    }
    let n = state.num_qubits as usize;
    let valid_mask: u128 = if n >= 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    };

    let mut x_offsets = Vec::new();
    for (q, poly) in state.out_state.iter().enumerate() {
        if (poly.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        for &t in &poly.terms {
            if t == 0 {
                x_offsets.push(q);
            } else if t.count_ones() != 1 {
                return "None".to_string();
            }
        }
    }

    let mut parities: Vec<(u128, f64)> = Vec::new();
    for (i, &mask) in state.continuous_poly.parities.iter().enumerate() {
        parities.push((mask, crate::engine::ticks_to_angle(state.continuous_poly.phases[i])));
    }

    let mut monomials: Vec<(u128, f64)> = Vec::new();
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 125;
        let mask = term.0 & 0x1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
        if (mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        monomials.push((mask, (phase_unit as f64) * std::f64::consts::PI / 4.0));
    }
    match crate::engine::engine_128::phase_monomials_to_parities(&monomials) {
        Some(disc) => parities.extend(disc),
        None => return "None".to_string(),
    }

    let target: Vec<u128> = (0..n).map(|i| state.out_state[i].variable_mask).collect();
    match crate::engine::engine_128::synthesize_gray_network(&parities, &target, n) {
        Some((mut instructions, total_cnots)) => {
            append_x_layer(&mut instructions, &x_offsets);
            let rz_count = instructions.len() as i64 - total_cnots;
            let synth_cost = rz_count * rz_weight + total_cnots * cnot_weight;
            if synth_cost >= hw_cost {
                return "None".to_string();
            }
            if instructions.is_empty() {
                return "empty".to_string();
            }
            instructions.join(";")
        }
        None => "None".to_string(),
    }
}

pub fn synthesize_pmh_logic_128(state: PSum128, gate_count: i64) -> String {
    if gate_count <= 2 {
        return "None".to_string();
    }
    if state.is_overflowed() {
        return "None".to_string();
    }
    // 1. Purity Check
    // 1a. No discrete or continuous phase polynomial terms (rules out Z/S/T/H/Rz gates).
    if !state.phase_poly.terms.is_empty() || !state.continuous_poly.parities.is_empty() {
        return "None".to_string();
    }
    // 1b. No path variables in out_state (rules out H gates which introduce fresh path vars).
    let valid_mask = (1_u128 << state.num_qubits) - 1;
    for poly in &state.out_state {
        if (poly.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
    }
    // 1c. Affine/X offsets kept and emitted as a trailing X layer.
    let x_offsets = affine_x_offsets_128(&state);

    // 2. Strict CNOT block boundary
    let rank = rank_m_minus_i_128(&state);
    if rank == 0 && x_offsets.is_empty() {
        return "None".to_string();
    }

    // 3. Rank(M - I) lower bound
    let r = rank;
    if rank > 0 && gate_count <= r {
        return "None".to_string();
    }

    // 4. PMH Synthesis
    let mut matrix = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        matrix.push(state.out_state[i].variable_mask);
    }

    if let Ok(cnots) =
        crate::engine::engine_128::synthesize_cnot_matrix(matrix, state.num_qubits as usize)
    {
        let mut instructions: Vec<String> = cnots
            .iter()
            .map(|&(c, t)| format!("cx {},{}", c, t))
            .collect();
        append_x_layer(&mut instructions, &x_offsets);
        if (instructions.len() as i64) < gate_count {
            if instructions.is_empty() {
                return "empty".to_string();
            }
            if x_offsets.is_empty() {
                return cnots
                    .iter()
                    .map(|&(c, t)| format!("{},{}", c, t))
                    .collect::<Vec<_>>()
                    .join(";");
            }
            return instructions.join(";");
        }
    } else if rank == 0 && !x_offsets.is_empty() {
        let mut instructions = Vec::new();
        append_x_layer(&mut instructions, &x_offsets);
        if (instructions.len() as i64) < gate_count {
            return instructions.join(";");
        }
    }

    "None".to_string()
}

// --- EGG SORT REGISTRATION ---

#[derive(Debug)]
pub struct PathSumSort64;
impl BaseSort for PathSumSort64 {
    type Base = PSum64;
    fn name(&self) -> &str {
        "PathSum64"
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(
            eg,
            "rust_id_pathsum_64" = |q: i64| -> PSum64 { id_pathsum_logic_64(q) }
        );
        add_primitive!(
            eg,
            "rust_apply_x_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_no_reduce_logic_64(s, q, |st, q_| st.apply_x(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_z_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_logic_64(s, q, |st, q_| st.apply_z(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_s_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_logic_64(s, q, |st, q_| st.apply_s(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_sdg_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_logic_64(s, q, |st, q_| st.apply_sdg(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_t_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_logic_64(s, q, |st, q_| st.apply_t(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_tdg_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_logic_64(s, q, |st, q_| st.apply_tdg(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_sx_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_logic_64(s, q, |st, q_| st.apply_sx(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_h_64" = |s: PSum64, q: i64| -> PSum64 {
                apply_gate_logic_64(s, q, |st, q_| st.apply_h(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_cx_64" =
                |s: PSum64, qc: i64, qt: i64| -> PSum64 { apply_cx_logic_64(s, qc, qt) }
        );
        add_primitive!(
            eg,
            "rust_apply_rz_64" =
                |s: PSum64, q: i64, t: i64| -> PSum64 { apply_rz_logic_64(s, q, t) }
        );
        add_primitive!(
            eg,
            "rust_synthesize_pmh_64" =
                |s: PSum64, count: i64| -> S { S::new(synthesize_pmh_logic_64(s, count)) }
        );
        add_primitive!(
            eg,
            "rust_synthesize_steiner_64" =
                |s: PSum64, count: i64, hw_cost: i64, top: S, cnot_wt: i64, rz_wt: i64| -> S {
                    S::new(synthesize_steiner_logic_64(
                        s,
                        count,
                        hw_cost,
                        top.to_string(),
                        cnot_wt,
                        rz_wt,
                    ))
                }
        );
        add_primitive!(
            eg,
            "rust_synthesize_gray_64" =
                |s: PSum64, count: i64, hw_cost: i64, cnot_wt: i64, rz_wt: i64| -> S {
                    S::new(synthesize_gray_logic_64(s, count, hw_cost, cnot_wt, rz_wt))
                }
        );
        add_primitive!(
            eg,
            "rust_state_fingerprint_64" =
                |s: PSum64| -> S { S::new(state_fingerprint_logic_64(s)) }
        );
        add_primitive!(
            eg,
            "rust_debug_pathsum_64" = |s: PSum64| -> S { S::new(debug_pathsum_logic_64(s)) }
        );
        add_primitive!(
            eg,
            "rust_add_rz_bits_64" = |a: i64, b: i64| -> i64 { rust_add_rz_bits_logic(a, b) }
        );
        add_primitive!(
            eg,
            "rust_negate_rz_bits_64" = |a: i64| -> i64 { rust_negate_rz_bits_logic(a) }
        );
    }
    fn reconstruct_termdag(&self, _bv: &BaseValues, _v: Value, td: &mut TermDag) -> TermId {
        let arg = td.lit(Literal::Int(0));
        td.app("rust_id_pathsum_64".to_string(), vec![arg])
    }
}

#[derive(Debug)]
pub struct PathSumSort128;
impl BaseSort for PathSumSort128 {
    type Base = PSum128;
    fn name(&self) -> &str {
        "PathSum128"
    }

    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(
            eg,
            "rust_id_pathsum_128" = |q: i64| -> PSum128 { id_pathsum_logic_128(q) }
        );
        add_primitive!(
            eg,
            "rust_apply_x_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_no_reduce_logic_128(s, q, |st, q_| st.apply_x(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_z_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_logic_128(s, q, |st, q_| st.apply_z(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_s_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_logic_128(s, q, |st, q_| st.apply_s(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_sdg_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_logic_128(s, q, |st, q_| st.apply_sdg(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_t_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_logic_128(s, q, |st, q_| st.apply_t(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_tdg_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_logic_128(s, q, |st, q_| st.apply_tdg(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_sx_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_logic_128(s, q, |st, q_| st.apply_sx(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_h_128" = |s: PSum128, q: i64| -> PSum128 {
                apply_gate_logic_128(s, q, |st, q_| st.apply_h(q_))
            }
        );
        add_primitive!(
            eg,
            "rust_apply_cx_128" =
                |s: PSum128, qc: i64, qt: i64| -> PSum128 { apply_cx_logic_128(s, qc, qt) }
        );
        add_primitive!(
            eg,
            "rust_apply_rz_128" =
                |s: PSum128, q: i64, t: i64| -> PSum128 { apply_rz_logic_128(s, q, t) }
        );
        add_primitive!(
            eg,
            "rust_synthesize_pmh_128" =
                |s: PSum128, count: i64| -> S { S::new(synthesize_pmh_logic_128(s, count)) }
        );
        add_primitive!(
            eg,
            "rust_synthesize_steiner_128" =
                |s: PSum128, count: i64, hw_cost: i64, top: S, cnot_wt: i64, rz_wt: i64| -> S {
                    S::new(synthesize_steiner_logic_128(
                        s,
                        count,
                        hw_cost,
                        top.to_string(),
                        cnot_wt,
                        rz_wt,
                    ))
                }
        );
        add_primitive!(
            eg,
            "rust_synthesize_gray_128" =
                |s: PSum128, count: i64, hw_cost: i64, cnot_wt: i64, rz_wt: i64| -> S {
                    S::new(synthesize_gray_logic_128(s, count, hw_cost, cnot_wt, rz_wt))
                }
        );
        add_primitive!(
            eg,
            "rust_state_fingerprint_128" =
                |s: PSum128| -> S { S::new(state_fingerprint_logic_128(s)) }
        );
        add_primitive!(
            eg,
            "rust_debug_pathsum_128" = |s: PSum128| -> S { S::new(debug_pathsum_logic_128(s)) }
        );
        add_primitive!(
            eg,
            "rust_add_rz_bits_128" = |a: i64, b: i64| -> i64 { rust_add_rz_bits_logic(a, b) }
        );
        add_primitive!(
            eg,
            "rust_negate_rz_bits_128" = |a: i64| -> i64 { rust_negate_rz_bits_logic(a) }
        );
    }
    fn reconstruct_termdag(&self, _bv: &BaseValues, _v: Value, td: &mut TermDag) -> TermId {
        let arg = td.lit(Literal::Int(0));
        td.app("rust_id_pathsum_128".to_string(), vec![arg])
    }
}

/// A host-side PathSum gate for [`fingerprint_ops_logic_64`] /
/// [`fingerprint_ops_logic_128`]. Unsupported names return `None` from
/// [`FingerprintOp::parse`] so the caller can refuse the circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FingerprintOp {
    X(i64),
    Z(i64),
    S(i64),
    Sdg(i64),
    T(i64),
    Tdg(i64),
    H(i64),
    Sx(i64),
    RzBits(i64, i64),
    Cx(i64, i64),
}

impl FingerprintOp {
    pub fn parse(name: &str, args: &[i64]) -> Option<Self> {
        match name {
            "x" if !args.is_empty() => Some(Self::X(args[0])),
            "z" if !args.is_empty() => Some(Self::Z(args[0])),
            "s" if !args.is_empty() => Some(Self::S(args[0])),
            "sdg" if !args.is_empty() => Some(Self::Sdg(args[0])),
            "t" if !args.is_empty() => Some(Self::T(args[0])),
            "tdg" if !args.is_empty() => Some(Self::Tdg(args[0])),
            "h" if !args.is_empty() => Some(Self::H(args[0])),
            "sx" if !args.is_empty() => Some(Self::Sx(args[0])),
            "rz_bits" if args.len() >= 2 => Some(Self::RzBits(args[0], args[1])),
            "cx" if args.len() >= 2 => Some(Self::Cx(args[0], args[1])),
            _ => None,
        }
    }
}

/// Fold `ops` through the same apply/`reduce` path as the FFI wrappers and
/// return the interned `eq_fingerprint` string. Overflowed states return
/// their overflow fingerprint (not `None`). `None` is only for an
/// unsupported gate name.
pub fn fingerprint_ops_logic_64(n: i64, ops: &[FingerprintOp]) -> Option<String> {
    let mut ps = id_pathsum_logic_64(n);
    for op in ops {
        ps = match *op {
            FingerprintOp::X(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_x(qq)),
            FingerprintOp::Z(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_z(qq)),
            FingerprintOp::S(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_s(qq)),
            FingerprintOp::Sdg(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_sdg(qq)),
            FingerprintOp::T(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_t(qq)),
            FingerprintOp::Tdg(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_tdg(qq)),
            FingerprintOp::H(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_h(qq)),
            FingerprintOp::Sx(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_sx(qq)),
            FingerprintOp::RzBits(q, bits) => apply_rz_logic_64(ps, q, bits),
            FingerprintOp::Cx(c, t) => apply_cx_logic_64(ps, c, t),
        };
    }
    Some(state_fingerprint_logic_64(ps))
}

pub fn fingerprint_ops_logic_128(n: i64, ops: &[FingerprintOp]) -> Option<String> {
    let mut ps = id_pathsum_logic_128(n);
    for op in ops {
        ps = match *op {
            FingerprintOp::X(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_x(qq)),
            FingerprintOp::Z(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_z(qq)),
            FingerprintOp::S(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_s(qq)),
            FingerprintOp::Sdg(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_sdg(qq)),
            FingerprintOp::T(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_t(qq)),
            FingerprintOp::Tdg(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_tdg(qq)),
            FingerprintOp::H(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_h(qq)),
            FingerprintOp::Sx(q) => apply_gate_logic_128(ps, q, |st, qq| st.apply_sx(qq)),
            FingerprintOp::RzBits(q, bits) => apply_rz_logic_128(ps, q, bits),
            FingerprintOp::Cx(c, t) => apply_cx_logic_128(ps, c, t),
        };
    }
    Some(state_fingerprint_logic_128(ps))
}

/// Complete canonical state fingerprint (64-bit): the interned equality
/// key of `EvaluatedPathSum` (`PartialEq` / `Hash`), so two fingerprints
/// match iff `state_union` would merge the states. Continuous phases are
/// the 1e8 snap ticks, not raw `f64` Debug (`debug_pathsum` is still not
/// an equality check: it omits `out_state`).
pub fn state_fingerprint_logic_64(state: PSum64) -> String {
    state.into_inner().eq_fingerprint()
}

/// Complete canonical state fingerprint (128-bit); see the 64-bit variant.
pub fn state_fingerprint_logic_128(state: PSum128) -> String {
    state.into_inner().eq_fingerprint()
}

pub fn debug_pathsum_logic_64(state: PSum64) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Continuous Parities: {:?}\n",
        state.continuous_poly.parities
    ));
    out.push_str(&format!(
        "Continuous Phases: {:?}\n",
        state
            .continuous_poly
            .phases
            .iter()
            .map(|&t| crate::engine::ticks_to_angle(t))
            .collect::<Vec<f64>>()
    ));
    out.push_str("Phase Poly Terms: ");
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 61;
        let mask = term.0 & 0x1FFFFFFFFFFFFFFF;
        out.push_str(&format!("(mask={}, phase_unit={}) ", mask, phase_unit));
    }
    out
}

pub fn debug_pathsum_logic_128(state: PSum128) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Continuous Parities: {:?}\n",
        state.continuous_poly.parities
    ));
    out.push_str(&format!(
        "Continuous Phases: {:?}\n",
        state
            .continuous_poly
            .phases
            .iter()
            .map(|&t| crate::engine::ticks_to_angle(t))
            .collect::<Vec<f64>>()
    ));
    out.push_str("Phase Poly Terms: ");
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 125;
        let mask = term.0 & 0x1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
        out.push_str(&format!("(mask={}, phase_unit={}) ", mask, phase_unit));
    }
    out
}

#[cfg(test)]
mod gauge_ffi_tests {
    use super::*;

    /// The X / CX wrappers skip `reduce()`; they must still hand back a
    /// canonical gauge so a CX-terminated state interns equal to the same
    /// operator reached through a reducing gate.
    #[test]
    fn cx_and_x_wrappers_keep_the_gauge_canonical() {
        // H0 H1 CX01 == CX10 H0 H1
        let mut a = id_pathsum_logic_64(2);
        a = apply_gate_logic_64(a, 0, |st, q| st.apply_h(q));
        a = apply_gate_logic_64(a, 1, |st, q| st.apply_h(q));
        a = apply_cx_logic_64(a, 0, 1);
        let mut b = id_pathsum_logic_64(2);
        b = apply_cx_logic_64(b, 1, 0);
        b = apply_gate_logic_64(b, 0, |st, q| st.apply_h(q));
        b = apply_gate_logic_64(b, 1, |st, q| st.apply_h(q));
        assert_eq!(a, b, "CX-terminated state must be canonical");
        assert_eq!(
            state_fingerprint_logic_64(a.clone()),
            state_fingerprint_logic_64(b.clone())
        );

        // H0 X0 == Z0 H0
        let mut c = id_pathsum_logic_64(1);
        c = apply_gate_logic_64(c, 0, |st, q| st.apply_h(q));
        c = apply_gate_no_reduce_logic_64(c, 0, |st, q| st.apply_x(q));
        let mut d = id_pathsum_logic_64(1);
        d = apply_gate_logic_64(d, 0, |st, q| st.apply_z(q));
        d = apply_gate_logic_64(d, 0, |st, q| st.apply_h(q));
        assert_eq!(c, d, "X-terminated state must be canonical");

        // Same for the 128-bit engine.
        let mut a = id_pathsum_logic_128(2);
        a = apply_gate_logic_128(a, 0, |st, q| st.apply_h(q));
        a = apply_gate_logic_128(a, 1, |st, q| st.apply_h(q));
        a = apply_cx_logic_128(a, 0, 1);
        let mut b = id_pathsum_logic_128(2);
        b = apply_cx_logic_128(b, 1, 0);
        b = apply_gate_logic_128(b, 0, |st, q| st.apply_h(q));
        b = apply_gate_logic_128(b, 1, |st, q| st.apply_h(q));
        assert_eq!(a, b);

        // Negative control: H0 H1 CX10 (= CX01 H0 H1) is a different operator.
        let mut e = id_pathsum_logic_64(2);
        e = apply_gate_logic_64(e, 0, |st, q| st.apply_h(q));
        e = apply_gate_logic_64(e, 1, |st, q| st.apply_h(q));
        e = apply_cx_logic_64(e, 1, 0);
        assert_ne!(e, c);
        assert_ne!(state_fingerprint_logic_64(e), state_fingerprint_logic_64(d));
    }
}

#[cfg(test)]
mod fingerprint_tests {
    use super::*;
    use crate::engine::engine_64;

    /// `state_union` / interned `PartialEq` compare continuous phases on the
    /// integer tick lattice. The host fingerprint must follow that lattice,
    /// not raw `f64` angles, so pre-pass string compare matches in-cycle merge.
    #[test]
    fn fingerprint_matches_snapped_eq_not_raw_debug() {
        use crate::engine::{TICKS_PER_TURN, ticks_to_angle};
        let tick = std::f64::consts::TAU / TICKS_PER_TURN as f64;
        // An angle 0.3 ticks above a lattice point: +0.15 ticks rounds to the
        // same tick, +0.4 ticks crosses to the next one.
        let base = ticks_to_angle(31_615_000) + 0.3 * tick;

        let mut a = engine_64::EvaluatedPathSum::new_id(1);
        a.apply_rz(0, base);
        assert!(
            !a.continuous_poly.phases.is_empty(),
            "the angle is not a π/4 multiple; it must stay in the continuous poly"
        );

        let mut b = engine_64::EvaluatedPathSum::new_id(1);
        b.apply_rz(0, base + 0.15 * tick);
        assert_eq!(
            a, b,
            "perturbation inside the same lattice tick must be PartialEq"
        );

        let fp_a = state_fingerprint_logic_64(PSum64::new(Arc::new(a.clone())));
        let fp_b = state_fingerprint_logic_64(PSum64::new(Arc::new(b)));
        assert_eq!(
            fp_a, fp_b,
            "tick-equal states must share the host fingerprint"
        );
        assert!(
            fp_a.contains("num_qubits: 1"),
            "AGES still matches this Debug-shaped field; got {fp_a}"
        );
        assert!(
            !fp_a.contains("phases: [0."),
            "fingerprint must emit lattice ticks, not raw f64 angles: {fp_a}"
        );

        let mut c = engine_64::EvaluatedPathSum::new_id(1);
        c.apply_rz(0, base + 0.4 * tick);
        assert_ne!(a, c, "crossing a lattice tick must change PartialEq");
        assert_ne!(
            state_fingerprint_logic_64(PSum64::new(Arc::new(a))),
            state_fingerprint_logic_64(PSum64::new(Arc::new(c))),
            "crossing a lattice tick must change the host fingerprint"
        );
    }

    fn fold_ops_64(n: i64, ops: &[FingerprintOp]) -> PSum64 {
        let mut ps = id_pathsum_logic_64(n);
        for op in ops {
            ps = match *op {
                FingerprintOp::X(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_x(qq)),
                FingerprintOp::Z(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_z(qq)),
                FingerprintOp::S(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_s(qq)),
                FingerprintOp::Sdg(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_sdg(qq)),
                FingerprintOp::T(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_t(qq)),
                FingerprintOp::Tdg(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_tdg(qq)),
                FingerprintOp::H(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_h(qq)),
                FingerprintOp::Sx(q) => apply_gate_logic_64(ps, q, |st, qq| st.apply_sx(qq)),
                FingerprintOp::RzBits(q, bits) => apply_rz_logic_64(ps, q, bits),
                FingerprintOp::Cx(c, t) => apply_cx_logic_64(ps, c, t),
            };
        }
        ps
    }

    #[test]
    fn fingerprint_ops_matches_ffi_eq_fingerprint() {
        let bits = |theta: f64| theta.to_bits() as i64;
        let h01 = [FingerprintOp::H(0), FingerprintOp::H(1)];
        let h10 = [FingerprintOp::H(1), FingerprintOp::H(0)];
        let nam_h = [FingerprintOp::H(0)];
        let ibm_h = [
            FingerprintOp::RzBits(0, bits(std::f64::consts::FRAC_PI_2)),
            FingerprintOp::Sx(0),
            FingerprintOp::RzBits(0, bits(std::f64::consts::FRAC_PI_2)),
        ];
        for (n, ops) in [
            (2i64, h01.as_slice()),
            (2, h10.as_slice()),
            (1, nam_h.as_slice()),
            (1, ibm_h.as_slice()),
        ] {
            let folded = fold_ops_64(n, ops);
            let via_ops = fingerprint_ops_logic_64(n, ops).expect("supported ops");
            assert_eq!(
                via_ops,
                state_fingerprint_logic_64(folded),
                "fingerprint_ops must match FFI-folded eq_fingerprint for {ops:?}"
            );
        }
        assert_eq!(
            fingerprint_ops_logic_64(2, &h01),
            fingerprint_ops_logic_64(2, &h10),
            "commuting H order must share a fingerprint"
        );
        assert_eq!(
            fingerprint_ops_logic_64(1, &nam_h),
            fingerprint_ops_logic_64(1, &ibm_h),
            "nam H and ibm RZ SX RZ must share a fingerprint"
        );
        assert_eq!(
            FingerprintOp::parse("ry", &[0]),
            None,
            "unsupported gates must refuse"
        );
        assert_eq!(fingerprint_ops_logic_64(1, &[]).unwrap().contains("num_qubits: 1"), true);
    }

    #[test]
    fn fingerprint_ops_overflow_returns_string_not_none() {
        // n=1 with 60 live path vars is at the 64-bit cap; the next H overflows.
        let mut ops = vec![FingerprintOp::H(0); 61];
        // 61 H without reduce would overflow; FFI reduce may cancel pairs.
        // Force overflow via a hand-built state and compare tokens via apply.
        let mut state = engine_64::EvaluatedPathSum::new_id(1);
        state.num_path_vars = 60;
        state.apply_h(0);
        assert!(state.is_overflowed());
        let overflow_fp = state.eq_fingerprint();
        let id_fp = fingerprint_ops_logic_64(1, &[]).unwrap();
        assert_ne!(overflow_fp, id_fp);
        // Direct helper still returns Some for a supported (empty) list.
        assert!(fingerprint_ops_logic_64(1, &ops).is_some());
        let _ = ops.pop();
    }

    #[test]
    fn fingerprint_ops_sixty_qubit_second_h_overflows() {
        // 60 qubits + 1 live path var is the 64-bit cap. The first H allocates;
        // the second H must overflow and still return a string.
        let ops = [FingerprintOp::H(0), FingerprintOp::H(1)];
        let overflowed = fingerprint_ops_logic_64(60, &ops).expect("supported ops");
        let again = fingerprint_ops_logic_64(60, &ops).expect("supported ops");
        let well = fingerprint_ops_logic_64(2, &ops).expect("supported ops");
        let identity = fingerprint_ops_logic_64(60, &[]).expect("empty ops");
        assert_ne!(overflowed, well);
        assert_ne!(overflowed, identity);
        assert_ne!(
            overflowed, again,
            "overflow tokens must be unique across calls"
        );
        assert_ne!(
            overflow_id_in(&overflowed),
            0,
            "overflow fingerprint must carry a nonzero token: {overflowed}"
        );
        assert_eq!(
            overflow_id_in(&well),
            0,
            "below-cap H H must stay well-formed: {well}"
        );
    }

    fn overflow_id_in(fp: &str) -> u64 {
        let key = "overflow_id: ";
        let start = fp.find(key).unwrap_or_else(|| panic!("{fp}")) + key.len();
        let digits = fp[start..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(fp.len() - start);
        fp[start..start + digits].parse().unwrap()
    }
}

#[cfg(test)]
mod gray_tests {
    use super::*;
    use crate::engine::{engine_128, engine_64};

    fn bits(theta: f64) -> i64 {
        // Snap like the Python DSL does so angles are grid-aligned.
        let snapped = (theta * 100_000_000.0).round() / 100_000_000.0;
        snapped.to_bits() as i64
    }

    /// Applies a synthesized ";"-joined instruction string ("cx c,t" /
    /// "rz q,bits") to a fresh identity state through the real engine.
    fn replay(instructions: &str, n: i64) -> PSum64 {
        let mut ps = id_pathsum_logic_64(n);
        for instr in instructions.split(';') {
            let instr = instr.trim();
            if instr.is_empty() {
                continue;
            }
            let (gate, args) = instr.split_once(' ').expect("gate args");
            let parts: Vec<i64> = args
                .split(',')
                .map(|v| v.trim().parse::<i64>().unwrap())
                .collect();
            ps = match gate {
                "cx" => apply_cx_logic_64(ps, parts[0], parts[1]),
                "rz" => apply_rz_logic_64(ps, parts[0], parts[1]),
                "x" => apply_gate_logic_64(ps, parts[0], |st, q| st.apply_x(q)),
                other => panic!("unexpected gate from gray synthesis: {other}"),
            };
        }
        ps
    }

    fn build(gates: &[(&str, i64, i64)], n: i64) -> PSum64 {
        let mut ps = id_pathsum_logic_64(n);
        for &(g, a, b) in gates {
            ps = match g {
                "cx" => apply_cx_logic_64(ps, a, b),
                "rz" => apply_rz_logic_64(ps, a, b),
                "t" => apply_gate_logic_64(ps, a, |st, q| st.apply_t(q)),
                "s" => apply_gate_logic_64(ps, a, |st, q| st.apply_s(q)),
                "z" => apply_gate_logic_64(ps, a, |st, q| st.apply_z(q)),
                other => panic!("unexpected test gate {other}"),
            };
        }
        ps
    }

    fn assert_gray_equal(gates: &[(&str, i64, i64)], n: i64) -> String {
        let ps = build(gates, n);
        let orig = ps.clone();
        let fp_orig = state_fingerprint_logic_64(ps.clone());
        let out = synthesize_gray_logic_64(ps, 1_000, i64::MAX, 50, 1);
        assert_ne!(out, "None", "gray refused a synthesizable state");
        let replayed = if out == "empty" {
            id_pathsum_logic_64(n)
        } else {
            replay(&out, n)
        };
        assert_eq!(
            state_fingerprint_logic_64(replayed.clone()),
            fp_orig,
            "resynthesized circuit is not engine-equal (instructions: {out})"
        );
        assert_eq!(
            orig.into_inner(),
            replayed.into_inner(),
            "PathSum PartialEq failed (the state_union trust base); instructions: {out}"
        );
        out
    }

    #[test]
    fn continuous_parity_ladder_round_trips() {
        // Classic UCCSD-style CX ladder with a rotation at the bottom.
        let g = [
            ("cx", 0, 1),
            ("cx", 1, 2),
            ("cx", 2, 3),
            ("rz", 3, bits(0.37)),
            ("cx", 2, 3),
            ("cx", 1, 2),
            ("cx", 0, 1),
        ];
        let out = assert_gray_equal(&g, 4);
        // Quality: the naive build/unbuild ladder uses 6 CX; GraySynth must
        // not exceed that on a single parity.
        let cx_count = out.matches("cx").count();
        assert!(cx_count <= 6, "no CX saving over per-term trees: {out}");
    }

    #[test]
    fn discrete_monomials_convert_and_round_trip() {
        // T on a parity wire stores multilinear monomials; Mobius inversion
        // must recover a parity network the engine proves equal.
        let g = [
            ("cx", 0, 1),
            ("t", 1, 0),
            ("cx", 0, 1),
            ("s", 0, 0),
            ("z", 2, 0),
        ];
        assert_gray_equal(&g, 3);
    }

    #[test]
    fn shared_prefixes_beat_per_term_trees() {
        // Two rotations on overlapping parities plus a residual linear
        // function (the trailing CX ladder is NOT undone).
        let g = [
            ("cx", 0, 1),
            ("rz", 1, bits(0.25)),
            ("cx", 2, 1),
            ("rz", 1, bits(0.5)),
            ("cx", 3, 1),
            ("rz", 1, bits(1.25)),
        ];
        assert_gray_equal(&g, 4);
    }

    #[test]
    fn pure_linear_block_round_trips() {
        let g = [("cx", 0, 1), ("cx", 1, 2), ("cx", 2, 0)];
        assert_gray_equal(&g, 3);
    }

    #[test]
    fn rejects_states_with_path_variables() {
        // A Hadamard introduces a path variable; gray must refuse.
        let mut ps = id_pathsum_logic_64(2);
        ps = apply_gate_logic_64(ps, 0, |st, q| st.apply_h(q));
        ps = apply_cx_logic_64(ps, 0, 1);
        let out = synthesize_gray_logic_64(ps, 1_000, i64::MAX, 50, 1);
        assert_eq!(out, "None");
    }

    fn assert_synth_sound_64(state: PSum64) {
        let n = state.num_qubits as i64;
        let gray = synthesize_gray_logic_64(state.clone(), 1_000, i64::MAX, 1, 1);
        let steiner =
            synthesize_steiner_logic_64(state.clone(), 1_000, i64::MAX, String::new(), 1, 1);
        let pmh = synthesize_pmh_logic_64(state.clone(), 1_000);
        for (kind, out) in [("gray", gray), ("steiner", steiner), ("pmh", pmh)] {
            if out == "None" {
                continue;
            }
            let replayed = if out == "empty" {
                id_pathsum_logic_64(n)
            } else {
                replay(&out, n)
            };
            assert_eq!(
                state.clone().into_inner(),
                replayed.clone().into_inner(),
                "{kind} replay interned Eq failed (instructions: {out})"
            );
            assert_eq!(
                state_fingerprint_logic_64(state.clone()),
                state_fingerprint_logic_64(replayed),
                "{kind} replay fingerprint failed (instructions: {out})"
            );
        }
    }

    #[test]
    fn refuses_h_then_rz_path_var_continuous() {
        let mut ps = id_pathsum_logic_64(2);
        ps = apply_gate_logic_64(ps, 0, |st, q| st.apply_h(q));
        ps = apply_rz_logic_64(ps, 0, bits(0.37));
        assert_eq!(
            synthesize_gray_logic_64(ps.clone(), 1_000, i64::MAX, 50, 1),
            "None"
        );
        assert_eq!(
            synthesize_steiner_logic_64(ps, 1_000, i64::MAX, String::new(), 50, 1),
            "None"
        );
    }

    #[test]
    fn refuses_path_var_bits_on_continuous_parity_with_clean_out_state() {
        // out_state is identity (qubit-linear) but a continuous parity carries
        // a path-variable bit. Steiner used to drop that bit via variable_mask.
        let mut state = engine_64::EvaluatedPathSum::new_id(2);
        state.num_path_vars = 1;
        let v = 1u64 << 2;
        state.continuous_poly.apply_phase(
            engine_64::BooleanPoly::from_terms(smallvec::smallvec![1u64 << 0, v]),
            0.37,
        );
        let ps = PSum64::new(Arc::new(state));
        assert_eq!(
            synthesize_gray_logic_64(ps.clone(), 1_000, i64::MAX, 50, 1),
            "None"
        );
        assert_eq!(
            synthesize_steiner_logic_64(ps.clone(), 1_000, i64::MAX, String::new(), 50, 1),
            "None"
        );

        let mut state128 = engine_128::EvaluatedPathSum::new_id(2);
        state128.num_path_vars = 1;
        let v128 = 1u128 << 2;
        state128.continuous_poly.apply_phase(
            engine_128::BooleanPoly::from_terms(smallvec::smallvec![1u128 << 0, v128]),
            0.37,
        );
        let ps128 = PSum128::new(Arc::new(state128));
        assert_eq!(
            synthesize_gray_logic_128(ps128.clone(), 1_000, i64::MAX, 50, 1),
            "None"
        );
        assert_eq!(
            synthesize_steiner_logic_128(ps128, 1_000, i64::MAX, String::new(), 50, 1),
            "None"
        );
    }

    #[test]
    fn overflowed_state_refuses_synthesis() {
        let mut state = engine_64::EvaluatedPathSum::new_id(1);
        state.num_path_vars = 60;
        state.apply_h(0);
        assert!(state.is_overflowed());
        let ps = PSum64::new(Arc::new(state));
        assert_eq!(
            synthesize_gray_logic_64(ps.clone(), 1_000, i64::MAX, 50, 1),
            "None"
        );
        assert_eq!(
            synthesize_steiner_logic_64(ps.clone(), 1_000, i64::MAX, String::new(), 50, 1),
            "None"
        );
        assert_eq!(synthesize_pmh_logic_64(ps, 1_000), "None");
    }

    #[test]
    fn hfree_synth_is_sound() {
        let mut rng = 0xA11C_E555_0000_0006u64;
        let mut lcg = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng
        };
        for n in [2i64, 3, 4] {
            for _ in 0..16 {
                let mut ps = id_pathsum_logic_64(n);
                for _ in 0..8 {
                    let q = (lcg() % n as u64) as i64;
                    match lcg() % 3 {
                        0 => ps = apply_gate_logic_64(ps, q, |st, qq| st.apply_x(qq)),
                        1 if n > 1 => {
                            let mut t = (lcg() % n as u64) as i64;
                            if t == q {
                                t = (t + 1) % n;
                            }
                            ps = apply_cx_logic_64(ps, q, t);
                        }
                        _ => ps = apply_rz_logic_64(ps, q, bits(0.37 + (lcg() % 7) as f64 * 0.11)),
                    }
                }
                assert_synth_sound_64(ps);
            }
        }
    }

    #[test]
    fn cost_gate_rejects_expensive_synthesis() {
        let g = [("cx", 0, 1), ("rz", 1, bits(0.37)), ("cx", 0, 1)];
        let ps = build(&g, 2);
        // hw_cost 0 means any synthesis is too expensive.
        let out = synthesize_gray_logic_64(ps, 1_000, 0, 50, 1);
        assert_eq!(out, "None");
    }

    #[test]
    fn affine_x_offset_round_trips_gray() {
        // CX block followed by a net X — previously refused as affine.
        let mut ps = id_pathsum_logic_64(2);
        ps = apply_cx_logic_64(ps, 0, 1);
        ps = apply_gate_logic_64(ps, 1, |st, q| st.apply_x(q));
        let orig = ps.clone();
        let fp_orig = state_fingerprint_logic_64(ps.clone());
        let out = synthesize_gray_logic_64(ps, 1_000, i64::MAX, 50, 1);
        assert_ne!(out, "None", "gray should accept affine/X offsets");
        assert!(out.contains("x 1"), "expected trailing X layer: {out}");
        let replayed = replay(&out, 2);
        assert_eq!(
            state_fingerprint_logic_64(replayed.clone()),
            fp_orig,
            "affine resynthesis not engine-equal (instructions: {out})"
        );
        assert_eq!(
            orig.into_inner(),
            replayed.into_inner(),
            "affine PathSum PartialEq failed (instructions: {out})"
        );
    }

    #[test]
    fn affine_x_offset_round_trips_pmh() {
        let mut ps = id_pathsum_logic_64(2);
        ps = apply_cx_logic_64(ps, 0, 1);
        ps = apply_gate_logic_64(ps, 0, |st, q| st.apply_x(q));
        let fp_orig = state_fingerprint_logic_64(ps.clone());
        let out = synthesize_pmh_logic_64(ps, 1_000);
        assert_ne!(out, "None", "pmh should accept affine/X offsets");
        assert!(out.contains("x 0"), "expected trailing X layer: {out}");
        let replayed = replay(&out, 2);
        assert_eq!(
            state_fingerprint_logic_64(replayed),
            fp_orig,
            "pmh affine resynthesis not engine-equal (instructions: {out})"
        );
    }

    fn assert_steiner_equal(gates: &[(&str, i64, i64)], n: i64) -> String {
        let ps = build(gates, n);
        let fp_orig = state_fingerprint_logic_64(ps.clone());
        // Empty topology string => fully connected (all-to-all).
        let out = synthesize_steiner_logic_64(ps, 1_000, i64::MAX, String::new(), 50, 1);
        assert_ne!(out, "None", "steiner refused a synthesizable state");
        let replayed = if out == "empty" {
            id_pathsum_logic_64(n)
        } else {
            replay(&out, n)
        };
        assert_eq!(
            state_fingerprint_logic_64(replayed),
            fp_orig,
            "steiner resynthesized circuit is not engine-equal (instructions: {out})"
        );
        out
    }

    #[test]
    fn steiner_discrete_monomials_mobius_round_trip() {
        // Without Mobius, T-on-parity stores multi-bit monomials that Steiner
        // would mis-interpret as parities; with Mobius the replay must match.
        let g = [
            ("cx", 0, 1),
            ("t", 1, 0),
            ("cx", 0, 1),
            ("s", 0, 0),
            ("z", 2, 0),
        ];
        assert_steiner_equal(&g, 3);
    }

    #[test]
    fn steiner_agrees_with_gray_on_all_to_all() {
        // Same discrete monomial fixture Gray already round-trips; empty
        // topology => all-to-all, so both synthesizers must match the source.
        let g = [
            ("cx", 0, 1),
            ("t", 1, 0),
            ("cx", 0, 1),
            ("s", 0, 0),
            ("z", 2, 0),
        ];
        let ps = build(&g, 3);
        let fp_orig = state_fingerprint_logic_64(ps.clone());
        let gray = synthesize_gray_logic_64(ps.clone(), 1_000, i64::MAX, 50, 1);
        let steiner = synthesize_steiner_logic_64(ps, 1_000, i64::MAX, String::new(), 50, 1);
        assert_ne!(gray, "None");
        assert_ne!(steiner, "None");
        assert_eq!(
            state_fingerprint_logic_64(replay(&gray, 3)),
            fp_orig,
            "gray replay drifted"
        );
        assert_eq!(
            state_fingerprint_logic_64(replay(&steiner, 3)),
            fp_orig,
            "steiner replay drifted (instructions: {steiner})"
        );
    }

    fn cx_count(instr: &[String]) -> i64 {
        instr.iter().filter(|s| s.starts_with("cx ")).count() as i64
    }

    #[test]
    fn paper_gray_eq_and_not_worse_than_nn_mean_cx() {
        // Random H-free CX+RZ maps: paper tour must PartialEq-replay and
        // must not exceed NN mean CX (the old production tour).
        let mut rng = 0xC0FFEE_u64;
        let mut lcg = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng
        };
        let mut paper_cx = 0i64;
        let mut nn_cx = 0i64;
        let mut cases = 0i64;
        for n in [3usize, 4, 6, 8] {
            for _ in 0..32 {
                let n_terms = 2 + (lcg() as usize % (n + 2));
                let mut parities = Vec::new();
                for _ in 0..n_terms {
                    let mask = lcg() & ((1u64 << n) - 1);
                    if mask == 0 {
                        continue;
                    }
                    let ang = ((lcg() % 7) as f64 + 1.0) * std::f64::consts::PI / 8.0;
                    parities.push((mask, ang));
                }
                if parities.is_empty() {
                    continue;
                }
                let target: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
                let paper =
                    crate::engine::engine_64::synthesize_gray_network(&parities, &target, n);
                let nn =
                    crate::engine::engine_64::synthesize_gray_network_nn(&parities, &target, n);
                let (p_instr, _) = paper.expect("paper GraySynth refused a linear H-free set");
                let (n_instr, _) = nn.expect("NN GraySynth refused a linear H-free set");
                paper_cx += cx_count(&p_instr);
                nn_cx += cx_count(&n_instr);
                cases += 1;

                let p_ast = p_instr.join(";");
                let n_ast = n_instr.join(";");
                let p_ps = if p_ast.is_empty() {
                    id_pathsum_logic_64(n as i64)
                } else {
                    replay(&p_ast, n as i64)
                };
                let n_ps = if n_ast.is_empty() {
                    id_pathsum_logic_64(n as i64)
                } else {
                    replay(&n_ast, n as i64)
                };
                assert_eq!(
                    p_ps.into_inner(),
                    n_ps.into_inner(),
                    "paper vs NN PathSum Eq failed n={n} paper={p_ast} nn={n_ast}"
                );
            }
        }
        assert!(cases > 0);
        assert!(
            paper_cx <= nn_cx,
            "paper mean CX worse than NN: paper={paper_cx} nn={nn_cx} over {cases} cases"
        );
    }

    #[test]
    fn paper_example_4_2_identity_pointed() {
        // arXiv:1712.01859 Example 4.2 columns (1-indexed x1..x4 = bits 0..3):
        // {x2⊕x3, x1, x1⊕x4, x1⊕x2⊕x3, x1⊕x2⊕x4, x1⊕x2}, all T = π/8, pointed at I.
        let pi8 = std::f64::consts::PI / 8.0;
        let parities: Vec<(u64, f64)> = vec![
            (0b0110, pi8),
            (0b0001, pi8),
            (0b1001, pi8),
            (0b0111, pi8),
            (0b1011, pi8),
            (0b0011, pi8),
        ];
        let n = 4usize;
        let target: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
        let (instr, _) = crate::engine::engine_64::synthesize_gray_network(&parities, &target, n)
            .expect("paper Example 4.2 must synthesize");
        let ast = instr.join(";");
        let replayed = replay(&ast, n as i64);

        let mut orig = id_pathsum_logic_64(n as i64);
        for &(mask, ang) in &parities {
            let members: Vec<i64> = (0..n as i64).filter(|&i| (mask >> i) & 1 == 1).collect();
            let t = *members.last().unwrap();
            for &s in &members[..members.len() - 1] {
                orig = apply_cx_logic_64(orig, s, t);
            }
            orig = apply_rz_logic_64(orig, t, ang.to_bits() as i64);
            for &s in members[..members.len() - 1].iter().rev() {
                orig = apply_cx_logic_64(orig, s, t);
            }
        }
        assert_eq!(
            orig.into_inner(),
            replayed.into_inner(),
            "Example 4.2 PathSum Eq failed (instructions: {ast})"
        );
    }
}

#[cfg(test)]
mod pmh_pathsum_tests {
    use super::*;

    fn replay_pmh_64(instructions: &str, n: i64) -> PSum64 {
        if instructions == "empty" {
            return id_pathsum_logic_64(n);
        }
        let mut ps = id_pathsum_logic_64(n);
        for instr in instructions.split(';') {
            let instr = instr.trim();
            if instr.is_empty() {
                continue;
            }
            if let Some(args) = instr.strip_prefix("cx ") {
                let parts: Vec<i64> = args.split(',').map(|v| v.trim().parse().unwrap()).collect();
                ps = apply_cx_logic_64(ps, parts[0], parts[1]);
            } else if let Some(args) = instr.strip_prefix("x ") {
                let q: i64 = args.trim().parse().unwrap();
                ps = apply_gate_logic_64(ps, q, |st, q| st.apply_x(q));
            } else {
                let parts: Vec<i64> = instr
                    .split(',')
                    .map(|v| v.trim().parse().unwrap())
                    .collect();
                ps = apply_cx_logic_64(ps, parts[0], parts[1]);
            }
        }
        ps
    }

    fn replay_pmh_128(instructions: &str, n: i64) -> PSum128 {
        if instructions == "empty" {
            return id_pathsum_logic_128(n);
        }
        let mut ps = id_pathsum_logic_128(n);
        for instr in instructions.split(';') {
            let instr = instr.trim();
            if instr.is_empty() {
                continue;
            }
            if let Some(args) = instr.strip_prefix("cx ") {
                let parts: Vec<i64> = args.split(',').map(|v| v.trim().parse().unwrap()).collect();
                ps = apply_cx_logic_128(ps, parts[0], parts[1]);
            } else if let Some(args) = instr.strip_prefix("x ") {
                let q: i64 = args.trim().parse().unwrap();
                ps = apply_gate_logic_128(ps, q, |st, q| st.apply_x(q));
            } else {
                let parts: Vec<i64> = instr
                    .split(',')
                    .map(|v| v.trim().parse().unwrap())
                    .collect();
                ps = apply_cx_logic_128(ps, parts[0], parts[1]);
            }
        }
        ps
    }

    #[test]
    fn rejects_identity_and_tiny_count() {
        let id = id_pathsum_logic_64(3);
        assert_eq!(synthesize_pmh_logic_64(id.clone(), 8), "None");
        let mut ps = id_pathsum_logic_64(2);
        ps = apply_cx_logic_64(ps, 0, 1);
        assert_eq!(synthesize_pmh_logic_64(ps, 2), "None");
    }

    #[test]
    fn rejects_phase_and_path_vars() {
        let mut phase = id_pathsum_logic_64(2);
        phase = apply_gate_logic_64(phase, 0, |st, q| st.apply_z(q));
        assert_eq!(synthesize_pmh_logic_64(phase, 8), "None");

        let mut h = id_pathsum_logic_64(2);
        h = apply_gate_logic_64(h, 0, |st, q| st.apply_h(q));
        h = apply_cx_logic_64(h, 0, 1);
        assert_eq!(synthesize_pmh_logic_64(h, 8), "None");
    }

    #[test]
    fn cx_ladder_round_trips() {
        let mut ps = id_pathsum_logic_64(3);
        ps = apply_cx_logic_64(ps, 0, 1);
        ps = apply_cx_logic_64(ps, 1, 2);
        ps = apply_cx_logic_64(ps, 2, 0);
        let fp = state_fingerprint_logic_64(ps.clone());
        let out = synthesize_pmh_logic_64(ps, 1_000);
        assert_ne!(out, "None", "pmh refused a linear CX block");
        assert_eq!(
            state_fingerprint_logic_64(replay_pmh_64(&out, 3)),
            fp,
            "pmh replay drifted (instructions: {out})"
        );
    }

    #[test]
    fn extended_pathsum_random_cx_affine_and_compose() {
        fn lcg(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        }

        // High bits: LCG low bits have period 2^k, so `% 2^k` can emit a
        // repeating self-inverse CX pair and leave the state at I_n.
        fn rand_qubit(rng: &mut u64, n: i64) -> i64 {
            ((lcg(rng) >> 32) as i64).rem_euclid(n)
        }

        fn random_cx_64(n: i64, n_ops: usize, seed: u64) -> PSum64 {
            let mut rng = seed;
            let mut ps = id_pathsum_logic_64(n);
            for _ in 0..n_ops {
                let c = rand_qubit(&mut rng, n);
                let t = rand_qubit(&mut rng, n);
                if c != t {
                    ps = apply_cx_logic_64(ps, c, t);
                }
            }
            if state_fingerprint_logic_64(ps.clone())
                == state_fingerprint_logic_64(id_pathsum_logic_64(n))
            {
                ps = apply_cx_logic_64(ps, 0, 1.min(n - 1));
            }
            ps
        }

        fn random_cx_128(n: i64, n_ops: usize, seed: u64) -> PSum128 {
            let mut rng = seed;
            let mut ps = id_pathsum_logic_128(n);
            for _ in 0..n_ops {
                let c = rand_qubit(&mut rng, n);
                let t = rand_qubit(&mut rng, n);
                if c != t {
                    ps = apply_cx_logic_128(ps, c, t);
                }
            }
            if state_fingerprint_logic_128(ps.clone())
                == state_fingerprint_logic_128(id_pathsum_logic_128(n))
            {
                ps = apply_cx_logic_128(ps, 0, 1.min(n - 1));
            }
            ps
        }

        let mut accepted = 0usize;
        for n in [6i64, 8, 12, 16] {
            let id_fp = state_fingerprint_logic_64(id_pathsum_logic_64(n));
            let mut n_accepted = 0usize;
            for k in 0..32 {
                let ps = random_cx_64(n, (n as usize) * 4, 0xF00D + (n as u64) * 97 + k);
                let fp = state_fingerprint_logic_64(ps.clone());
                assert_ne!(fp, id_fp, "random CX product was identity n={n} k={k}");
                let out = synthesize_pmh_logic_64(ps, 1_000);
                assert_ne!(
                    out, "None",
                    "PMH refused a non-identity linear CX product n={n} k={k}"
                );
                accepted += 1;
                n_accepted += 1;
                assert_eq!(
                    state_fingerprint_logic_64(replay_pmh_64(&out, n)),
                    fp,
                    "64-bit random CX replay drifted n={n} k={k}: {out}"
                );
            }
            eprintln!("pathsum random CX n={n}: accepted={n_accepted}/32");
        }
        assert_eq!(accepted, 128);

        let mut affine_ok = 0usize;
        for k in 0..24 {
            let mut rng = 0xAFFE + k;
            let n = 8i64;
            let mut ps = random_cx_64(n, 24, 0xA11E + k);
            for _ in 0..3 {
                let q = rand_qubit(&mut rng, n);
                ps = apply_gate_logic_64(ps, q, |st, q| st.apply_x(q));
            }
            let fp = state_fingerprint_logic_64(ps.clone());
            let out = synthesize_pmh_logic_64(ps, 1_000);
            assert_ne!(out, "None", "PMH refused an affine CX+X state k={k}");
            affine_ok += 1;
            assert!(
                out.contains("x ") || out.contains("cx ") || out.contains(','),
                "affine n=8 k={k} produced unexpected AST: {out}"
            );
            assert_eq!(
                state_fingerprint_logic_64(replay_pmh_64(&out, n)),
                fp,
                "affine replay drifted k={k}: {out}"
            );
        }
        assert_eq!(affine_ok, 24);

        for k in 0..16 {
            let n = 8i64;
            let a = random_cx_64(n, 16, 0xC0DE + k);
            let b = random_cx_64(n, 16, 0xC0DF + k);
            let out_a = synthesize_pmh_logic_64(a, 1_000);
            let out_b = synthesize_pmh_logic_64(b, 1_000);
            if out_a == "None" || out_b == "None" {
                continue;
            }
            let mut composed = replay_pmh_64(&out_a, n);
            for instr in out_b.split(';') {
                let instr = instr.trim();
                if instr.is_empty() {
                    continue;
                }
                if let Some(args) = instr.strip_prefix("cx ") {
                    let parts: Vec<i64> =
                        args.split(',').map(|v| v.trim().parse().unwrap()).collect();
                    composed = apply_cx_logic_64(composed, parts[0], parts[1]);
                } else {
                    let parts: Vec<i64> = instr
                        .split(',')
                        .map(|v| v.trim().parse().unwrap())
                        .collect();
                    composed = apply_cx_logic_64(composed, parts[0], parts[1]);
                }
            }
            let fp = state_fingerprint_logic_64(composed.clone());
            let out = synthesize_pmh_logic_64(composed, 1_000);
            if out == "None" {
                continue;
            }
            assert_eq!(
                state_fingerprint_logic_64(replay_pmh_64(&out, n)),
                fp,
                "composed PathSum replay drifted k={k}: {out}"
            );
        }

        for &(n, samples, ops) in &[(40i64, 8, 80), (64, 4, 96)] {
            let id_fp = state_fingerprint_logic_128(id_pathsum_logic_128(n));
            for k in 0..samples {
                let ps = random_cx_128(n, ops, 0x4040 + (n as u64) * 11 + k);
                let fp = state_fingerprint_logic_128(ps.clone());
                assert_ne!(
                    fp, id_fp,
                    "128-bit random CX product was identity n={n} k={k}"
                );
                let out = synthesize_pmh_logic_128(ps, 1_000);
                assert_ne!(
                    out, "None",
                    "PMH 128 refused a non-identity linear block n={n} k={k}"
                );
                assert_eq!(
                    state_fingerprint_logic_128(replay_pmh_128(&out, n)),
                    fp,
                    "128-bit PathSum replay drifted n={n} k={k}: {out}"
                );
            }
        }
    }

    /// Linear PathSum whose `out_state[i].variable_mask` is row `i` of `M`.
    fn pathsum_from_matrix_64(matrix: &[u64], n: usize) -> PSum64 {
        let mut st = engine_64::EvaluatedPathSum::new_id(n as u32);
        for i in 0..n {
            st.out_state[i] = engine_64::BooleanPoly::from_mask(matrix[i]);
        }
        PSum64::new(Arc::new(st))
    }

    fn pathsum_from_matrix_128(matrix: &[u128], n: usize) -> PSum128 {
        let mut st = engine_128::EvaluatedPathSum::new_id(n as u32);
        for i in 0..n {
            st.out_state[i] = engine_128::BooleanPoly::from_mask(matrix[i]);
        }
        PSum128::new(Arc::new(st))
    }

    fn apply_cnots_pathsum_64(cnots: &[(usize, usize)], n: i64) -> PSum64 {
        let mut ps = id_pathsum_logic_64(n);
        for &(c, t) in cnots {
            ps = apply_cx_logic_64(ps, c as i64, t as i64);
        }
        ps
    }

    fn apply_cnots_pathsum_128(cnots: &[(usize, usize)], n: i64) -> PSum128 {
        let mut ps = id_pathsum_logic_128(n);
        for &(c, t) in cnots {
            ps = apply_cx_logic_128(ps, c as i64, t as i64);
        }
        ps
    }

    fn masks_64(ps: &PSum64) -> Vec<u64> {
        (0..ps.num_qubits as usize)
            .map(|i| ps.out_state[i].variable_mask)
            .collect()
    }

    fn masks_128(ps: &PSum128) -> Vec<u128> {
        (0..ps.num_qubits as usize)
            .map(|i| ps.out_state[i].variable_mask)
            .collect()
    }

    fn dense_gl_64(n: usize, seed: u64) -> Vec<u64> {
        let mut rng = seed;
        let lcg = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s
        };
        let mut rand_bit = || lcg(&mut rng) & 1 == 1;
        let mut l = vec![0u64; n];
        let mut u = vec![0u64; n];
        for i in 0..n {
            l[i] |= 1u64 << i;
            u[i] |= 1u64 << i;
            for j in 0..i {
                if rand_bit() {
                    l[i] |= 1u64 << j;
                }
            }
            for j in (i + 1)..n {
                if rand_bit() {
                    u[i] |= 1u64 << j;
                }
            }
        }
        let mut m = vec![0u64; n];
        for i in 0..n {
            for j in 0..n {
                if ((l[i] >> j) & 1) == 1 {
                    m[i] ^= u[j];
                }
            }
        }
        m
    }

    fn dense_gl_128(n: usize, seed: u64) -> Vec<u128> {
        let mut rng = seed;
        let lcg = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s
        };
        let mut rand_bit = || lcg(&mut rng) & 1 == 1;
        let mut l = vec![0u128; n];
        let mut u = vec![0u128; n];
        for i in 0..n {
            l[i] |= 1u128 << i;
            u[i] |= 1u128 << i;
            for j in 0..i {
                if rand_bit() {
                    l[i] |= 1u128 << j;
                }
            }
            for j in (i + 1)..n {
                if rand_bit() {
                    u[i] |= 1u128 << j;
                }
            }
        }
        let mut m = vec![0u128; n];
        for i in 0..n {
            for j in 0..n {
                if ((l[i] >> j) & 1) == 1 {
                    m[i] ^= u[j];
                }
            }
        }
        m
    }

    fn permutation_gl_64(n: usize, seed: u64) -> Vec<u64> {
        let mut rng = seed;
        let lcg = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s
        };
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (lcg(&mut rng) as usize) % (i + 1);
            perm.swap(i, j);
        }
        perm.into_iter().map(|src| 1u64 << src).collect()
    }

    fn assert_matrix_pathsum_eq_64(matrix: &[u64], n: usize, label: &str) {
        let id: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
        if matrix == id.as_slice() {
            let ps = pathsum_from_matrix_64(matrix, n);
            assert_eq!(
                synthesize_pmh_logic_64(ps, i64::MAX),
                "None",
                "{label}: identity PathSum must be refused"
            );
            return;
        }

        let cnots = engine_64::synthesize_cnot_matrix(matrix.to_vec(), n)
            .unwrap_or_else(|e| panic!("{label}: matrix PMH failed: {e}"));
        assert_eq!(
            engine_64::apply_cnots_to_identity(&cnots, n),
            matrix,
            "{label}: CX list does not implement M"
        );

        let from_mask = pathsum_from_matrix_64(matrix, n);
        assert_eq!(
            masks_64(&from_mask),
            matrix,
            "{label}: from_mask drifted from M"
        );

        let from_tape = apply_cnots_pathsum_64(&cnots, n as i64);
        assert_eq!(
            masks_64(&from_tape),
            matrix,
            "{label}: PathSum tape masks drifted from M"
        );
        assert_eq!(
            from_mask.into_inner(),
            from_tape.into_inner(),
            "{label}: mask-built PathSum is not Eq to tape-built PathSum"
        );

        let src = pathsum_from_matrix_64(matrix, n);
        let fp = state_fingerprint_logic_64(src.clone());
        let out = synthesize_pmh_logic_64(src.clone(), i64::MAX);
        assert_ne!(
            out, "None",
            "{label}: PathSum PMH refused a non-identity linear state"
        );
        let replayed = replay_pmh_64(&out, n as i64);
        assert_eq!(
            state_fingerprint_logic_64(replayed.clone()),
            fp,
            "{label}: PathSum fingerprint drifted (instructions: {out})"
        );
        assert_eq!(
            src.into_inner(),
            replayed.into_inner(),
            "{label}: PathSum PartialEq failed (the state_union trust base)"
        );
        let replayed = replay_pmh_64(&out, n as i64);
        assert_eq!(
            masks_64(&replayed),
            matrix,
            "{label}: replayed PathSum masks drifted from M"
        );
    }

    fn assert_matrix_pathsum_eq_128(matrix: &[u128], n: usize, label: &str) {
        let cnots = engine_128::synthesize_cnot_matrix(matrix.to_vec(), n)
            .unwrap_or_else(|e| panic!("{label}: matrix PMH failed: {e}"));
        assert_eq!(
            engine_128::apply_cnots_to_identity(&cnots, n),
            matrix,
            "{label}: CX list does not implement M"
        );

        let from_mask = pathsum_from_matrix_128(matrix, n);
        assert_eq!(
            masks_128(&from_mask),
            matrix,
            "{label}: from_mask drifted from M"
        );
        let from_tape = apply_cnots_pathsum_128(&cnots, n as i64);
        assert_eq!(
            from_mask.into_inner(),
            from_tape.into_inner(),
            "{label}: mask-built PathSum is not Eq to tape-built PathSum"
        );

        let src = pathsum_from_matrix_128(matrix, n);
        let fp = state_fingerprint_logic_128(src.clone());
        let out = synthesize_pmh_logic_128(src.clone(), i64::MAX);
        assert_ne!(
            out, "None",
            "{label}: PathSum PMH 128 refused a linear state"
        );
        let replayed = replay_pmh_128(&out, n as i64);
        assert_eq!(
            state_fingerprint_logic_128(replayed.clone()),
            fp,
            "{label}: 128-bit fingerprint drifted (instructions: {out})"
        );
        assert_eq!(
            src.into_inner(),
            replayed.into_inner(),
            "{label}: 128-bit PathSum PartialEq failed"
        );
        let replayed = replay_pmh_128(&out, n as i64);
        assert_eq!(
            masks_128(&replayed),
            matrix,
            "{label}: replayed 128-bit masks drifted from M"
        );
    }

    #[test]
    fn matrix_suites_merge_with_pathsum_eq() {
        // The matrix suites only checked "CX list implements M". This takes
        // those same ensembles through PathSum: mask-built state Eq tape-built
        // state, and synthesize_pmh replay is PartialEq (the state_union base).
        for &(n, samples) in &[(8usize, 64), (12, 64), (16, 64), (20, 32), (24, 32)] {
            for k in 0..samples {
                let m = dense_gl_64(n, 0xE7E7 + (n as u64) * 100_003 + k);
                assert_matrix_pathsum_eq_64(&m, n, &format!("dense n={n} k={k}"));
            }
        }

        for n in [8usize, 12, 16] {
            for k in 0..16 {
                let m = permutation_gl_64(n, 0xA0A0 + (n as u64) * 31 + k);
                assert_matrix_pathsum_eq_64(&m, n, &format!("perm n={n} k={k}"));
            }
        }

        for k in 0..24 {
            let n = 8usize;
            let m = dense_gl_64(n, 0xA11E + k);
            let cnots = engine_64::synthesize_cnot_matrix(m.clone(), n).unwrap();
            let mut ps = apply_cnots_pathsum_64(&cnots, n as i64);
            ps = apply_gate_logic_64(ps, (k % n as u64) as i64, |st, q| st.apply_x(q));
            let fp = state_fingerprint_logic_64(ps.clone());
            let out = synthesize_pmh_logic_64(ps.clone(), i64::MAX);
            assert_ne!(out, "None", "affine n=8 k={k} refused");
            assert!(
                out.contains("x "),
                "affine n=8 k={k} missing trailing X: {out}"
            );
            let replayed = replay_pmh_64(&out, n as i64);
            assert_eq!(
                state_fingerprint_logic_64(replayed.clone()),
                fp,
                "affine fingerprint drifted k={k}: {out}"
            );
            assert_eq!(
                ps.into_inner(),
                replayed.into_inner(),
                "affine PathSum PartialEq failed k={k}"
            );
            let replayed = replay_pmh_64(&out, n as i64);
            assert_eq!(masks_64(&replayed), m, "affine linear masks drifted k={k}");
        }

        for &(n, samples) in &[(16usize, 16), (40, 8), (70, 4)] {
            for k in 0..samples {
                let m = dense_gl_128(n, 0x7070 + (n as u64) * 1009 + k);
                assert_matrix_pathsum_eq_128(&m, n, &format!("dense128 n={n} k={k}"));
            }
        }
    }

    #[test]
    fn wide_n70_pathsum_round_trips() {
        let mut ps = id_pathsum_logic_128(70);
        ps = apply_cx_logic_128(ps, 0, 65);
        ps = apply_cx_logic_128(ps, 65, 69);
        ps = apply_cx_logic_128(ps, 69, 0);
        let fp = state_fingerprint_logic_128(ps.clone());
        let out = synthesize_pmh_logic_128(ps, 1_000);
        assert_ne!(out, "None", "pmh 128 refused a high-wire linear block");
        assert_eq!(
            state_fingerprint_logic_128(replay_pmh_128(&out, 70)),
            fp,
            "pmh 128 replay drifted (instructions: {out})"
        );
    }
}
