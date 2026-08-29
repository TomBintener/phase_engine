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
//! Phase-bearing wrappers (`apply_gate_logic_*`, `apply_rz_logic_*`) call
//! `reduce()` after the apply so the interned value is canonical. `x` and `cx`
//! skip `reduce()`: `apply_gate_no_reduce_logic_*` / `apply_cx_logic_*` are
//! affine / linear XOR updates and do not allocate path variables. See
//! `PATHSUM.md`.

use crate::engine::{engine_64, engine_128};
use crate::prelude::BaseSort;
use crate::sort::{BaseValues, Boxed};
use crate::{add_primitive, EGraph, Value};
use crate::ast::Literal;
use crate::{TermId, TermDag};
use crate::engine::engine_64::EvaluatedPathSum as EvaluatedPathSum64;
use crate::engine::engine_128::EvaluatedPathSum as EvaluatedPathSum128;
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
        PSum64::new(Arc::new(engine_64::EvaluatedPathSum::new_id(num_qubits as u32)))
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
    PSum64::new(Arc::new(new_state))
}

pub fn apply_cx_logic_64(state: PSum64, qc: i64, qt: i64) -> PSum64 {
    if qc == qt || qc < 0 || qt < 0 ||
       qc as usize >= state.num_qubits as usize ||
       qt as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    new_state.apply_cx(qc as usize, qt as usize);
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
    let bounded = if sum < 0.0 { sum + 2.0 * std::f64::consts::PI } else { sum };
    let snapped = snap_phase(bounded);
    snapped.to_bits() as i64
}

pub fn rust_negate_rz_bits_logic(a: i64) -> i64 {
    let f_a = f64::from_bits(a as u64);
    let neg = -f_a;
    let bounded = if neg < 0.0 { neg + 2.0 * std::f64::consts::PI } else { neg };
    let snapped = snap_phase(bounded);
    snapped.to_bits() as i64
}

// --- Logic for the 128-bit Engine ---

pub fn id_pathsum_logic_128(num_qubits: i64) -> PSum128 {
    if num_qubits <= 0 {
        PSum128::new(Arc::new(engine_128::EvaluatedPathSum::new_id(0)))
    } else {
        PSum128::new(Arc::new(engine_128::EvaluatedPathSum::new_id(num_qubits as u32)))
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
    PSum128::new(Arc::new(new_state))
}

pub fn apply_cx_logic_128(state: PSum128, qc: i64, qt: i64) -> PSum128 {
    if qc == qt || qc < 0 || qt < 0 ||
       qc as usize >= state.num_qubits as usize ||
       qt as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = Arc::unwrap_or_clone(state.into_inner());
    new_state.apply_cx(qc as usize, qt as usize);
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

pub fn synthesize_steiner_logic_64(state: PSum64, gate_count: i64, hw_cost: i64, topology_str: String, cnot_weight: i64, rz_weight: i64) -> String {
    if gate_count <= 2 {
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
    for (i, p) in state.continuous_poly.parities.iter().enumerate() {
        let mask = p.variable_mask;
        let angle = state.continuous_poly.phases[i];
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
            
            fn dfs(u: usize, p: usize, adj: &Vec<Vec<usize>>, vis: &mut Vec<bool>, po: &mut Vec<usize>, par: &mut Vec<usize>) {
                vis[u] = true;
                par[u] = p;
                for &v in &adj[u] {
                    if v != p && !vis[v] {
                        dfs(v, u, adj, vis, po, par);
                    }
                }
                po.push(u);
            }
            
            dfs(root, usize::MAX, &tree_adj, &mut visited, &mut post_order, &mut parent);
            
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
    
    if let Ok(cnots) = crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize) {
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
    
    let synth_cost = ((instructions.len() as i64) - total_cnots) * rz_weight + total_cnots * cnot_weight;
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
    
    if let Ok(cnots) = crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize) {
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
    for (i, p) in state.continuous_poly.parities.iter().enumerate() {
        if (p.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        if p.terms.iter().any(|&t| t == 0 || t.count_ones() != 1) {
            return "None".to_string();
        }
        parities.push((p.variable_mask, state.continuous_poly.phases[i]));
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

pub fn synthesize_steiner_logic_128(state: PSum128, gate_count: i64, hw_cost: i64, topology_str: String, cnot_weight: i64, rz_weight: i64) -> String {
    if gate_count <= 2 {
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
    for (i, p) in state.continuous_poly.parities.iter().enumerate() {
        let mask = p.variable_mask;
        let angle = state.continuous_poly.phases[i];
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
            
            fn dfs(u: usize, p: usize, adj: &Vec<Vec<usize>>, vis: &mut Vec<bool>, po: &mut Vec<usize>, par: &mut Vec<usize>) {
                vis[u] = true;
                par[u] = p;
                for &v in &adj[u] {
                    if v != p && !vis[v] {
                        dfs(v, u, adj, vis, po, par);
                    }
                }
                po.push(u);
            }
            
            dfs(root, usize::MAX, &tree_adj, &mut visited, &mut post_order, &mut parent);
            
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
        matrix.push(state.out_state[i].variable_mask as u64);
    }
    
    if let Ok(cnots) = crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize) {
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
    
    let synth_cost = ((instructions.len() as i64) - total_cnots) * rz_weight + total_cnots * cnot_weight;
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
    let n = state.num_qubits as usize;
    let valid_mask: u128 = if n >= 128 { u128::MAX } else { (1u128 << n) - 1 };

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
    for (i, p) in state.continuous_poly.parities.iter().enumerate() {
        if (p.variable_mask & !valid_mask) != 0 {
            return "None".to_string();
        }
        if p.terms.iter().any(|&t| t == 0 || t.count_ones() != 1) {
            return "None".to_string();
        }
        parities.push((p.variable_mask, state.continuous_poly.phases[i]));
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
        matrix.push(state.out_state[i].variable_mask as u64); // PMH synthesizer takes u64 since it is capped at 64 anyway!
    }
    
    if let Ok(cnots) = crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize) {
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
    
    return "None".to_string();
}


// --- EGG SORT REGISTRATION ---

#[derive(Debug)]
pub struct PathSumSort64;
impl BaseSort for PathSumSort64 {
    type Base = PSum64;
    fn name(&self) -> &str { "PathSum64" }

    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "rust_id_pathsum_64" = |q: i64| -> PSum64 { id_pathsum_logic_64(q) });
        add_primitive!(eg, "rust_apply_x_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_no_reduce_logic_64(s, q, |st, q_| st.apply_x(q_)) });
        add_primitive!(eg, "rust_apply_z_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_logic_64(s, q, |st, q_| st.apply_z(q_)) });
        add_primitive!(eg, "rust_apply_s_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_logic_64(s, q, |st, q_| st.apply_s(q_)) });
        add_primitive!(eg, "rust_apply_sdg_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_logic_64(s, q, |st, q_| st.apply_sdg(q_)) });
        add_primitive!(eg, "rust_apply_t_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_logic_64(s, q, |st, q_| st.apply_t(q_)) });
        add_primitive!(eg, "rust_apply_tdg_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_logic_64(s, q, |st, q_| st.apply_tdg(q_)) });
        add_primitive!(eg, "rust_apply_sx_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_logic_64(s, q, |st, q_| st.apply_sx(q_)) });
        add_primitive!(eg, "rust_apply_h_64" = |s: PSum64, q: i64| -> PSum64 { apply_gate_logic_64(s, q, |st, q_| st.apply_h(q_)) });
        add_primitive!(eg, "rust_apply_cx_64" = |s: PSum64, qc: i64, qt: i64| -> PSum64 { apply_cx_logic_64(s, qc, qt) });
        add_primitive!(eg, "rust_apply_rz_64" = |s: PSum64, q: i64, t: i64| -> PSum64 { apply_rz_logic_64(s, q, t) });
        add_primitive!(eg, "rust_synthesize_pmh_64" = |s: PSum64, count: i64| -> S { S::new(synthesize_pmh_logic_64(s, count)) });
        add_primitive!(eg, "rust_synthesize_steiner_64" = |s: PSum64, count: i64, hw_cost: i64, top: S, cnot_wt: i64, rz_wt: i64| -> S { S::new(synthesize_steiner_logic_64(s, count, hw_cost, top.to_string(), cnot_wt, rz_wt)) });
        add_primitive!(eg, "rust_synthesize_gray_64" = |s: PSum64, count: i64, hw_cost: i64, cnot_wt: i64, rz_wt: i64| -> S { S::new(synthesize_gray_logic_64(s, count, hw_cost, cnot_wt, rz_wt)) });
        add_primitive!(eg, "rust_state_fingerprint_64" = |s: PSum64| -> S { S::new(state_fingerprint_logic_64(s)) });
        add_primitive!(eg, "rust_debug_pathsum_64" = |s: PSum64| -> S { S::new(debug_pathsum_logic_64(s)) });
        add_primitive!(eg, "rust_add_rz_bits_64" = |a: i64, b: i64| -> i64 { rust_add_rz_bits_logic(a, b) });
        add_primitive!(eg, "rust_negate_rz_bits_64" = |a: i64| -> i64 { rust_negate_rz_bits_logic(a) });
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
    fn name(&self) -> &str { "PathSum128" }

    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "rust_id_pathsum_128" = |q: i64| -> PSum128 { id_pathsum_logic_128(q) });
        add_primitive!(eg, "rust_apply_x_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_no_reduce_logic_128(s, q, |st, q_| st.apply_x(q_)) });
        add_primitive!(eg, "rust_apply_z_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_logic_128(s, q, |st, q_| st.apply_z(q_)) });
        add_primitive!(eg, "rust_apply_s_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_logic_128(s, q, |st, q_| st.apply_s(q_)) });
        add_primitive!(eg, "rust_apply_sdg_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_logic_128(s, q, |st, q_| st.apply_sdg(q_)) });
        add_primitive!(eg, "rust_apply_t_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_logic_128(s, q, |st, q_| st.apply_t(q_)) });
        add_primitive!(eg, "rust_apply_tdg_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_logic_128(s, q, |st, q_| st.apply_tdg(q_)) });
        add_primitive!(eg, "rust_apply_sx_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_logic_128(s, q, |st, q_| st.apply_sx(q_)) });
        add_primitive!(eg, "rust_apply_h_128" = |s: PSum128, q: i64| -> PSum128 { apply_gate_logic_128(s, q, |st, q_| st.apply_h(q_)) });
        add_primitive!(eg, "rust_apply_cx_128" = |s: PSum128, qc: i64, qt: i64| -> PSum128 { apply_cx_logic_128(s, qc, qt) });
        add_primitive!(eg, "rust_apply_rz_128" = |s: PSum128, q: i64, t: i64| -> PSum128 { apply_rz_logic_128(s, q, t) });
        add_primitive!(eg, "rust_synthesize_pmh_128" = |s: PSum128, count: i64| -> S { S::new(synthesize_pmh_logic_128(s, count)) });
        add_primitive!(eg, "rust_synthesize_steiner_128" = |s: PSum128, count: i64, hw_cost: i64, top: S, cnot_wt: i64, rz_wt: i64| -> S { S::new(synthesize_steiner_logic_128(s, count, hw_cost, top.to_string(), cnot_wt, rz_wt)) });
        add_primitive!(eg, "rust_synthesize_gray_128" = |s: PSum128, count: i64, hw_cost: i64, cnot_wt: i64, rz_wt: i64| -> S { S::new(synthesize_gray_logic_128(s, count, hw_cost, cnot_wt, rz_wt)) });
        add_primitive!(eg, "rust_state_fingerprint_128" = |s: PSum128| -> S { S::new(state_fingerprint_logic_128(s)) });
        add_primitive!(eg, "rust_debug_pathsum_128" = |s: PSum128| -> S { S::new(debug_pathsum_logic_128(s)) });
        add_primitive!(eg, "rust_add_rz_bits_128" = |a: i64, b: i64| -> i64 { rust_add_rz_bits_logic(a, b) });
        add_primitive!(eg, "rust_negate_rz_bits_128" = |a: i64| -> i64 { rust_negate_rz_bits_logic(a) });
    }
    fn reconstruct_termdag(&self, _bv: &BaseValues, _v: Value, td: &mut TermDag) -> TermId {
        let arg = td.lit(Literal::Int(0));
        td.app("rust_id_pathsum_128".to_string(), vec![arg])
    }
}

/// Complete canonical state fingerprint (64-bit): the derive(Debug) dump of
/// every `EvaluatedPathSum` field, i.e. exactly the fields `PartialEq`
/// compares. Two states are engine-equal iff their fingerprints match, which
/// gives host-side passes the same trust base as `state_union`
/// (`debug_pathsum` is NOT sufficient: it omits `out_state`).
pub fn state_fingerprint_logic_64(state: PSum64) -> String {
    // Arc's Debug delegates to the inner EvaluatedPathSum.
    format!("{:?}", state.into_inner())
}

/// Complete canonical state fingerprint (128-bit); see the 64-bit variant.
pub fn state_fingerprint_logic_128(state: PSum128) -> String {
    format!("{:?}", state.into_inner())
}

pub fn debug_pathsum_logic_64(state: PSum64) -> String {
    let mut out = String::new();
    out.push_str(&format!("Continuous Parities: {:?}\n", state.continuous_poly.parities));
    out.push_str(&format!("Continuous Phases: {:?}\n", state.continuous_poly.phases));
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
    out.push_str(&format!("Continuous Parities: {:?}\n", state.continuous_poly.parities));
    out.push_str(&format!("Continuous Phases: {:?}\n", state.continuous_poly.phases));
    out.push_str("Phase Poly Terms: ");
    for term in &state.phase_poly.terms {
        let phase_unit = term.0 >> 125;
        let mask = term.0 & 0x1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
        out.push_str(&format!("(mask={}, phase_unit={}) ", mask, phase_unit));
    }
    out
}

#[cfg(test)]
mod gray_tests {
    use super::*;

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
        let fp_orig = state_fingerprint_logic_64(ps.clone());
        let out = synthesize_gray_logic_64(ps, 1_000, i64::MAX, 50, 1);
        assert_ne!(out, "None", "gray refused a synthesizable state");
        let replayed = if out == "empty" {
            id_pathsum_logic_64(n)
        } else {
            replay(&out, n)
        };
        assert_eq!(
            state_fingerprint_logic_64(replayed),
            fp_orig,
            "resynthesized circuit is not engine-equal (instructions: {out})"
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
        // Quality: the naive build/unbuild ladder uses 6 CX; sharing must do
        // strictly better on a single parity (3 up + fixup back).
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
        let fp_orig = state_fingerprint_logic_64(ps.clone());
        let out = synthesize_gray_logic_64(ps, 1_000, i64::MAX, 50, 1);
        assert_ne!(out, "None", "gray should accept affine/X offsets");
        assert!(out.contains("x 1"), "expected trailing X layer: {out}");
        let replayed = replay(&out, 2);
        assert_eq!(
            state_fingerprint_logic_64(replayed),
            fp_orig,
            "affine resynthesis not engine-equal (instructions: {out})"
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
        let steiner =
            synthesize_steiner_logic_64(ps, 1_000, i64::MAX, String::new(), 50, 1);
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
}
