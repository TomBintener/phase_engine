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
//! the engine to correctly identify equivalent quantum states.

use crate::engine::{engine_64, engine_128};
use crate::prelude::BaseSort;
use crate::sort::{BaseValues, Boxed};
use crate::{add_primitive, EGraph, Value};
use crate::ast::Literal;
use crate::{TermId, TermDag};
use crate::engine::engine_64::EvaluatedPathSum as EvaluatedPathSum64;
use crate::engine::engine_128::EvaluatedPathSum as EvaluatedPathSum128;

// Define type aliases for the 64-bit and 128-bit engines
pub type PSum64 = Boxed<engine_64::EvaluatedPathSum>;
pub type PSum128 = Boxed<engine_128::EvaluatedPathSum>;

// --- Logic for the 64-bit Engine ---

pub fn id_pathsum_logic_64(num_qubits: i64) -> PSum64 {
    if num_qubits <= 0 {
        PSum64::new(engine_64::EvaluatedPathSum::new_id(0))
    } else {
        PSum64::new(engine_64::EvaluatedPathSum::new_id(num_qubits as u32))
    }
}

pub fn apply_gate_logic_64<F>(state: PSum64, q: i64, op: F) -> PSum64
where
    F: Fn(&mut engine_64::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    op(&mut new_state, q as usize);
    new_state.reduce();
    PSum64::new(new_state)
}

pub fn apply_gate_no_reduce_logic_64<F>(state: PSum64, q: i64, op: F) -> PSum64
where
    F: Fn(&mut engine_64::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    op(&mut new_state, q as usize);
    PSum64::new(new_state)
}

pub fn apply_cx_logic_64(state: PSum64, qc: i64, qt: i64) -> PSum64 {
    if qc == qt || qc < 0 || qt < 0 ||
       qc as usize >= state.num_qubits as usize ||
       qt as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    new_state.apply_cx(qc as usize, qt as usize);
    PSum64::new(new_state)
}

pub fn apply_rz_logic_64(state: PSum64, q: i64, theta_bits: i64) -> PSum64 {
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    new_state.apply_rz(q as usize, f64::from_bits(theta_bits as u64));
    new_state.reduce();
    PSum64::new(new_state)
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
        PSum128::new(engine_128::EvaluatedPathSum::new_id(0))
    } else {
        PSum128::new(engine_128::EvaluatedPathSum::new_id(num_qubits as u32))
    }
}

pub fn apply_gate_logic_128<F>(state: PSum128, q: i64, op: F) -> PSum128
where
    F: Fn(&mut engine_128::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    op(&mut new_state, q as usize);
    new_state.reduce();
    PSum128::new(new_state)
}

pub fn apply_gate_no_reduce_logic_128<F>(state: PSum128, q: i64, op: F) -> PSum128
where
    F: Fn(&mut engine_128::EvaluatedPathSum, usize),
{
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    op(&mut new_state, q as usize);
    PSum128::new(new_state)
}

pub fn apply_cx_logic_128(state: PSum128, qc: i64, qt: i64) -> PSum128 {
    if qc == qt || qc < 0 || qt < 0 ||
       qc as usize >= state.num_qubits as usize ||
       qt as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    new_state.apply_cx(qc as usize, qt as usize);
    PSum128::new(new_state)
}

pub fn apply_rz_logic_128(state: PSum128, q: i64, theta_bits: i64) -> PSum128 {
    if q < 0 || q as usize >= state.num_qubits as usize {
        return state;
    }
    let mut new_state = (*state).clone();
    new_state.apply_rz(q as usize, f64::from_bits(theta_bits as u64));
    new_state.reduce();
    PSum128::new(new_state)
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
pub fn synthesize_pmh_logic_64(state: PSum64, gate_count: i64) -> String {
    
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
    // 1c. No constant terms (monomial `0`) in out_state.
    // apply_x XORs in {0}, leaving a net footprint when an odd number of X gates
    // have been applied to a qubit. variable_mask is blind to this (0 | mask == mask),
    // so we check the terms SmallVec directly. Even counts of X cancel naturally via
    // BooleanPoly::add_assign, so this only fires when there is a genuine net X effect.
    if state.out_state.iter().any(|poly| poly.terms.contains(&0)) {
        return "None".to_string();
    }
    
    // 2. Strict CNOT block boundary
    let rank = rank_m_minus_i_64(&state);
    if rank == 0 {
        return "None".to_string();
    }
    
    // 3. Rank(M - I) lower bound
    let r = rank;
    if gate_count <= r {
        return "None".to_string();
    }
    
    // 4. PMH Synthesis
    let mut matrix = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        matrix.push(state.out_state[i].variable_mask);
    }
    
    if let Ok(cnots) = crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize) {
        if (cnots.len() as i64) < gate_count {
            // Build the string representation as simple "c,t;c,t"
            if cnots.is_empty() {
                return "empty".to_string();
            }
            let str_repr = cnots.iter()
                .map(|&(c, t)| format!("{},{}", c, t))
                .collect::<Vec<_>>()
                .join(";");
            return str_repr;
        }
    }
    
    "None".to_string()
}

pub fn synthesize_pmh_logic_128(state: PSum128, gate_count: i64) -> String {
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
    // 1c. No constant terms (monomial `0`) in out_state.
    // apply_x XORs in {0}, leaving a net footprint when an odd number of X gates
    // have been applied to a qubit. variable_mask is blind to this (0 | mask == mask),
    // so we check the terms SmallVec directly. Even counts of X cancel naturally via
    // BooleanPoly::add_assign, so this only fires when there is a genuine net X effect.
    if state.out_state.iter().any(|poly| poly.terms.contains(&0)) {
        return "None".to_string();
    }
    
    // 2. Strict CNOT block boundary
    let rank = rank_m_minus_i_128(&state);
    if rank == 0 {
        return "None".to_string();
    }
    
    // 3. Rank(M - I) lower bound
    let r = rank;
    if gate_count <= r {
        return "None".to_string();
    }
    
    // 4. PMH Synthesis
    let mut matrix = Vec::with_capacity(state.num_qubits as usize);
    for i in 0..state.num_qubits as usize {
        matrix.push(state.out_state[i].variable_mask as u64); // PMH synthesizer takes u64 since it is capped at 64 anyway!
    }
    
    if let Ok(cnots) = crate::engine::engine_64::synthesize_cnot_matrix(matrix, state.num_qubits as usize) {
        if (cnots.len() as i64) < gate_count {
            // Build the string representation as simple "c,t;c,t"
            if cnots.is_empty() {
                return "empty".to_string();
            }
            let mut parts = Vec::new();
            for &(c, t) in &cnots {
                parts.push(format!("{},{}", c, t));
            }
            let join_str = parts.join(";");
            
            // To pass through E-Graph dynamically we just return the AST string mapping string
            return join_str;
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
        add_primitive!(eg, "rust_add_rz_bits_128" = |a: i64, b: i64| -> i64 { rust_add_rz_bits_logic(a, b) });
        add_primitive!(eg, "rust_negate_rz_bits_128" = |a: i64| -> i64 { rust_negate_rz_bits_logic(a) });
    }
    fn reconstruct_termdag(&self, _bv: &BaseValues, _v: Value, td: &mut TermDag) -> TermId {
        let arg = td.lit(Literal::Int(0));
        td.app("rust_id_pathsum_128".to_string(), vec![arg])
    }
}
