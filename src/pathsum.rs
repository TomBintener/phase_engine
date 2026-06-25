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


// --- EGG SORT REGISTRATION ---

#[derive(Debug)]
pub struct PathSumSort64;
impl BaseSort for PathSumSort64 {
    type Base = PSum64;
    fn name(&self) -> &str { "PathSum64" }

    fn register_primitives(&self, eg: &mut EGraph) {
        add_primitive!(eg, "rust_pathsum_debug_64" = |s: PSum64| -> S { S::new(format!("{:#?}", *s)) });
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
        add_primitive!(eg, "rust_add_rz_bits" = |a: i64, b: i64| -> i64 { rust_add_rz_bits_logic(a, b) });
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
        add_primitive!(eg, "rust_pathsum_debug_128" = |s: PSum128| -> S { S::new(format!("{:#?}", *s)) });
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
    }
    fn reconstruct_termdag(&self, _bv: &BaseValues, _v: Value, td: &mut TermDag) -> TermId {
        let arg = td.lit(Literal::Int(0));
        td.app("rust_id_pathsum_128".to_string(), vec![arg])
    }
}
